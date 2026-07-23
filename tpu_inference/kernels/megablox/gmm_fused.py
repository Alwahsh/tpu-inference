# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused MoE FFN kernel: GMM1 + activation + GMM2 in one pipeline.

Fuses the two grouped matmuls of an MoE FFN layer,

    out = gmm_v2(act(gmm_v2(lhs, w1, fuse_act)), w2)

into a single Pallas TPU kernel built from gmm_v2's building blocks: one
emit_pipeline over gm (row) tiles where each grid step performs the full
per-tile dataflow

    lhs rows -> GMM1 (gate/up, postscale) -> act -> cast to the bridge
             dtype -> dynamic lhs requant -> GMM2 -> out rows

with the [tile_m, inter] intermediate never leaving VMEM/vregs.

Numerics contract: BITWISE-equal to the sequential composition

    mid = gmm_v2(lhs, w1, group_sizes, rhs_scale=w1_scale, fuse_act=...)
    out = gmm_v2(mid, w2, group_sizes, rhs_scale=w2_scale)

whenever the sequential kernels run with a single k/n tile per matmul
(their accumulation order then matches the fused single-tile contract).
This holds because (a) both matmuls call gmm_v2.matmul_tile, the exact op
sequence of the sequential kernel; (b) the activation output is masked and
cast to the sequential mid dtype BEFORE requantization — the "bitwise
bridge" — so GMM2 quantizes exactly the values the sequential GMM2 would
read back from HBM; and (c) the boundary-row argument documented on
fused_inner_kernel.

Single-tile contract (asserted): tile_k1 = hidden, tile_n1 = inter (per
gate/up half), tile_k2 = inter, tile_n2 = hidden, i.e. num_k = num_n = 1
for both matmuls, grid = (1, num_gm, 1).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.megablox import gmm_v2

# Pipeline buffer counts, for the VMEM estimate only. These mirror the
# buffering gmm_v2.generate_block_specs sets up (lhs/out double-buffered,
# weights triple-buffered via pl.Buffered) and exist here so the estimate
# stays in sync if those counts ever change.
_LHS_BUFFER_COUNT = 2
_WEIGHT_BUFFER_COUNT = 3
_OUT_BUFFER_COUNT = 2

# Zero-fill scratch target, same heuristic as gmm_v2.
_ZERO_REF_TARGET_BYTES = 2 * 1024 * 1024


def fused_inner_kernel(
        # In
        tiled_lhs_ref: jax.Array,
        # [tile_m // sublane, sublane, hidden]
        w1_ref: gmm_v2.
    RhsRef,  # gmm_v2.FusedWeightsRef: gate/up [hidden, inter] (+ scales)
        w2_ref: gmm_v2.RhsRef,  # gmm_v2.WeightsRef: [inter, hidden] (+ scale)
        # Out
    tiled_out_ref: jax.Array,  # [tile_m // sublane, sublane, hidden]
        # Scratch
    partial_out_ref: jax.Array,  # [sublane, hidden]
        metadata_ref: gmm_v2.MetadataRef,
        *,
        cfgs1: gmm_v2.GmmConfigs,  # GMM1 (gate/up) config, fuse_act set
        cfgs2: gmm_v2.GmmConfigs,  # GMM2 (down) config, fuse_act None
):
    """Per-gm-tile fused FFN body: GMM1 -> act -> bridge -> GMM2 -> store.

    Both matmuls go through gmm_v2.matmul_tile — the single MXU-body call
    site — so each stage performs the exact op sequence of the sequential
    kernel.

    Correctness of per-tile fusion at sublane-boundary rows
    -------------------------------------------------------
    Groups partition rows: every row belongs to exactly ONE group. What
    adjacent gm tiles share is a sublane BLOCK, never a row: when a group
    boundary falls inside a sublane, both tiles' row windows cover that
    sublane, each owning a disjoint subset of its rows.

    Sequential path, per row r owned by group g:
      * GMM1's tile for g computes mid[r] from lhs[r] and w1[g]; rows of
        the shared sublane owned by the OTHER group are masked to 0 in
        this tile's store, and the true values land via the adjacent
        tile's store plus the partial_out accumulation (x + 0 adds, exact
        in bf16). So the finalized HBM mid holds exactly the per-row
        value each owning tile computed.
      * GMM2's tile for g loads that sublane back, requantizes PER ROW
        (block abs-max over the row's k-blocks — no cross-row coupling),
        matmuls with w2[g] (row-wise: out[r] depends only on mid[r]),
        masks non-owned rows to 0, and partial_out combines tiles again.

    Fused path, same row r: this tile computes the identical mid[r]
    (same lhs rows via the same index maps/metadata, same gmm_v2.matmul_tile op
    sequence, same act), masks the SAME non-owned rows to 0 (before the
    bridge, so GMM2 sees 0 there: quant(0) = 0, 0 @ w2 = 0, masked to 0
    again — identical to sequential's masked garbage rows), casts to
    cfgs1.out_dtype exactly as sequential's HBM store would, and
    requantizes per row. Since every op after the accumulator is
    row-wise, in-group rows are bitwise-identical to sequential, and
    non-owned rows contribute exactly 0 — the final gmm_v2.store_output_tile
    (reused verbatim) then reconstructs boundary sublanes through the
    same partial_out adds as sequential GMM2.

    The one non-row-wise subtlety is the sequential mid's `x + 0`
    partial adds (they could flip -0.0 to +0.0 on boundary rows, which
    the fused path skips). This cannot change results: a ±0 lhs element
    quantizes to ±0, its products are ±0, and IEEE round-to-nearest
    addition into the (+0-initialized) accumulator gives +0 either way;
    abs() kills the sign in the quant scale. So sign-of-zero differences
    are annihilated before anything observable.

    Masking order matches gmm_v2 (act THEN mask): jnp.where selects, so
    act garbage/NaNs on non-owned rows never propagate.
    """
    sublane = cfgs1.dims.size_lhs_sublane
    gm_id = pl.program_id(1)
    m_start_local, m_end_local = gmm_v2.compute_local_row_bounds(
        metadata_ref, gm_id, sublane)

    # GMM1: [tile_m, hidden] x [hidden, 2 * inter] -> [tile_m, 2 * inter]
    # (lane-interleaved gate/up columns). The lhs is dynamically quantized
    # per row per block inside gmm_v2.matmul_tile, exactly like gmm_v2.
    tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs1.tiles.tile_k)[...]
    acc1 = gmm_v2.matmul_tile(tiled_lhs,
                              w1_ref,
                              cfgs=cfgs1,
                              is_last_k_step=True)

    # Activation: [tile_m, 2 * inter] -> [tile_m, inter].
    act = gmm_v2.apply_act_fn(acc1, cfgs1.fuse_act)

    # Mask rows not owned by this tile's group BEFORE the bridge: their
    # acc1/act values are garbage (wrong expert), and GMM2 must consume 0
    # there (see docstring).
    mid = gmm_v2.mask_out_of_group_rows(act, m_start_local, m_end_local)

    # BITWISE BRIDGE: cast to the dtype the sequential path materializes
    # in HBM between the two kernels (gmm_v2's out_dtype) BEFORE
    # requantizing, so GMM2's dynamic quant sees exactly the sequential
    # values.
    mid = mid.astype(cfgs1.out_dtype)

    # GMM2: [tile_m, inter] x [inter, hidden] -> [tile_m, hidden]. The
    # same gmm_v2.matmul_tile requantizes mid per row per k-block, exactly like
    # sequential GMM2 quantizes its lhs.
    acc2 = gmm_v2.matmul_tile(mid, w2_ref, cfgs=cfgs2, is_last_k_step=True)

    # Epilogue: gmm_v2's masking + sublane-boundary partial accumulation,
    # applied to the FINAL output only.
    acc2_masked = gmm_v2.mask_out_of_group_rows(acc2, m_start_local,
                                                m_end_local)
    gmm_v2.store_output_tile(acc2_masked,
                             tiled_out_ref,
                             partial_out_ref,
                             gm_id,
                             m_end_local,
                             sublane=sublane)


def _run_fused_pipeline(
    num_gm: jax.Array,
    metadata_ref: gmm_v2.MetadataRef,
    lhs_ref: jax.Array,
    w1_ref: gmm_v2.WeightsRef,
    w2_ref: gmm_v2.WeightsRef,
    out_ref: jax.Array,
    partial_out_ref: jax.Array,
    zero_ref: jax.Array | None,
    semaphore_ref: jax.Array | None,
    *,
    cfgs1: gmm_v2.GmmConfigs,
    cfgs2: gmm_v2.GmmConfigs,
):
    """Zero-init, block specs and the fused software pipeline."""
    sublane = cfgs1.dims.size_lhs_sublane

    if cfgs2.zero_init:
        zero_size = gmm_v2.zero_out_start(
            out_ref,
            zero_ref,
            semaphore_ref,
            metadata_ref,
            num_gm,
            dims=cfgs2.dims,
        )

    # lhs spec + w1 spec come from the GMM1 config, w2 spec + out spec
    # from the GMM2 config; both share the same metadata/row windows.
    (lhs_spec, w1_spec), _ = gmm_v2.generate_block_specs(metadata_ref, cfgs1)
    (_, w2_spec), out_spec = gmm_v2.generate_block_specs(metadata_ref, cfgs2)

    # Split w1 into gate/up refs, same as kernel_main's fuse_act path
    # (w1 layout [g, hidden, 2 * inter]: gate = [..., :inter],
    # up = [..., inter:]).
    w1_up_ref = jax.tree.map(lambda x: x.at[..., cfgs1.out_size_n:], w1_ref)
    w1_ref = gmm_v2.FusedWeightsRef(gate=w1_ref, up=w1_up_ref)
    w1_spec = gmm_v2.FusedWeightsRef(gate=w1_spec, up=w1_spec)

    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(fused_inner_kernel, cfgs1=cfgs1, cfgs2=cfgs2),
        grid=(1, num_gm, 1),
        in_specs=(lhs_spec, w1_spec, w2_spec),
        out_specs=out_spec,
    )

    # Bounded slice requires second last dim to be aligned to the sublane
    # size. Weight refs use static tiling thus reshape is not needed.
    lhs_in = lhs_ref.reshape(-1, sublane, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, sublane, out_ref.shape[-1])
    scratches = [partial_out_ref, metadata_ref]
    pipeline_fn(lhs_in, w1_ref, w2_ref, out_in, scratches=scratches)

    if cfgs2.zero_init:
        gmm_v2.zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs2.dims)


def fused_kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    # In
    lhs_ref: jax.Array,  # [size_m, hidden]
    w1_ref: gmm_v2.WeightsRef,  # [size_group, hidden, 2 * inter] (+ scale)
    w2_ref: gmm_v2.WeightsRef,  # [size_group, inter, hidden] (+ scale)
    # Out
    out_ref: jax.Array,  # [size_m, hidden]
    # Scratch memory
    partial_out_ref: jax.Array,  # [sublane, hidden]
    metadata_ref: gmm_v2.MetadataRef,
    zero_ref: jax.Array | None,
    semaphore_ref: jax.Array | None,
    *,
    cfgs1: gmm_v2.GmmConfigs,
    cfgs2: gmm_v2.GmmConfigs,
):
    """Entry point for the fused FFN kernel.

    One gmm_v2.fill_metadata scan serves both matmuls: the gm tiling is a
    function of (group_sizes, group_offset, tile_m, sublane) only, all of
    which are identical between the two stages by construction (asserted
    in gmm_fused).
    """
    num_gm = gmm_v2.fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs2,
    )
    _run_fused_pipeline(num_gm,
                        metadata_ref,
                        lhs_ref,
                        w1_ref,
                        w2_ref,
                        out_ref,
                        partial_out_ref,
                        zero_ref,
                        semaphore_ref,
                        cfgs1=cfgs1,
                        cfgs2=cfgs2)


def fused_vmem_estimate(cfgs1: gmm_v2.GmmConfigs,
                        cfgs2: gmm_v2.GmmConfigs) -> int:
    """Approximate VMEM footprint (bytes) of the fused pipeline."""
    t1, t2 = cfgs1.tiles, cfgs2.tiles
    lhs_bytes = jax.dtypes.itemsize_bits(cfgs1.lhs_cfgs.dtype) // 8
    out_bytes = jnp.dtype(cfgs2.out_dtype).itemsize
    acc_bytes = jnp.dtype(cfgs1.acc_dtype).itemsize
    sublane = cfgs1.dims.size_lhs_sublane

    lhs_vmem = _LHS_BUFFER_COUNT * t1.tile_m * t1.tile_k * lhs_bytes
    # w1: gate + up weight tiles + f32 scales.
    w1_tile_bits = t1.tile_k * t1.tile_n * jax.dtypes.itemsize_bits(
        cfgs1.rhs_cfgs.dtype)
    w1_scale_bytes = cfgs1.num_quant_blocks_per_tile_k * t1.tile_n * 4
    w1_vmem = _WEIGHT_BUFFER_COUNT * 2 * (w1_tile_bits // 8 + w1_scale_bytes)
    w2_tile_bits = t2.tile_k * t2.tile_n * jax.dtypes.itemsize_bits(
        cfgs2.rhs_cfgs.dtype)
    w2_scale_bytes = cfgs2.num_quant_blocks_per_tile_k * t2.tile_n * 4
    w2_vmem = _WEIGHT_BUFFER_COUNT * (w2_tile_bits // 8 + w2_scale_bytes)
    out_vmem = _OUT_BUFFER_COUNT * t1.tile_m * t2.tile_n * out_bytes
    # Live intermediate values (acc1, act/mid, acc2); no explicit acc
    # scratch is allocated (num_k == 1 by contract -> no cross-step
    # accumulation), Mosaic spills these as needed.
    live_vmem = t1.tile_m * (2 * t1.tile_n + t1.tile_n + t2.tile_n) * acc_bytes
    partial_vmem = sublane * t2.tile_n * out_bytes
    zero_vmem = _ZERO_REF_TARGET_BYTES if cfgs2.zero_init else 0
    return (lhs_vmem + w1_vmem + w2_vmem + out_vmem + live_vmem +
            partial_vmem + zero_vmem)


def get_fused_cost_estimate(cfgs1: gmm_v2.GmmConfigs,
                            cfgs2: gmm_v2.GmmConfigs) -> pl.CostEstimate:
    """Cost of both matmuls minus the fused-away intermediate HBM trips."""
    c1 = gmm_v2.get_cost_estimate(cfgs1)
    c2 = gmm_v2.get_cost_estimate(cfgs2)
    mid_out_bytes = (cfgs1.dims.size_m * cfgs1.out_size_n *
                     jnp.dtype(cfgs1.out_dtype).itemsize)
    c2_lhs_dtype = cfgs2.lhs_cfgs.quant_dtype or cfgs2.lhs_cfgs.dtype
    mid_in_bytes = (cfgs2.dims.size_m * cfgs2.dims.size_k *
                    jnp.dtype(c2_lhs_dtype).itemsize)
    return pl.CostEstimate(
        flops=c1.flops + c2.flops,
        bytes_accessed=(c1.bytes_accessed + c2.bytes_accessed - mid_out_bytes -
                        mid_in_bytes),
        transcendentals=0,
    )


def get_fused_scope_name(cfgs1: gmm_v2.GmmConfigs,
                         cfgs2: gmm_v2.GmmConfigs) -> str:
    dims1, dims2 = cfgs1.dims, cfgs2.dims
    return (f"gmm_fused-g_{dims1.size_group}-m_{dims1.size_m}"
            f"-h_{dims1.size_k}-i_{dims2.size_k}-act_{cfgs1.fuse_act}"
            f"-tm_{cfgs1.tiles.tile_m}")


def get_fused_metadata(cfgs1: gmm_v2.GmmConfigs, cfgs2: gmm_v2.GmmConfigs):
    ret = {f"gmm1.{k}": v for k, v in gmm_v2.get_metadata(cfgs1).items()}
    ret.update({f"gmm2.{k}": v for k, v in gmm_v2.get_metadata(cfgs2).items()})
    return ret


def _default_tile_m(size_m: int) -> int:
    """Default gm-tile rows; matches calculate_tiling's pre-bucketing
    choice for a quantized lhs on 8-bit weights (128)."""
    return min(128, size_m)


@jax.jit(static_argnames=[
    "fuse_act",
    "tile_m",
    "vmem_limit_bytes",
    "preferred_element_type",
    "acc_dtype",
    "zero_initialize",
])
def gmm_fused(
    lhs: jax.Array,  # [size_m, hidden]
    w1: jax.Array,  # [size_group, hidden, 2 * inter] (gate|up fused)
    w2: jax.Array,  # [size_group, inter, hidden]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    w1_scale: jax.Array,  # [size_group, nb1, 1, 2 * inter] f32
    w2_scale: jax.Array,  # [size_group, nb2, 1, hidden] f32
    group_offset: jax.Array | None = None,  # int32[1]
    *,
    fuse_act: str = "silu",
    tile_m: int | None = None,
    vmem_limit_bytes: int | None = None,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    zero_initialize: bool = True,
) -> jax.Array:
    """Fused MoE FFN: GMM1 (gate/up) + activation + GMM2 (down).

    Bitwise-equivalent to (and a drop-in replacement for)

        mid = gmm_v2(lhs, w1, group_sizes, rhs_scale=w1_scale,
                     group_offset=group_offset, fuse_act=fuse_act)
        out = gmm_v2(mid, w2, group_sizes, rhs_scale=w2_scale,
                     group_offset=group_offset)

    under the module docstring's conditions (the pair traced with a single
    k/n tile per matmul), but with the intermediate activation
    VMEM-resident (never written to or read from HBM) and its
    requantization done in-register.

    Only the quantized matmul path is supported: the weights must be
    quantized (with scales) such that gmm_v2 would dynamically quantize
    the lhs (see gmm_v2.make_gmm_configs); biases are not supported.

    Args:
        lhs: Input rows [size_m, hidden]; dynamically quantized in-kernel
            exactly like gmm_v2 (per row per quant block).
        w1: Fused gate+up projection weights [g, hidden, 2 * inter].
        w2: Down projection weights [g, inter, hidden].
        group_sizes: Rows per group, int32[size_lhs_group].
        w1_scale: f32 scales [g, nb1, 1, 2 * inter], nb1 quant blocks
            along hidden (nb1 = 1 for channelwise).
        w2_scale: f32 scales [g, nb2, 1, hidden], nb2 quant blocks along
            inter.
        group_offset: Optional first group to process, int32[1].
        fuse_act: Activation between the matmuls (required; the gate/up
            fused w1 layout implies one). Same options as gmm_v2.
        tile_m: gm-tile rows. Defaults to min(128, size_m), halved until
            the VMEM estimate fits the limit. An explicit tile_m that
            does not fit raises instead.
        vmem_limit_bytes: VMEM limit (default 90% of capacity).
        preferred_element_type: dtype of the FINAL output. The internal
            GMM1->GMM2 bridge dtype is always lhs.dtype, matching what
            the sequential composition materializes.
        acc_dtype: Accumulator dtype for BOTH matmuls (default bf16 on
            the quantized path, as in gmm_v2).
        zero_initialize: Zero rows of the output outside the computed
            range (matches gmm_v2 default).

    Returns:
        Output of shape [size_m, hidden].
    """
    # Shape contract.
    size_m, hidden = lhs.shape
    size_group = w1.shape[0]
    if w1.shape[1] != hidden or w1.shape[2] % 2 != 0:
        raise ValueError(f"w1 shape {w1.shape} incompatible with lhs "
                         f"[{size_m}, {hidden}] (want [g, hidden, 2*inter])")
    inter = w1.shape[2] // 2
    if w2.shape != (size_group, inter, hidden):
        raise ValueError(
            f"w2 shape {w2.shape} != {(size_group, inter, hidden)}")
    if w1_scale is None or w2_scale is None:
        raise ValueError("gmm_fused requires quantized weights: w1_scale and "
                         "w2_scale must be provided")
    # Scale layout mirrors gmm_v2.validate_inputs: [g, nb, 1, n] with
    # nb quant blocks along the contraction dim (nb = 1 for channelwise).
    # num_k == 1 is asserted below, so all quant blocks of a stage live in
    # its single k tile and gmm_v2.matmul_tile applies them per block exactly as
    # the sequential pair does.
    w1_nb = w1_scale.shape[1]
    if (w1_scale.shape != (size_group, w1_nb, 1, 2 * inter)
            or hidden % w1_nb != 0):
        raise ValueError(
            f"w1_scale must be [g, nb, 1, 2*inter] with nb | hidden "
            f"(nb=1 channelwise), got {w1_scale.shape} for hidden={hidden}")
    w2_nb = w2_scale.shape[1]
    if (w2_scale.shape != (size_group, w2_nb, 1, hidden)
            or inter % w2_nb != 0):
        raise ValueError(
            f"w2_scale must be [g, nb, 1, hidden] with nb | inter "
            f"(nb=1 channelwise), got {w2_scale.shape} for inter={inter}")
    if fuse_act is None:
        raise ValueError("fuse_act is required (w1 is gate|up fused)")

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if jnp.isscalar(group_offset):
            group_offset = group_offset[None]

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

    # Build the two stage configs exactly as the sequential composition
    # would (mid_proxy stands in for the materialized intermediate), so
    # quant/acc/out dtype decisions are identical. Clamp tile_m (halving)
    # if the default choice overflows the VMEM budget.
    tile_m_arg = tile_m
    tile_m = tile_m_arg if tile_m_arg is not None else _default_tile_m(size_m)
    # GMM1->GMM2 bridge dtype: what the sequential composition would
    # materialize in HBM between the two kernels (gmm_v2's default
    # out_dtype, i.e. lhs.dtype).
    mid_proxy = jax.ShapeDtypeStruct((size_m, inter), lhs.dtype)
    while True:
        # bucket_base == tile_m: single bucket, every tile computes the
        # full tile_m rows (out-of-group rows are masked). Bucketing is a
        # partial-tile optimization of the sequential kernel that the
        # fused kernel does not use.
        tiles1 = gmm_v2.TileSizes(tile_m=tile_m,
                                  tile_k=hidden,
                                  tile_n=inter,
                                  bucket_base=tile_m)
        tiles2 = gmm_v2.TileSizes(tile_m=tile_m,
                                  tile_k=inter,
                                  tile_n=hidden,
                                  bucket_base=tile_m)
        cfgs1 = gmm_v2.make_gmm_configs(
            lhs,
            w1,
            w1_scale,
            None,
            group_sizes,
            group_offset,
            tile_info=tiles1,
            vmem_limit_bytes=vmem_limit_bytes,
            out_dtype=None,  # bridge dtype, like the sequential mid
            acc_dtype=acc_dtype,
            maybe_quantize_lhs=True,
            zero_initialize=False,  # no HBM intermediate to zero
            fuse_act=fuse_act,
        )
        cfgs2 = gmm_v2.make_gmm_configs(
            mid_proxy,
            w2,
            w2_scale,
            None,
            group_sizes,
            group_offset,
            tile_info=tiles2,
            vmem_limit_bytes=vmem_limit_bytes,
            out_dtype=preferred_element_type,
            acc_dtype=acc_dtype,
            maybe_quantize_lhs=True,
            zero_initialize=zero_initialize,
            fuse_act=None,
        )
        if fused_vmem_estimate(cfgs1, cfgs2) <= vmem_limit_bytes:
            break
        sublane = cfgs2.dims.size_lhs_sublane
        if tile_m_arg is not None or tile_m <= sublane:
            raise ValueError(
                f"fused kernel does not fit VMEM: {tile_m=} needs "
                f"{fused_vmem_estimate(cfgs1, cfgs2)} bytes "
                f"(limit {vmem_limit_bytes})")
        # Keep the halved tile a sublane multiple (tile_m may start at a
        # non-power-of-two size_m).
        half = tile_m // 2
        tile_m = max(half - half % sublane, sublane)

    dims1, dims2 = cfgs1.dims, cfgs2.dims
    tiles1, tiles2 = cfgs1.tiles, cfgs2.tiles

    # Contract asserts: single-tile matmuls and shared gm tiling, so one
    # metadata scan / row-window granularity serves both stages.
    assert pl.cdiv(dims1.size_k, tiles1.tile_k) == 1  # num_k1 == 1
    assert pl.cdiv(cfgs1.out_size_n, tiles1.tile_n) == 1  # num_n1 == 1
    assert pl.cdiv(dims2.size_k, tiles2.tile_k) == 1  # num_k2 == 1
    assert pl.cdiv(cfgs2.out_size_n, tiles2.tile_n) == 1  # num_n2 == 1
    assert dims1.size_lhs_sublane == dims2.size_lhs_sublane
    assert dims1.size_group == dims2.size_group
    assert dims1.size_lhs_group == dims2.size_lhs_group
    if tile_m % dims1.size_lhs_sublane != 0:
        raise ValueError(f"{tile_m=} must be a multiple of the sublane size "
                         f"({dims1.size_lhs_sublane})")
    if cfgs1.lhs_cfgs.quant_dtype is None or cfgs2.lhs_cfgs.quant_dtype is None:
        raise NotImplementedError(
            "gmm_fused requires the quantized (postscale) matmul path; "
            "got an unquantized config — check weight dtype/scales and "
            "hardware fp8/int8 support")
    # The requantization bridge assumes the sequential mid dtype feeds
    # GMM2's dynamic quant the same way.
    assert cfgs2.lhs_cfgs.dtype == cfgs1.out_dtype

    # Prepare block specs (scales stay in HBM, windowed by the pipeline).
    w1_scale = w1_scale.astype(jnp.float32)
    w2_scale = w2_scale.astype(jnp.float32)
    hbm_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    # Scratch shapes; metadata is shared by both stages (identical tiling).
    max_num_gm = dims2.size_group + pl.cdiv(dims2.size_m, tile_m) - 1
    scratch_shapes = [
        # partial_out_ref (final output columns = tile_n2 = hidden)
        pltpu.VMEM((dims2.size_lhs_sublane, tiles2.tile_n), cfgs2.out_dtype),
        gmm_v2.MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1, ), jnp.int32),
        ),
    ]

    num_lanes = pltpu.get_tpu_info().num_lanes
    if cfgs2.zero_init:
        # Same zero-fill buffer strategy as gmm_v2 (see comment there).
        out_bytes = jnp.dtype(cfgs2.out_dtype).itemsize
        tile_zero_m = _ZERO_REF_TARGET_BYTES // num_lanes // out_bytes
        tile_zero_m = min(tile_zero_m, dims2.size_m)
        scratch_shapes += [
            pltpu.VMEM((tile_zero_m, num_lanes), cfgs2.out_dtype),
            pltpu.SemaphoreType.DMA((1, )),
        ]
    else:
        scratch_shapes += [None, None]

    aligned_n = gmm_v2.align_to(cfgs2.out_size_n, num_lanes)
    out_init = jax.ShapeDtypeStruct((dims2.size_m, aligned_n), cfgs2.out_dtype)
    w1_weights = gmm_v2.WeightsRef(weight=w1, scale=w1_scale, bias=None)
    w2_weights = gmm_v2.WeightsRef(weight=w2, scale=w2_scale, bias=None)

    return pl.pallas_call(
        functools.partial(fused_kernel_main, cfgs1=cfgs1, cfgs2=cfgs2),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                gmm_v2.WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=hbm_scale_spec,
                    bias=None,
                ),
                gmm_v2.WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=hbm_scale_spec,
                    bias=None,
                ),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=get_fused_scope_name(cfgs1, cfgs2),
        cost_estimate=get_fused_cost_estimate(cfgs1, cfgs2),
        metadata=get_fused_metadata(cfgs1, cfgs2),
    )(group_sizes, group_offset, lhs, w1_weights,
      w2_weights)[:, :cfgs2.out_size_n]
