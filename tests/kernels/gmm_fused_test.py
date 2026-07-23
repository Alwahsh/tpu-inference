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
"""Tests for the fused MoE FFN kernel (gmm_fused).

The kernel's contract is BITWISE equality with the sequential gmm_v2
composition (GMM1 with fused activation -> GMM2), so most tests assert
exact equality against that pair rather than a tolerance against a dense
reference; one test anchors the pair itself against a float32 reference.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tests.kernels.gmm_test import get_group_sizes, quantize_tensor
from tpu_inference.kernels.megablox.gmm_fused import gmm_fused
from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2

jax.config.parse_flags_with_absl()

# Small defaults satisfying the kernel's constraints: intermediate size a
# multiple of the lane count (128), and every contraction dim's quant block
# (channelwise: the full dim) at least the MXU column size (256 on v6e and
# newer) — smaller blocks take gmm_v2's dequantize-before-matmul path, which
# gmm_fused does not support.
HIDDEN = 512
INTER = 512
NUM_GROUPS = 16


def sequential_gmm_pair(
    lhs: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    group_sizes: jax.Array,
    w1_scale: jax.Array,
    w2_scale: jax.Array,
    group_offset: jax.Array | None = None,
    fuse_act: str = "silu",
) -> jax.Array:
    """The sequential composition gmm_fused is bitwise-equivalent to."""
    mid = gmm_v2(lhs,
                 w1,
                 group_sizes,
                 rhs_scale=w1_scale,
                 group_offset=group_offset,
                 fuse_act=fuse_act)
    return gmm_v2(mid,
                  w2,
                  group_sizes,
                  rhs_scale=w2_scale,
                  group_offset=group_offset)


def make_moe_weights(key: jax.Array,
                     num_groups: int,
                     hidden: int = HIDDEN,
                     inter: int = INTER,
                     dtype: jnp.dtype = jnp.float8_e4m3fn,
                     block_size: int | None = None):
    """Random gate/up + down weights, quantized along the contraction dim.

    block_size None means one quant block per column (channelwise scales).
    """
    k1, k2 = jax.random.split(key)
    w1 = jax.random.uniform(k1, (num_groups, hidden, 2 * inter), jnp.bfloat16,
                            -1, 1)
    w2 = jax.random.uniform(k2, (num_groups, inter, hidden), jnp.bfloat16, -1,
                            1)
    w1_q, w1_scale = quantize_tensor(w1,
                                     dtype,
                                     axis=1,
                                     block_size=block_size or hidden)
    w2_q, w2_scale = quantize_tensor(w2,
                                     dtype,
                                     axis=1,
                                     block_size=block_size or inter)
    # [g, nb, n] -> [g, nb, 1, n], the kernel's scale layout.
    w1_scale = jnp.expand_dims(w1_scale, axis=2)
    w2_scale = jnp.expand_dims(w2_scale, axis=2)
    return w1_q, w1_scale, w2_q, w2_scale


def make_group_sizes(pattern: str, batch_size: int,
                     num_groups: int) -> jax.Array:
    if pattern == "uniform":
        base = batch_size // num_groups
        sizes = jnp.full((num_groups, ), base, dtype=jnp.int32)
        return sizes.at[-1].add(batch_size - base * num_groups)
    if pattern == "skewed":
        # Uneven sizes with empty groups; group boundaries land mid-sublane.
        sizes = get_group_sizes(batch_size, num_groups)
        sizes = sizes.at[::3].set(0)
        return sizes.at[-1].set(batch_size - jnp.sum(sizes[:-1]))
    if pattern == "onehot":
        sizes = jnp.zeros((num_groups, ), dtype=jnp.int32)
        return sizes.at[num_groups // 2].set(batch_size)
    raise ValueError(f"unknown pattern {pattern}")


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class GmmFusedTest(jtu.JaxTestCase):

    def _assert_fused_matches_pair(self,
                                   batch_size,
                                   group_sizes,
                                   group_offset,
                                   dtype=jnp.float8_e4m3fn,
                                   block_size=None,
                                   fuse_act="silu",
                                   tile_m=None):
        num_local_groups = NUM_GROUPS - group_offset
        lhs = jax.random.normal(jax.random.key(1), (batch_size, HIDDEN),
                                dtype=jnp.bfloat16)
        w1, w1_scale, w2, w2_scale = make_moe_weights(jax.random.key(2),
                                                      num_local_groups,
                                                      dtype=dtype,
                                                      block_size=block_size)
        group_offset = jnp.array([group_offset], dtype=jnp.int32)

        expected = sequential_gmm_pair(lhs,
                                       w1,
                                       w2,
                                       group_sizes,
                                       w1_scale,
                                       w2_scale,
                                       group_offset=group_offset,
                                       fuse_act=fuse_act)
        actual = gmm_fused(lhs,
                           w1,
                           w2,
                           group_sizes,
                           w1_scale,
                           w2_scale,
                           group_offset=group_offset,
                           fuse_act=fuse_act,
                           tile_m=tile_m)

        # Bitwise contract (assertArraysEqual treats NaNs as equal, which
        # matches: only garbage rows the pair also produces may be NaN).
        self.assertArraysEqual(actual, expected)
        return actual

    @parameterized.product(
        batch_size=[80, 128],
        fill=["uniform", "skewed", "onehot"],
        group_offset=[0, 3],
        tile_m=[None, 32],
    )
    def test_fused_matches_sequential_pair(self, batch_size, fill,
                                           group_offset, tile_m):
        """Bitwise equality across group fills, EP windows and tilings.

        batch 80 exercises partial sublanes and odd tile counts; skewed
        fills exercise empty groups and mid-sublane group boundaries (with
        group_offset != 0 they also produce a non-sublane-aligned prefix);
        onehot exercises a single multi-tile group.
        """
        group_sizes = make_group_sizes(fill, batch_size, NUM_GROUPS)
        self._assert_fused_matches_pair(batch_size,
                                        group_sizes,
                                        group_offset,
                                        tile_m=tile_m)

    @parameterized.product(fuse_act=["swigluoai", "gelu"])
    def test_fused_activations(self, fuse_act):
        """Non-default activations pass through the same parity."""
        group_sizes = make_group_sizes("skewed", 128, NUM_GROUPS)
        self._assert_fused_matches_pair(128,
                                        group_sizes,
                                        group_offset=0,
                                        fuse_act=fuse_act)

    @parameterized.product(
        dtype=[jnp.int8, jnp.float8_e4m3fn, jnp.float4_e2m1fn],
        block_size=[None, 256],
    )
    def test_fused_weight_dtypes_and_block_scales(self, dtype, block_size):
        """Weight dtypes and blockwise (nb > 1) scale layouts."""
        if dtype == jnp.float4_e2m1fn and not jtu.is_device_tpu_at_least(
                version=7):
            self.skipTest("float4_e2m1fn requires TPU v7+")
        group_sizes = make_group_sizes("skewed", 128, NUM_GROUPS)
        self._assert_fused_matches_pair(128,
                                        group_sizes,
                                        group_offset=0,
                                        dtype=dtype,
                                        block_size=block_size)

    @parameterized.product(group_offset=[0, 3])
    def test_fused_group_boundaries_and_nan_masking(self, group_offset):
        """Crafted mid-sublane boundaries, -0.0 rows and a poisoned group.

        Group sizes put boundaries inside sublanes, so adjacent gm tiles
        share sublane blocks. One group's w1_scale is blown up to 1e30 so
        its rows (including the non-owned rows of shared sublanes computed
        by its tile) go inf/NaN through the activation: those must be
        masked before the GMM1->GMM2 bridge and must not leak into
        neighboring groups' rows. -0.0 lhs rows check the sign-of-zero
        argument (the fused path skips the sequential mid's x + 0 partial
        adds).
        """
        group_sizes = jnp.array(
            [3, 13, 7, 9, 5, 11, 1, 15, 2, 14, 6, 10, 4, 12, 8, 8],
            dtype=jnp.int32)
        batch_size = int(jnp.sum(group_sizes))
        num_local_groups = NUM_GROUPS - group_offset

        lhs = jax.random.normal(jax.random.key(1), (batch_size, HIDDEN),
                                dtype=jnp.bfloat16)
        lhs = lhs.at[0:4].set(jnp.float32(-0.0).astype(jnp.bfloat16))
        w1, w1_scale, w2, w2_scale = make_moe_weights(jax.random.key(2),
                                                      num_local_groups)
        # Poison a group whose row window starts and ends mid-sublane.
        poisoned = 2
        w1_scale = w1_scale.at[poisoned].multiply(1e30)
        group_offset_arr = jnp.array([group_offset], dtype=jnp.int32)

        expected = sequential_gmm_pair(lhs,
                                       w1,
                                       w2,
                                       group_sizes,
                                       w1_scale,
                                       w2_scale,
                                       group_offset=group_offset_arr)
        actual = gmm_fused(lhs,
                           w1,
                           w2,
                           group_sizes,
                           w1_scale,
                           w2_scale,
                           group_offset=group_offset_arr)

        self.assertArraysEqual(actual, expected)

        # Guard: the poison stays localized to the poisoned group's rows.
        row_ends = np.cumsum(np.array(group_sizes))
        row_starts = np.concatenate([[0], row_ends[:-1]])
        poisoned_global = poisoned + group_offset
        finite_rows = np.ones(batch_size, dtype=bool)
        finite_rows[row_starts[poisoned_global]:row_ends[poisoned_global]] = (
            False)
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(actual[finite_rows].astype(
                jnp.float32)))))

    def test_fused_empty_group_window(self):
        """All groups in the shard's window empty: output is all zeros."""
        batch_size = 128
        group_offset = 12
        num_local_groups = NUM_GROUPS - group_offset
        # All rows belong to groups before the window.
        group_sizes = make_group_sizes("uniform", batch_size, group_offset)
        group_sizes = jnp.concatenate(
            [group_sizes,
             jnp.zeros((num_local_groups, ), dtype=jnp.int32)])

        actual = self._assert_fused_matches_pair(batch_size, group_sizes,
                                                 group_offset)
        self.assertArraysEqual(actual, jnp.zeros_like(actual))

    def test_fused_matches_float32_reference(self):
        """Anchors the fused output against a dense float32 reference.

        The bitwise tests above prove fused == sequential pair; this one
        proves the pair itself computes the right values (rel-L2 within
        the lhs+weight quantization noise floor).
        """
        batch_size = 128
        group_sizes = make_group_sizes("skewed", batch_size, NUM_GROUPS)
        lhs = jax.random.normal(jax.random.key(1), (batch_size, HIDDEN),
                                dtype=jnp.bfloat16)
        w1, w1_scale, w2, w2_scale = make_moe_weights(jax.random.key(2),
                                                      NUM_GROUPS)

        actual = gmm_fused(lhs, w1, w2, group_sizes, w1_scale,
                           w2_scale).astype(jnp.float32)

        # Dense per-group reference with dequantized weights: the raw GMM1
        # output over w1's columns is [gate | up] concatenated.
        w1_f32 = w1.astype(jnp.float32) * w1_scale.reshape(NUM_GROUPS, 1, -1)
        w2_f32 = w2.astype(jnp.float32) * w2_scale.reshape(NUM_GROUPS, 1, -1)
        expected = jnp.zeros((batch_size, HIDDEN), dtype=jnp.float32)
        starts = jnp.concatenate(
            [jnp.zeros((1, ), jnp.int32),
             jnp.cumsum(group_sizes)[:-1]])
        for g in range(NUM_GROUPS):
            start, end = int(starts[g]), int(starts[g] + group_sizes[g])
            raw = lhs[start:end].astype(jnp.float32) @ w1_f32[g]
            gate, up = jnp.split(raw, 2, axis=-1)
            mid = jax.nn.silu(gate) * up
            expected = expected.at[start:end].set(mid @ w2_f32[g])

        rel_l2 = (jnp.linalg.norm(actual - expected) /
                  jnp.linalg.norm(expected))
        self.assertLess(float(rel_l2), 0.05)

    def test_fused_input_validation(self):
        batch_size = 128
        group_sizes = make_group_sizes("uniform", batch_size, NUM_GROUPS)
        lhs = jax.random.normal(jax.random.key(1), (batch_size, HIDDEN),
                                dtype=jnp.bfloat16)
        w1, w1_scale, w2, w2_scale = make_moe_weights(jax.random.key(2),
                                                      NUM_GROUPS)

        # w2 not matching w1's intermediate size (e.g. a padded w1).
        with self.assertRaisesRegex(ValueError, "w2 shape"):
            gmm_fused(lhs, w1, w2[:, :INTER // 2, :], group_sizes, w1_scale,
                      w2_scale)

        # Unquantized weights are not supported.
        with self.assertRaisesRegex(ValueError, "requires quantized"):
            gmm_fused(lhs, w1, w2, group_sizes, None, None)

        # Quant blocks smaller than the MXU force dequantize-before-matmul,
        # which disables the dynamic lhs quantization the kernel requires.
        w1_small, w1_scale_small, w2_small, w2_scale_small = make_moe_weights(
            jax.random.key(2), NUM_GROUPS, dtype=jnp.int8, block_size=64)
        with self.assertRaises(NotImplementedError):
            gmm_fused(lhs, w1_small, w2_small, group_sizes, w1_scale_small,
                      w2_scale_small)

        # An explicit tile_m that does not fit in VMEM raises rather than
        # being silently clamped.
        with self.assertRaisesRegex(ValueError, "does not fit VMEM"):
            gmm_fused(lhs,
                      w1,
                      w2,
                      group_sizes,
                      w1_scale,
                      w2_scale,
                      tile_m=128,
                      vmem_limit_bytes=1024 * 1024)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
