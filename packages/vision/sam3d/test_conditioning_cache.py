# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Cache isolation and invalidation checks; runs on CPU without checkpoints."""
import os
import unittest
from unittest.mock import patch

import torch
from sam3d_objects.conditioning_cache import (
    conditioning_cast, conditioning_projection as project,
    conditioning_zero, sampling_cache,
    sampling_stage, _weights,
)


class ConditioningCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.module = torch.nn.Linear(4, 8).eval()
        self.context = torch.randn(1, 3, 4)

    @torch.no_grad()
    def test_cfg_and_request_isolation(self):
        with sampling_cache() as state:
            cond = conditioning_cast(self.context, torch.float32)
            zero = conditioning_zero(self.context)
            conditional = project(self.module, cond)
            unconditional = project(self.module, zero)
            self.assertIs(project(self.module, cond), conditional)
            self.assertIs(project(self.module, zero), unconditional)
            self.assertFalse(torch.equal(conditional, unconditional))
            torch.testing.assert_close(conditional, self.module(self.context), rtol=0, atol=0)
            torch.testing.assert_close(unconditional, self.module(self.context * 0), rtol=0, atol=0)
        self.assertFalse(state.entries)
        with sampling_cache():
            self.assertIsNot(project(self.module, cond), conditional)
            other = self.context + 1
            torch.testing.assert_close(project(self.module, other), self.module(other), rtol=0, atol=0)

    @torch.no_grad()
    def test_mutations_invalidate(self):
        with sampling_cache():
            previous = project(self.module, self.context)
            self.context.add_(1)
            current = project(self.module, self.context)
            self.assertFalse(torch.equal(previous, current))
            torch.testing.assert_close(current, self.module(self.context), rtol=0, atol=0)
            self.module.weight.mul_(2)
            torch.testing.assert_close(project(self.module, self.context), self.module(self.context), rtol=0, atol=0)
            current = project(self.module, self.context)
            current.zero_()
            torch.testing.assert_close(project(self.module, self.context), self.module(self.context), rtol=0, atol=0)

    def test_gradients_and_training_use_original(self):
        with sampling_cache() as state:
            project(self.module, self.context).sum().backward()
            self.assertIsNotNone(self.module.weight.grad)
            self.assertFalse(state.entries)
        self.module.train()
        with torch.no_grad(), sampling_cache() as state:
            project(self.module, self.context)
            self.assertFalse(state.entries)

    def test_exception_cleanup_and_disable(self):
        with self.assertRaisesRegex(RuntimeError, 'sample failed'):
            with torch.no_grad(), sampling_cache() as state:
                project(self.module, self.context)
                raise RuntimeError('sample failed')
        self.assertFalse(state.entries)
        self.assertIsNone(sampling_stage())
        with patch.dict(os.environ, SAM3D_CACHE_CROSS_ATTN='0'):
            with sampling_cache('sample_slat') as disabled:
                self.assertIsNone(disabled)
                self.assertEqual(sampling_stage(), 'sample_slat')
        self.assertIsNone(sampling_stage())

    @torch.no_grad()
    def test_dtype_and_inference_tensor_fallback(self):
        with sampling_cache():
            fp16 = conditioning_cast(self.context, torch.float16)
            fp64 = conditioning_cast(self.context, torch.float64)
            self.assertEqual(fp16.dtype, torch.float16)
            self.assertEqual(fp64.dtype, torch.float64)
            self.assertIs(conditioning_cast(self.context, torch.float16), fp16)
        with torch.inference_mode(), sampling_cache() as state:
            context = self.context.clone()
            project(self.module, context)
            self.assertFalse(state.entries)

    @torch.no_grad()
    def test_autocast_and_parameter_replacement(self):
        with sampling_cache():
            fp32 = project(self.module, self.context)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                bf16 = project(self.module, self.context)
                self.assertEqual(bf16.dtype, torch.bfloat16)
                torch.testing.assert_close(bf16, self.module(self.context), rtol=0, atol=0)
            self.assertIs(project(self.module, self.context), fp32)
            self.module.weight = torch.nn.Parameter(self.module.weight + 1)
            torch.testing.assert_close(project(self.module, self.context), self.module(self.context), rtol=0, atol=0)

    @torch.no_grad()
    def test_parameter_cache_isolation_and_mutation(self):
        weight = self.module.weight
        compute = lambda: weight.to(torch.float16)
        with sampling_cache('sample_sparse_structure'):
            state = _weights.get()
            first = state.get(('linear',), weight, compute)
            self.assertIs(state.get(('linear',), weight, compute), first)
            weight.add_(1)
            second = state.get(('linear',), weight, compute)
            torch.testing.assert_close(second, compute(), rtol=0, atol=0)
            self.assertIsNot(first, second)
            second.zero_()
            torch.testing.assert_close(state.get(('linear',), weight, compute), compute(), rtol=0, atol=0)
            with sampling_cache('sample_slat'):
                self.assertIsNot(_weights.get(), state)
                self.assertFalse(_weights.get().entries)
            self.assertIs(_weights.get(), state)
        self.assertFalse(state.entries)
        self.assertIsNone(_weights.get())
        with patch.dict(os.environ, SAM3D_CACHE_LINEAR='0'), sampling_cache('sample_sparse_structure'):
            self.assertIsNone(_weights.get())

    @torch.no_grad()
    def test_cache_bounds_and_parameter_exception_cleanup(self):
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            with sampling_cache('sample_sparse_structure'):
                state = _weights.get()
                state.max_bytes = 0
                state.get(('linear',), self.module.weight, lambda: self.module.weight.half())
                self.assertFalse(state.entries)
                state.max_bytes = 1024
                state.get(('linear',), self.module.weight, lambda: self.module.weight.half())
                self.assertTrue(state.entries)
                raise RuntimeError('failed')
        self.assertFalse(state.entries)
        self.assertIsNone(_weights.get())


if __name__ == '__main__':
    unittest.main()
