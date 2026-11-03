# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Share frozen DINO weights and reuse identical inputs during reconstruction."""
from contextvars import ContextVar
from functools import wraps
import os

import torch

from sam3d_objects.conditioning_cache import _version


_features = ContextVar('sam3d_image_features', default=None)


def cache_image_features(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        state = {'entries': [], 'bytes': 0}
        token = _features.set(state)
        try:
            return function(*args, **kwargs)
        finally:
            _features.reset(token)
            state['entries'].clear()
    return wrapped


def backbone_features(model, x):
    state = _features.get()
    if (state is None or os.environ.get('SAM3D_CACHE_IMAGE_FEATURES', '1') == '0'
            or torch.is_grad_enabled() or x.device.type != 'cuda'
            or x.shape != (1, 3, 518, 518) or x.dtype not in (torch.float16, torch.float32)
            or type(model).__module__ != 'dinov2.models.vision_transformer'
            or type(model).__name__ != 'DinoVisionTransformer'
            or 'forward_features' in model.__dict__
            or any(m.training or m._forward_hooks or m._forward_pre_hooks
                   for m in model.modules())):
        return model.forward_features(x)
    parameters = tuple((id(p), _version(p))
                       for p in (*model.parameters(), *model.buffers()))
    if (any(p.requires_grad for p in model.parameters())
            or any(version is None for _, version in parameters)):
        return model.forward_features(x)
    configuration = tuple(getattr(model, name, None) for name in (
        'interpolate_antialias', 'interpolate_offset', 'num_register_tokens',
        'patch_size', 'chunked_blocks'))
    key = (model, parameters, configuration, x.dtype, x.device, x.stride(),
           torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))
    packed = x.detach().contiguous()
    for old_key, saved, output, versions in state['entries']:
        if (key == old_key and all(output[name] is tensor and _version(tensor) == version
                                  for name, tensor, version in versions)
                and torch.equal(saved.view(torch.uint8), packed.view(torch.uint8))):
            return output
    output = model.forward_features(x)
    versions = [(name, value, _version(value)) for name, value in output.items()
                if isinstance(value, torch.Tensor)]
    size = x.numel() * x.element_size() + sum(
        value.numel() * value.element_size() for _, value, _ in versions)
    if (len(state['entries']) < 8 and state['bytes'] + size <= 64 * 1024**2
            and all(version is not None for _, _, version in versions)):
        state['entries'].append((key, packed.clone(), output, versions))
        state['bytes'] += size
    return output


def _identical_frozen(left, right):
    if type(left) is not type(right) or repr(left) != repr(right):
        return False
    for model in (left, right):
        if any(m.training or m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
               for m in model.modules()):
            return False
        if any(p.requires_grad for p in model.parameters()):
            return False
    a, b = left.state_dict(), right.state_dict()
    a.update({('buffer', name): value for name, value in left.named_buffers()})
    b.update({('buffer', name): value for name, value in right.named_buffers()})
    return (a.keys() == b.keys() and all(
        type(a[key]) is torch.Tensor and type(b[key]) is torch.Tensor
        and a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
        and a[key].device == b[key].device
        and torch.equal(a[key].contiguous().reshape(-1).view(torch.uint8),
                        b[key].contiguous().reshape(-1).view(torch.uint8))
        for key in a))


def share_frozen_backbones(embedders):
    """Deduplicate after checkpoint loading; preserve each Dino preprocessing wrapper."""
    if os.environ.get('SAM3D_SHARE_BACKBONES', '1') == '0':
        return False
    from sam3d_objects.model.backbone.dit.embedder.dino import Dino
    canonical = []
    changed = False
    attributes = ('embed_dim', 'num_features', 'num_tokens', 'n_blocks', 'num_heads',
                  'patch_size', 'num_register_tokens', 'interpolate_antialias',
                  'interpolate_offset', 'chunked_blocks')
    for embedder in embedders.values():
        if embedder is None:
            continue
        for wrapper in embedder.modules():
            if type(wrapper) is not Dino:
                continue
            backbone = wrapper.backbone
            if (type(backbone).__module__ != 'dinov2.models.vision_transformer'
                    or type(backbone).__name__ != 'DinoVisionTransformer'):
                continue
            for previous in canonical:
                if (all(getattr(previous, name, None) == getattr(backbone, name, None)
                        for name in attributes) and _identical_frozen(previous, backbone)):
                    wrapper.backbone = previous
                    changed = changed or backbone is not previous
                    break
            else:
                canonical.append(backbone)
    return changed


def prepare_inference_memory(pipeline):
    changed = share_frozen_backbones(pipeline.condition_embedders)
    if (pipeline.shape_model_dtype == torch.float16 and pipeline.device.type == 'cuda'
            and os.environ.get('SAM3D_FP16_LINEAR_WEIGHTS', '1') != '0'
            and getattr(torch.cuda.get_device_properties(pipeline.device),
                        'gcnArchName', '').split(':')[0] in ('gfx1151', 'gfx1201')):
        from sam3d_objects.linear_rocm import InferenceLinear
        backbone = pipeline.models['ss_generator'].reverse_fn.backbone
        for module in backbone.modules():
            if type(module) is not InferenceLinear or module.training:
                continue
            parameters = tuple(module.parameters(recurse=False))
            if all(not p.requires_grad and p.dtype == torch.float32 for p in parameters):
                for parameter in parameters:
                    parameter.data = parameter.data.to(torch.float16)
                changed = True
    if changed:
        # Return the discarded weight allocations before activation buffers grow.
        torch.cuda.empty_cache()
