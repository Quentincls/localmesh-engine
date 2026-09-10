import torch
import torch.nn as nn
from ..modules import sparse as sp

MIX_PRECISION_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    sp.SparseConv3d,
    sp.SparseInverseConv3d,
    sp.SparseLinear,
)


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    unpatch_fp8_forward(l)
    if isinstance(l, MIX_PRECISION_MODULES):
        for p in l.parameters():
            p.data = p.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    unpatch_fp8_forward(l)
    if isinstance(l, MIX_PRECISION_MODULES):
        for p in l.parameters():
            p.data = p.data.float()


def convert_module_to(l, dtype):
    """
    Convert primitive modules to the given dtype.
    """
    if dtype == torch.float8_e4m3fn:
        convert_module_to_fp8(l)
        return

    unpatch_fp8_forward(l)
    if isinstance(l, MIX_PRECISION_MODULES):
        for p in l.parameters():
            p.data = p.data.to(dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def manual_cast(tensor, dtype):
    """
    Cast if autocast is not enabled.
    """
    if dtype == torch.float8_e4m3fn:
        dtype = torch.bfloat16

    if not torch.is_autocast_enabled():
        return tensor.type(dtype)
    return tensor


def str_to_dtype(dtype_str: str):
    return {
        'f16': torch.float16,
        'fp16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'f32': torch.float32,
        'fp32': torch.float32,
        'float32': torch.float32,
        'float8_e4m3fn': torch.float8_e4m3fn,
        'fp8': torch.float8_e4m3fn,
    }[dtype_str]

def unpatch_fp8_forward(module):
    if hasattr(module, "_original_forward_fp8"):
        module.forward = module._original_forward_fp8
        delattr(module, "_original_forward_fp8")

def patch_fp8_forward(module):
    if hasattr(module, "_original_forward_fp8"):
        return
    module._original_forward_fp8 = module.forward
    
    def fp8_forward(*args, **kwargs):
        with torch.no_grad():
            orig_weight_data = module.weight.data
            module.weight.data = orig_weight_data.to(torch.bfloat16)
            has_bias = getattr(module, "bias", None) is not None
            if has_bias:
                orig_bias_data = module.bias.data
                module.bias.data = orig_bias_data.to(torch.bfloat16)
                
        try:
            return module._original_forward_fp8(*args, **kwargs)
        finally:
            with torch.no_grad():
                module.weight.data = orig_weight_data
                if has_bias:
                    module.bias.data = orig_bias_data
                
    module.forward = fp8_forward

def convert_module_to_fp8(l):
    """
    Convert primitive modules to float8_e4m3fn and patch their forwards to execute dynamically.
    """
    if isinstance(l, MIX_PRECISION_MODULES):
        for p in l.parameters():
            p.data = p.data.to(torch.float8_e4m3fn)
        patch_fp8_forward(l)
