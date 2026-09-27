"""Backend-independent affine group quantisation and nibble packing.

The functions import torch lazily so registry and CLI discovery remain usable in
the base installation.  Packed values contain two unsigned four-bit integers
per byte; an odd final value occupies the low nibble of the last byte.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PackedInt4:
    packed: object
    scales: object
    zero_points: object | None
    shape: tuple[int, ...]
    padded_values: int
    group_size: int
    symmetric: bool


def _validate_group_size(group_size: int) -> None:
    if group_size < 1:
        raise ValueError("group_size must be a positive integer.")


def pack_nibbles(values):
    """Pack a flat uint8 tensor whose values are in ``[0, 15]``."""
    import torch

    values = values.to(torch.uint8).flatten()
    if values.numel() and bool((values > 15).any()):
        raise ValueError("INT4 values must be in [0, 15].")
    if values.numel() % 2:
        values = torch.cat((values, torch.zeros(1, dtype=torch.uint8, device=values.device)))
    return values[0::2] | (values[1::2] << 4)


def unpack_nibbles(packed, count: int):
    """Unpack ``count`` unsigned values from a byte tensor."""
    import torch

    if count < 0 or packed.numel() * 2 < count:
        raise ValueError("Packed tensor does not contain the requested number of values.")
    output = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    output[0::2] = packed & 0x0F
    output[1::2] = (packed >> 4) & 0x0F
    return output[:count]


def quantize_int4(tensor, group_size: int = 64, symmetric: bool = True) -> PackedInt4:
    """Quantise a floating tensor in contiguous groups and pack its values."""
    import torch

    _validate_group_size(group_size)
    source = tensor.detach().to(torch.float32).contiguous()
    flat = source.flatten()
    padding = (-flat.numel()) % group_size
    if padding:
        flat = torch.cat((flat, torch.zeros(padding, device=flat.device)))
    groups = flat.reshape(-1, group_size)
    if symmetric:
        scales = groups.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).eps) / 7.0
        quantized = torch.round(groups / scales[:, None]).clamp(-8, 7).to(torch.int8)
        unsigned = (quantized.to(torch.int16) + 8).to(torch.uint8)
        zero_points = None
    else:
        minimum, maximum = groups.amin(dim=1), groups.amax(dim=1)
        scales = ((maximum - minimum) / 15.0).clamp_min(torch.finfo(torch.float32).eps)
        zero_points = torch.round(-minimum / scales).clamp(0, 15).to(torch.uint8)
        unsigned = torch.round(groups / scales[:, None] + zero_points[:, None]).clamp(0, 15).to(torch.uint8)
    return PackedInt4(pack_nibbles(unsigned), scales, zero_points, tuple(source.shape), padding,
                      group_size, symmetric)


def dequantize_int4(value: PackedInt4, *, device=None, dtype=None):
    """Reconstruct a floating tensor from :class:`PackedInt4`."""
    import torch

    count = value.packed.numel() * 2
    unsigned = unpack_nibbles(value.packed, count).reshape(-1, value.group_size).to(torch.float32)
    scales = value.scales.to(unsigned.device)[:, None]
    if value.symmetric:
        output = (unsigned - 8.0) * scales
    else:
        output = (unsigned - value.zero_points.to(unsigned.device)[:, None]) * scales
    output = output.flatten()
    if value.padded_values:
        output = output[:-value.padded_values]
    output = output.reshape(value.shape)
    return output.to(device=device or value.packed.device, dtype=dtype or torch.float32)
