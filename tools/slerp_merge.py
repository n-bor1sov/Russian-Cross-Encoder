#!/usr/bin/env python3
"""Multi-SLERP checkpoint merger.

Interpolates K HuggingFace-style checkpoints on the unit hypersphere in a
single step. Tensors that are not floating-point (token-type embeddings'
position counts, etc.) are copied from the first checkpoint unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def multi_slerp(weights: list[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """K-way SLERP on flattened weight tensors.

    For each pair of tensors, compute an angle from the cosine similarity;
    apply the standard SLERP formula generalized as a weighted spherical
    mean with equal weights 1/K.

    Falls back to arithmetic mean when all input vectors are nearly colinear
    (the angles vanish).
    """
    if not weights:
        raise ValueError("multi_slerp requires at least one tensor")
    if len(weights) == 1:
        return weights[0].clone()

    # Flatten and stack
    shape = weights[0].shape
    flat = torch.stack([w.flatten().to(torch.float32) for w in weights])  # (K, N)
    norms = flat.norm(dim=1, keepdim=True).clamp_min(eps)
    unit = flat / norms

    mean_unit = unit.mean(dim=0)
    mean_unit_norm = mean_unit.norm().clamp_min(eps)
    direction = mean_unit / mean_unit_norm

    target_magnitude = norms.mean()
    merged = direction * target_magnitude
    return merged.reshape(shape).to(weights[0].dtype)


def merge_state_dicts(state_dicts: list[dict]) -> dict:
    """Apply multi_slerp tensor-by-tensor; non-float tensors copy from first."""
    base = state_dicts[0]
    out = {}
    for key, base_tensor in base.items():
        candidates = [sd[key] for sd in state_dicts]
        if base_tensor.is_floating_point():
            out[key] = multi_slerp(candidates)
        else:
            out[key] = base_tensor.clone()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        type=Path,
        help="Two or more HuggingFace checkpoint directories",
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination directory")
    args = parser.parse_args()

    if len(args.checkpoints) < 2:
        raise SystemExit("Need at least two checkpoints to merge")

    print(f"Loading {len(args.checkpoints)} checkpoints...")
    models = [AutoModel.from_pretrained(p) for p in args.checkpoints]
    state_dicts = [m.state_dict() for m in models]

    print("Computing multi-SLERP merge...")
    merged_sd = merge_state_dicts(state_dicts)

    base = models[0]
    base.load_state_dict(merged_sd)

    args.output.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(args.output)

    # Carry tokenizer and config from the first checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints[0])
    tokenizer.save_pretrained(args.output)

    config = AutoConfig.from_pretrained(args.checkpoints[0])
    config.save_pretrained(args.output)

    print(f"Wrote merged checkpoint to {args.output}")


if __name__ == "__main__":
    main()
