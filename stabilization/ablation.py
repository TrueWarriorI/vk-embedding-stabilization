"""Compare score SVD and Procrustes under reciprocal coordinate scaling."""

import argparse
from dataclasses import asdict
import math
from pathlib import Path

import torch

from .artifacts import write_report
from .method import align, fit_svd, procrustes
from .metrics import mean_cosine, score_error
from .runtime import runtime
from .synthetic import SyntheticConfig, make_embeddings, plane_rotation


@torch.no_grad()
def run_ablation(
    config: SyntheticConfig, scale: float = 3.0, *, device: str | torch.device = "cpu"
) -> dict:
    """Fit on alternating items and evaluate on the other items."""
    if not math.isfinite(scale) or scale <= 1:
        raise ValueError("Scale must be finite and greater than one.")
    if config.noise:
        raise ValueError("This experiment requires score-equivalent versions without additive noise.")
    if config.items < 2 * config.dim or config.users < config.dim:
        raise ValueError("Provide at least dim users and 2 * dim items.")

    base_users, base_items = make_embeddings(config, version=0, device=device)
    scales = torch.linspace(-1, 1, config.dim, device=device, dtype=base_items.dtype)
    scales = (scales * math.log(scale)).exp()
    rotation = plane_rotation(config.dim, config.angle, base_items.dtype, device)
    new_users = (base_users / scales) @ rotation
    new_items = (base_items * scales) @ rotation

    anchors = slice(0, None, 2)
    held_out = slice(1, None, 2)
    # Fit both steps without rows reserved for evaluation.
    reference = fit_svd(base_users, base_items[anchors])
    current = fit_svd(new_users, new_items[anchors])
    raw_rotation = procrustes(new_items[anchors], base_items[anchors])
    stable = align(current, new_items[anchors], base_items[anchors] @ reference.item)

    base_test = base_items[held_out]
    new_test = new_items[held_out]
    canonical_users, canonical_items = reference.apply(base_users, base_test)
    svd_users, svd_items = current.apply(new_users, new_test)
    stable_users, stable_items = stable.apply(new_users, new_test)
    variants = {
        "raw": (new_users, new_test, base_users, base_test, "raw"),
        "procrustes": (
            new_users @ raw_rotation, new_test @ raw_rotation, base_users, base_test, "raw"
        ),
        "svd": (svd_users, svd_items, canonical_users, canonical_items, "svd"),
        "full": (stable_users, stable_items, canonical_users, canonical_items, "svd"),
    }
    results = {}
    for name, (users, items, ref_users, ref_items, ref_basis) in variants.items():
        results[name] = {
            "reference_basis": ref_basis,
            "users_cosine": mean_cosine(users, ref_users)["mean"],
            "held_out_items_cosine": mean_cosine(items, ref_items)["mean"],
            "within_version_score_error": score_error(new_users, new_test, users, items),
            "mixed_version_score_error": score_error(new_users, new_test, users, ref_items),
        }
    return {
        "synthetic": asdict(config),
        "coordinate_scale": scale,
        "anchor_items": len(base_items[anchors]),
        "held_out_items": len(base_test),
        "device": str(device),
        "dtype": "float64",
        "original_score_error": score_error(base_users, base_test, new_users, new_test),
        "variants": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=4096)
    parser.add_argument("--items", type=int, default=3072)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--angle", type=float, default=75)
    parser.add_argument("--scale", type=float, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = SyntheticConfig(args.users, args.items, args.dim, args.seed, args.angle)
    with runtime(args.device, 1) as device:
        write_report(run_ablation(config, args.scale, device=device), args.output)


if __name__ == "__main__":
    main()
