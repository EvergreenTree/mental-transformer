from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrgs.data import validate_processed_payload


def feature_tensor(payload: Any, path: Path) -> torch.Tensor:
    if torch.is_tensor(payload):
        return payload.float()
    if isinstance(payload, dict):
        for key in ("features", "image_features", "embeddings"):
            if key in payload and torch.is_tensor(payload[key]):
                return payload[key].float()
    raise ValueError(f"Could not find feature tensor in {path}")


def repeated_features_identical(
    features: torch.Tensor,
    processed_ids: list[str],
    feature_ids: list[str],
) -> tuple[bool, int, float]:
    if feature_ids == processed_ids:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, stimulus_id in enumerate(processed_ids):
            groups[stimulus_id].append(index)
        nonidentical = 0
        max_diff = 0.0
        for indices in groups.values():
            if len(indices) < 2:
                continue
            diff = (features[indices] - features[indices[0]]).abs().max().item()
            max_diff = max(max_diff, diff)
            if diff > 1e-6:
                nonidentical += 1
        return nonidentical == 0, nonidentical, max_diff

    if len(feature_ids) == len(set(processed_ids)) and set(feature_ids) == set(processed_ids):
        return True, 0, 0.0
    return False, -1, float("nan")


def repeat_layout(ids: list[str]) -> str:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, stimulus_id in enumerate(ids):
        positions[stimulus_id].append(index)
    repeated = [indices for indices in positions.values() if len(indices) > 1]
    if not repeated:
        return "unique"
    grouped = all(indices == list(range(indices[0], indices[-1] + 1)) for indices in repeated)
    return "grouped" if grouped else "interleaved"


def audit_alignment(processed: Path, features: Path) -> dict[str, Any]:
    processed_payload = torch.load(processed, map_location="cpu")
    validate_processed_payload(processed_payload)
    feature_payload = torch.load(features, map_location="cpu")
    feats = feature_tensor(feature_payload, features)
    processed_ids = [str(value) for value in processed_payload["image_paths"]]

    if not isinstance(feature_payload, dict) or "image_paths" not in feature_payload:
        raise ValueError(f"Feature file lacks image_paths metadata and cannot be audited safely: {features}")
    feature_ids = [str(value) for value in feature_payload["image_paths"]]

    if feats.shape[0] != len(feature_ids):
        raise ValueError("Feature row count and feature image_paths length do not match")

    row_order_match = feature_ids == processed_ids
    unique_processed = list(dict.fromkeys(processed_ids))
    unique_layout_match = len(feature_ids) == len(unique_processed) and set(feature_ids) == set(unique_processed)
    if not row_order_match and not unique_layout_match:
        missing = sorted(set(processed_ids).difference(feature_ids))
        extra = sorted(set(feature_ids).difference(processed_ids))
        raise ValueError(
            "Feature IDs must either match processed row order exactly or cover exactly the unique processed stimuli. "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    identical, nonidentical_groups, max_duplicate_diff = repeated_features_identical(feats, processed_ids, feature_ids)
    if not identical:
        raise ValueError("Duplicate stimulus IDs do not have identical image features")

    repeat_counts = Counter(Counter(processed_ids).values())
    report = {
        "processed": str(processed),
        "features": str(features),
        "fmri_rows": int(processed_payload["fmri"].shape[0]),
        "feature_rows": int(feats.shape[0]),
        "unique_stimulus_ids": len(set(processed_ids)),
        "repeats_per_stimulus_id": {str(key): value for key, value in sorted(repeat_counts.items())},
        "num_classes": int(processed_payload["class_ids"].unique().numel()),
        "processed_first_10": processed_ids[:10],
        "processed_last_10": processed_ids[-10:],
        "feature_first_10": feature_ids[:10],
        "feature_last_10": feature_ids[-10:],
        "row_order_exact_match": row_order_match,
        "unique_stimulus_feature_layout": unique_layout_match,
        "duplicate_stimulus_features_identical": identical,
        "nonidentical_duplicate_groups": nonidentical_groups,
        "max_duplicate_feature_abs_diff": max_duplicate_diff,
        "repeat_layout": repeat_layout(processed_ids),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DIR fMRI rows against image feature rows by stimulus ID.")
    parser.add_argument("--processed", required=True)
    parser.add_argument("--features", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_alignment(Path(args.processed), Path(args.features))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
