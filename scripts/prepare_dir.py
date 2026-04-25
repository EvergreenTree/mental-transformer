from __future__ import annotations

import argparse
import csv
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROI_GROUPS = {
    "HVC": ("LOC", "FFA", "PPA"),
    "LOC": ("LOC",),
    "FFA": ("FFA",),
    "PPA": ("PPA",),
}


ARRAY_SUFFIXES = (".pt", ".npy", ".npz", ".mat", ".h5", ".hdf5")
TEXT_SUFFIXES = (".txt", ".csv", ".tsv", ".json")
TRAIN_WORDS = ("train", "training", "imagenettraining")
TEST_WORDS = ("test", "testing", "imagenettest")
SUBJECT_ALIASES = {
    "S1": ("s1", "subject1", "subject_1", "subj1", "sub-01", "sub01"),
    "S2": ("s2", "subject2", "subject_2", "subj2", "sub-02", "sub02"),
    "S3": ("s3", "subject3", "subject_3", "subj3", "sub-03", "sub03"),
}


def safe_extract_zip(path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = output_dir / member.filename
            if not target.resolve().is_relative_to(output_dir.resolve()):
                raise ValueError(f"Unsafe zip member path: {member.filename}")
        archive.extractall(output_dir)


def safe_extract_tar(path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            target = output_dir / member.name
            if not target.resolve().is_relative_to(output_dir.resolve()):
                raise ValueError(f"Unsafe tar member path: {member.name}")
        archive.extractall(output_dir)


def materialize_archives(raw_dir: Path) -> Path:
    extracted_root = raw_dir / "_extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)
    for path in raw_dir.rglob("*"):
        if not path.is_file() or "_extracted" in path.parts:
            continue
        suffixes = "".join(path.suffixes).lower()
        target = extracted_root / path.name.replace(".", "_")
        marker = target / ".complete"
        if marker.exists():
            continue
        if path.suffix.lower() == ".zip":
            print(f"EXTRACT {path} -> {target}")
            safe_extract_zip(path, target)
            marker.write_text("ok\n", encoding="utf-8")
        elif suffixes.endswith((".tar", ".tar.gz", ".tgz")):
            print(f"EXTRACT {path} -> {target}")
            safe_extract_tar(path, target)
            marker.write_text("ok\n", encoding="utf-8")
    return extracted_root


def load_array(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        if torch.is_tensor(payload):
            return payload.float()
        if isinstance(payload, dict):
            for key in ("fmri", "data", "array", "features"):
                if key in payload and torch.is_tensor(payload[key]):
                    return payload[key].float()
        raise ValueError(f"No tensor found in {path}")
    if suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()
    if suffix == ".npz":
        payload = np.load(path)
        key = "fmri" if "fmri" in payload.files else payload.files[0]
        return torch.from_numpy(payload[key]).float()
    if suffix == ".mat":
        from scipy.io import loadmat

        payload = loadmat(path)
        for key in ("fmri", "data", "X", "x"):
            if key in payload:
                return torch.from_numpy(np.asarray(payload[key])).float()
        candidates = [value for key, value in payload.items() if not key.startswith("__") and np.asarray(value).ndim == 2]
        if candidates:
            return torch.from_numpy(np.asarray(candidates[0])).float()
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Install h5py to read HDF5 fMRI files.") from exc

        candidates: list[np.ndarray] = []
        with h5py.File(path, "r") as handle:
            def visit(_: str, obj: Any) -> None:
                if hasattr(obj, "shape") and len(obj.shape) == 2 and np.issubdtype(obj.dtype, np.number):
                    candidates.append(np.asarray(obj))

            handle.visititems(visit)
        if candidates:
            largest = max(candidates, key=lambda arr: arr.shape[0] * arr.shape[1])
            return torch.from_numpy(largest).float()
    raise ValueError(f"Unsupported array format: {path}")


def load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_existing(base: Path, stems: list[str], suffixes: tuple[str, ...]) -> Path:
    for stem in stems:
        for suffix in suffixes:
            candidate = base / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"None of these files exist under {base}: {stems}")


def normalized_name(path: Path) -> str:
    return str(path).lower().replace("-", "_").replace(".", "_")


def split_words(split: str) -> tuple[str, ...]:
    return TRAIN_WORDS if split == "train" else TEST_WORDS


def score_path(path: Path, subject: str, split: str, terms: tuple[str, ...]) -> int:
    name = normalized_name(path)
    score = 0
    if any(alias in name for alias in SUBJECT_ALIASES.get(subject, (subject.lower(),))):
        score += 5
    if any(word in name for word in split_words(split)):
        score += 5
    for term in terms:
        if term in name:
            score += 2
    if "feature" in name or "decoded" in name or "vgg" in name or "cnn" in name:
        score -= 6
    return score


def candidate_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]


def discover_file(root: Path, subject: str, split: str, suffixes: tuple[str, ...], terms: tuple[str, ...]) -> Path | None:
    scored = [
        (score_path(path, subject, split, terms), path)
        for path in candidate_files(root, suffixes)
        if score_path(path, subject, split, terms) > 0
    ]
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1].stat().st_size), reverse=True)
    return scored[0][1]


def read_table_strings(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict):
            for key in ("image_paths", "stimulus_ids", "stimuli", "images", "paths", "class_names"):
                if key in payload and isinstance(payload[key], list):
                    return [str(item) for item in payload[key]]
        return []
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        rows = list(csv.DictReader(path.open("r", encoding="utf-8"), delimiter=delimiter))
        preferred = ("image_path", "image", "path", "stimulus", "stimulus_id", "filename", "class_name", "label")
        for key in preferred:
            values = [row.get(key, "") for row in rows if row.get(key)]
            if values:
                return values
        return [next(iter(row.values())) for row in rows if row]
    return load_lines(path)


def discover_strings(root: Path, subject: str, split: str, terms: tuple[str, ...], expected_len: int) -> list[str] | None:
    path = discover_file(root, subject, split, TEXT_SUFFIXES, terms)
    if path is None:
        return None
    values = read_table_strings(path)
    if expected_len < 0 and values:
        print(f"FOUND metadata {path}")
        return values
    if len(values) == expected_len:
        print(f"FOUND metadata {path}")
        return values
    return None


def discover_class_ids(root: Path, subject: str, split: str, expected_len: int) -> torch.Tensor | None:
    path = discover_file(root, subject, split, ARRAY_SUFFIXES, ("class", "label", "category", "target"))
    if path is not None:
        try:
            values = load_array(path).flatten().long()
            if values.numel() == expected_len:
                print(f"FOUND class ids {path}")
                return values
        except Exception:
            pass
    strings = discover_strings(root, subject, split, ("class", "label", "category", "target"), expected_len)
    if strings is None:
        return None
    mapping = {value: index for index, value in enumerate(sorted(set(strings)))}
    return torch.tensor([mapping[value] for value in strings], dtype=torch.long)


def fallback_image_paths(subject: str, split: str, n_samples: int) -> list[str]:
    return [f"{subject}:{split}:stimulus_{index:06d}" for index in range(n_samples)]


def fallback_class_ids(n_samples: int) -> torch.Tensor:
    return torch.arange(n_samples, dtype=torch.long)


def class_names_from_ids(class_ids: torch.Tensor) -> list[str]:
    return [f"class_{int(value):04d}" for value in torch.unique(class_ids).tolist()]


def load_roi_mask(subject_dir: Path, roi: str, fmri_dim: int) -> torch.Tensor | None:
    roi_names = ROI_GROUPS.get(roi.upper(), (roi,))
    masks: list[torch.Tensor] = []
    for name in roi_names:
        try:
            path = find_existing(subject_dir, [f"mask_{name}", f"{name}_mask", name], (".pt", ".npy", ".npz", ".mat"))
        except FileNotFoundError:
            continue
        mask = load_array(path).flatten().bool()
        masks.append(mask)
    if not masks:
        return None
    combined = torch.zeros(fmri_dim, dtype=torch.bool)
    for mask in masks:
        if mask.numel() != fmri_dim:
            raise ValueError(f"ROI mask size {mask.numel()} does not match fMRI dim {fmri_dim}")
        combined |= mask
    return combined


def build_split(subject_dir: Path, subject: str, split: str, roi: str) -> dict[str, Any]:
    fmri_path = find_existing(subject_dir, [f"{split}_fmri", f"fmri_{split}", split], (".pt", ".npy", ".npz", ".mat"))
    class_ids_path = find_existing(subject_dir, [f"{split}_class_ids", f"class_ids_{split}"], (".pt", ".npy", ".npz"))
    image_paths_path = find_existing(subject_dir, [f"{split}_image_paths", f"image_paths_{split}"], (".txt",))
    class_names_path = find_existing(subject_dir, [f"{split}_class_names", f"class_names_{split}", "class_names"], (".txt",))

    fmri = load_array(fmri_path)
    mask = load_roi_mask(subject_dir, roi, fmri.shape[1])
    if mask is not None:
        fmri = fmri[:, mask]

    class_ids = load_array(class_ids_path).flatten().long()
    image_paths = load_lines(image_paths_path)
    class_names = load_lines(class_names_path)
    if len(image_paths) != fmri.shape[0] or class_ids.shape[0] != fmri.shape[0]:
        raise ValueError(f"Sample count mismatch for {subject} {split}")
    return {
        "image_paths": image_paths,
        "class_ids": class_ids,
        "class_names": class_names,
        "fmri": fmri.float(),
        "subject": subject,
        "split": split,
    }


def auto_build_split(search_root: Path, subject: str, split: str, roi: str) -> dict[str, Any]:
    fmri_path = discover_file(search_root, subject, split, ARRAY_SUFFIXES, ("fmri", "bold", "response", roi.lower()))
    if fmri_path is None:
        raise FileNotFoundError(
            f"Could not find a preprocessed fMRI array for {subject} {split} under {search_root}. "
            "If the figshare files use an unsupported layout, stage files as S*/train_fmri.npy, "
            "S*/train_class_ids.npy, and S*/train_image_paths.txt."
        )
    print(f"FOUND fMRI {fmri_path}")
    fmri = load_array(fmri_path)
    if fmri.ndim != 2:
        fmri = fmri.reshape(fmri.shape[0], -1)

    mask = load_roi_mask(fmri_path.parent, roi, fmri.shape[1])
    if mask is None:
        mask = load_roi_mask(search_root, roi, fmri.shape[1])
    if mask is not None:
        fmri = fmri[:, mask]

    n_samples = int(fmri.shape[0])
    image_paths = discover_strings(search_root, subject, split, ("image", "stimulus", "stimuli", "path"), n_samples)
    if image_paths is None:
        image_paths = fallback_image_paths(subject, split, n_samples)
        print(f"USING stable stimulus IDs for {subject} {split}; real image paths were not found.")

    class_ids = discover_class_ids(search_root, subject, split, n_samples)
    if class_ids is None:
        class_ids = fallback_class_ids(n_samples)
        print(f"USING fallback class ids for {subject} {split}; class metadata was not found.")

    class_names = discover_strings(search_root, subject, split, ("class_name", "classnames", "category_name"), -1)
    if class_names is None:
        class_names = class_names_from_ids(class_ids)

    return {
        "image_paths": image_paths,
        "class_ids": class_ids.long(),
        "class_names": class_names,
        "fmri": fmri.float(),
        "subject": subject,
        "split": split,
    }


def build_split_from_available(raw_dir: Path, subject: str, split: str, roi: str) -> dict[str, Any]:
    subject_dir = raw_dir / subject
    if subject_dir.exists():
        try:
            return build_split(subject_dir, subject, split, roi)
        except FileNotFoundError:
            pass
    search_root = materialize_archives(raw_dir)
    return auto_build_split(raw_dir, subject, split, roi) if raw_dir != search_root else auto_build_split(search_root, subject, split, roi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble DIR files into minimal MRGS processed .pt files.")
    parser.add_argument("--raw-dir", default="data/raw/DIR")
    parser.add_argument("--raw", dest="raw_dir", help="Alias for --raw-dir.")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--out", dest="output_dir", help="Alias for --output-dir.")
    parser.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--roi", default="HVC")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        for split in ("train", "test"):
            payload = build_split_from_available(raw_dir, subject, split, args.roi)
            output = output_dir / f"{subject}_{split}.pt"
            print(
                {
                    "output": str(output),
                    "samples": int(payload["fmri"].shape[0]),
                    "voxels": int(payload["fmri"].shape[1]),
                    "classes": int(payload["class_ids"].unique().numel()),
                }
            )
            if not args.dry_run:
                torch.save(payload, output)


if __name__ == "__main__":
    main()
