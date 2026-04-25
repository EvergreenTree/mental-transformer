from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_member_target(root: Path, member_name: str) -> Path:
    target = root / member_name
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return target


def extract_synset_archive(path: Path, output_root: Path, max_images: int | None) -> int:
    synset = path.stem
    target_dir = output_root / synset
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue
            if max_images is not None and count >= max_images:
                break
            name = Path(member.name).name
            target = safe_member_target(target_dir, name)
            if target.exists() and target.stat().st_size > 0:
                count += 1
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with extracted, target.open("wb") as handle:
                handle.write(extracted.read())
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a bounded number of images from ImageNet synset archives.")
    parser.add_argument("--archives", required=True, help="Directory containing n*.tar synset archives.")
    parser.add_argument("--out", required=True, help="Output image directory.")
    parser.add_argument("--max-images-per-synset", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_dir = Path(args.archives)
    output_root = Path(args.out)
    archives = sorted(archive_dir.glob("n*.tar"))
    if not archives:
        raise FileNotFoundError(f"No synset .tar archives found under {archive_dir}")
    counts = {}
    for path in archives:
        counts[path.stem] = extract_synset_archive(path, output_root, args.max_images_per_synset)
        print(f"EXTRACTED {path.stem}: {counts[path.stem]}")
    print({"archives": len(archives), "images": sum(counts.values()), "out": str(output_root)})


if __name__ == "__main__":
    main()
