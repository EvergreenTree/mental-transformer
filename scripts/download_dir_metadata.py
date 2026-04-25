from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any


FIGSHARE_ARTICLE_API = "https://api.figshare.com/v2/articles/7033577"
FIGSHARE_PAGE = "https://figshare.com/articles/dataset/Deep_Image_Reconstruction/7033577"
OPENNEURO_PAGE = "https://openneuro.org/datasets/ds001506/versions/1.3.1"


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "mrgs-baseline/0.1"})
    with urllib.request.urlopen(request) as response, output.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def expected_processed_paths(root: Path) -> list[Path]:
    return [root / f"{subject}_{split}.pt" for subject in ("S1", "S2", "S3") for split in ("train", "test")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record DIR source metadata and optionally download figshare files.")
    parser.add_argument("--raw-dir", default="data/raw/DIR")
    parser.add_argument("--out", dest="raw_dir", help="Alias for --raw-dir.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--download", action="store_true", help="Download files listed by the figshare article API.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of figshare files to download.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = raw_dir / "sources.json"
    article = fetch_json(FIGSHARE_ARTICLE_API)
    files = article.get("files", [])
    metadata = {
        "figshare_page": FIGSHARE_PAGE,
        "figshare_api": FIGSHARE_ARTICLE_API,
        "openneuro_page": OPENNEURO_PAGE,
        "openneuro_required": False,
        "title": article.get("title"),
        "files": [
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "size": item.get("size"),
                "md5": item.get("computed_md5") or item.get("supplied_md5"),
                "mimetype": item.get("mimetype"),
                "download_url": item.get("download_url"),
                "local_path": str(raw_dir / item.get("name", "")),
            }
            for item in files
        ],
        "expected_processed_files": [str(path) for path in expected_processed_paths(Path(args.processed_dir))],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {metadata_path}")

    for path in expected_processed_paths(Path(args.processed_dir)):
        print(f"{'OK' if path.exists() else 'MISSING'} {path}")

    if args.download:
        selected = files[: args.limit] if args.limit is not None else files
        for item in selected:
            url = item.get("download_url")
            name = item.get("name")
            if not url or not name:
                continue
            output = raw_dir / name
            if output.exists():
                print(f"SKIP {output}")
                continue
            print(f"DOWNLOADING {name}")
            download_file(url, output)
            print(f"WROTE {output}")


if __name__ == "__main__":
    main()
