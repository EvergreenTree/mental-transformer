from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


FIGSHARE_ARTICLE_API = "https://api.figshare.com/v2/articles/7033577"
FIGSHARE_PAGE = "https://figshare.com/articles/dataset/Deep_Image_Reconstruction/7033577"
OPENNEURO_PAGE = "https://openneuro.org/datasets/ds001506/versions/1.3.1"


def open_with_retries(url: str, timeout: float, retries: int):
    request = urllib.request.Request(url, headers={"User-Agent": "mrgs-baseline/0.1"})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_s = min(2**attempt, 10)
            print(f"Retrying {url} after {type(exc).__name__}: {exc} ({attempt}/{retries})", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to open {url} after {retries} attempts: {last_error}") from last_error


def fetch_json(url: str, timeout: float, retries: int) -> dict[str, Any]:
    with open_with_retries(url, timeout=timeout, retries=retries) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, output: Path, timeout: float, retries: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    with open_with_retries(url, timeout=timeout, retries=retries) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(output)


def expected_processed_paths(root: Path) -> list[Path]:
    return [root / f"{subject}_{split}.pt" for subject in ("S1", "S2", "S3") for split in ("train", "test")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record DIR source metadata and optionally download figshare files.")
    parser.add_argument("--raw-dir", default="data/raw/DIR")
    parser.add_argument("--out", dest="raw_dir", help="Alias for --raw-dir.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--download", action="store_true", help="Download files listed by the figshare article API.")
    parser.add_argument("--include", default=None, help="Regex filter for file names to download.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of figshare files to download.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Network timeout per request in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Network retry attempts per request.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = raw_dir / "sources.json"
    article = fetch_json(FIGSHARE_ARTICLE_API, timeout=args.timeout, retries=args.retries)
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
    print(f"Wrote {metadata_path}", flush=True)

    for path in expected_processed_paths(Path(args.processed_dir)):
        print(f"{'OK' if path.exists() else 'MISSING'} {path}", flush=True)

    if args.download:
        selected = files
        if args.include is not None:
            pattern = re.compile(args.include)
            selected = [item for item in selected if pattern.search(item.get("name", ""))]
        if args.limit is not None:
            selected = selected[: args.limit]
        print(f"Selected {len(selected)} files for download", flush=True)
        for item in selected:
            url = item.get("download_url")
            name = item.get("name")
            if not url or not name:
                continue
            output = raw_dir / name
            if output.exists():
                print(f"SKIP {output}", flush=True)
                continue
            print(f"DOWNLOADING {name}", flush=True)
            download_file(url, output, timeout=args.timeout, retries=args.retries)
            print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
