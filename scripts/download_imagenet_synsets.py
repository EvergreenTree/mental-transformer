from __future__ import annotations

import argparse
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path


URL_TEMPLATE = "https://image-net.org/data/winter21_whole/{synset}.tar"


def read_synsets(path: Path) -> list[str]:
    synsets: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        synsets.append(value)
    return synsets


def read_cookie_header(cookies: Path) -> str:
    text = cookies.read_text(encoding="utf-8").strip()
    pairs: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if len(parts) >= 7:
            pairs.append(f"{parts[5]}={parts[6]}")
    if pairs:
        return "; ".join(pairs)
    return text


def build_opener(cookies: Path | None) -> urllib.request.OpenerDirector:
    opener = urllib.request.build_opener()
    if cookies is not None:
        opener.addheaders.append(("Cookie", read_cookie_header(cookies)))
    opener.addheaders.append(("User-Agent", "mrgs-imagenet-subset/0.1"))
    return opener


def download_synset(opener: urllib.request.OpenerDirector, synset: str, output: Path, timeout: float) -> None:
    url = URL_TEMPLATE.format(synset=synset)
    partial = output.with_suffix(output.suffix + ".part")
    output.parent.mkdir(parents=True, exist_ok=True)
    with opener.open(url, timeout=timeout) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty archive for {synset}")
    partial.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected ImageNet synset tar archives.")
    parser.add_argument("--synsets", required=True, help="Text file with one ImageNet synset ID per line.")
    parser.add_argument("--out", required=True, help="Output directory for synset .tar archives.")
    parser.add_argument("--cookies", default=None, help="Optional file containing a Cookie header value.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between downloads.")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synsets = read_synsets(Path(args.synsets))
    if args.limit is not None:
        synsets = synsets[: args.limit]
    output_dir = Path(args.out)
    opener = build_opener(Path(args.cookies) if args.cookies else None)
    failed: list[str] = []
    for index, synset in enumerate(synsets, start=1):
        output = output_dir / f"{synset}.tar"
        if output.exists() and output.stat().st_size > 0:
            print(f"SKIP {synset}: {output}")
            continue
        url = URL_TEMPLATE.format(synset=synset)
        if args.dry_run:
            print(f"DRY-RUN {synset}: {url} -> {output}")
            continue
        try:
            print(f"DOWNLOAD {index}/{len(synsets)} {synset}: {url}")
            download_synset(opener, synset, output, timeout=args.timeout)
            print(f"WROTE {output}")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            failed.append(synset)
            output.with_suffix(output.suffix + ".part").unlink(missing_ok=True)
            print(f"FAILED {synset}: {type(exc).__name__}: {exc}")
        if args.sleep > 0 and index < len(synsets):
            time.sleep(args.sleep)
    print({"requested": len(synsets), "failed": failed})
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
