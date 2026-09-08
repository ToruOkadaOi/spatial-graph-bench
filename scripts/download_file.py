"""Robust file downloader supporting automatic byte-range resume and retry."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_with_resume(url: str, dest_path: Path, max_retries: int = 50) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe total size
    total_size = 0
    for _ in range(5):
        try:
            head = requests.head(url, verify=False, timeout=15)
            if "content-length" in head.headers:
                total_size = int(head.headers["content-length"])
                break
        except Exception:
            time.sleep(1)

    print(f"Target: {dest_path} (Total size: {total_size / (1024 * 1024):.1f} MB)")
    retries = 0

    while retries < max_retries:
        current_size = dest_path.stat().st_size if dest_path.exists() else 0
        if total_size > 0 and current_size >= total_size:
            print(f"✓ Download complete: {dest_path} ({current_size / (1024 * 1024):.1f} MB)")
            return

        headers = {}
        if current_size > 0:
            headers["Range"] = f"bytes={current_size}-"
            print(
                f"Resuming from {current_size / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB ({current_size / total_size:.1%})..."
            )
        else:
            print(f"Starting download ({total_size / (1024 * 1024):.1f} MB)...")

        try:
            with requests.get(url, headers=headers, stream=True, verify=False, timeout=20) as r:
                if r.status_code not in (200, 206):
                    print(f"HTTP {r.status_code}, retrying in 2s...")
                    time.sleep(2)
                    retries += 1
                    continue
                with open(dest_path, "ab" if current_size > 0 else "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            f.write(chunk)
            retries = 0  # reset on progress
        except Exception as e:
            print(f"Connection dropped ({type(e).__name__}: {e}), retrying in 2s...")
            time.sleep(2)
            retries += 1

    raise RuntimeError(f"Failed to download after {max_retries} retries: {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL to download")
    parser.add_argument("dest", help="Destination file path")
    args = parser.parse_args()

    download_with_resume(args.url, Path(args.dest))
