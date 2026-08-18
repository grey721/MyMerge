#!/usr/bin/env python3
"""Parallel HTTP-range downloader for large Hugging Face model shards."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple

import requests
from huggingface_hub import HfApi


def ranges(size: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, size, chunk_size):
        yield start, min(size - 1, start + chunk_size - 1)


def download_range(
    url: str,
    fd: int,
    start: int,
    end: int,
    retries: int,
) -> int:
    expected = end - start + 1
    for attempt in range(retries):
        try:
            with requests.get(
                url,
                headers={
                    "Range": "bytes={}-{}".format(start, end),
                    "User-Agent": "AgentMerge-range-downloader/0.1",
                },
                stream=True,
                timeout=(30, 120),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        "Expected HTTP 206, got {}".format(response.status_code)
                    )
                position = start
                for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if block:
                        os.pwrite(fd, block, position)
                        position += len(block)
                if position - start != expected:
                    raise RuntimeError(
                        "Short range: expected {}, got {}".format(
                            expected, position - start
                        )
                    )
                return expected
        except Exception:
            if attempt + 1 >= retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    repo_id: str,
    revision: str,
    filename: str,
    size: int,
    expected_sha: str,
    output_dir: Path,
    workers: int,
    chunk_mib: int,
) -> None:
    output = output_dir / filename
    if output.exists() and output.stat().st_size == size:
        current_sha = sha256(output)
        if current_sha == expected_sha:
            print("{} already verified".format(filename), flush=True)
            return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".range-part")
    if temporary.exists():
        temporary.unlink()
    url = "https://huggingface.co/{}/resolve/{}/{}".format(
        repo_id, revision, filename
    )
    fd = os.open(str(temporary), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        os.ftruncate(fd, size)
        chunks = list(ranges(size, chunk_mib * 1024 * 1024))
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(download_range, url, fd, start, end, 5)
                for start, end in chunks
            ]
            for future in as_completed(futures):
                completed += future.result()
                print(
                    "{}: {:.1f}%".format(
                        filename, 100.0 * completed / size
                    ),
                    flush=True,
                )
        os.fsync(fd)
    finally:
        os.close(fd)
    actual_sha = sha256(temporary)
    if actual_sha != expected_sha:
        raise RuntimeError(
            "{} SHA256 mismatch: {} != {}".format(
                filename, actual_sha, expected_sha
            )
        )
    temporary.replace(output)
    print("{} verified: {}".format(filename, actual_sha), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument(
        "--suffix", default=".safetensors", help="File suffix to download"
    )
    args = parser.parse_args()

    info = HfApi().model_info(
        args.repo_id, revision=args.revision, files_metadata=True
    )
    files = [
        sibling
        for sibling in info.siblings
        if sibling.rfilename.endswith(args.suffix) and sibling.lfs is not None
    ]
    for sibling in files:
        download_file(
            args.repo_id,
            args.revision,
            sibling.rfilename,
            int(sibling.size),
            sibling.lfs.sha256,
            args.output_dir,
            args.workers,
            args.chunk_mib,
        )


if __name__ == "__main__":
    main()
