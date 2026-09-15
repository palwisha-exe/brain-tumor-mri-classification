#!/usr/bin/env python3
"""Download the frozen final checkpoint and verify its immutable SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "models" / "v2" / "v2_b0_224_last3_unfreeze_best.pt"
EXPECTED_SHA256 = "26e4b268c112533bf22b6e726625044b0d0ad774fa7d00bea2599e044c36e1d0"
DEFAULT_URL = (
    "https://github.com/palwisha-exe/brain-tumor-mri-classification/"
    "releases/download/v2.0.0/v2_b0_224_last3_unfreeze_best.pt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_final_model(destination: Path = DESTINATION, url: str = DEFAULT_URL) -> Path:
    """Return the verified V2 checkpoint, downloading the approved asset when absent."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("The checkpoint URL must be an absolute HTTPS URL.")

    if destination.exists():
        observed = sha256(destination)
        if observed == EXPECTED_SHA256:
            print(f"Verified existing checkpoint: {destination.relative_to(PROJECT_ROOT)}")
            return destination
        raise FileExistsError(
            f"Refusing to overwrite {destination}; its SHA-256 is {observed}, not the expected value."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "brain-tumor-mri-ai-release-fetcher/1"})
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(prefix="final-model-", suffix=".pt", delete=False, dir=destination.parent) as target:
            temporary_path = Path(target.name)
            with urlopen(request, timeout=120) as source:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)

        observed = sha256(temporary_path)
        if observed != EXPECTED_SHA256:
            raise ValueError(f"Downloaded checkpoint SHA-256 mismatch: {observed}")
        os.replace(temporary_path, destination)
        temporary_path = None
        print(f"Installed verified checkpoint: {destination.relative_to(PROJECT_ROOT)}")
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="HTTPS URL of the approved GitHub Release asset",
    )
    args = parser.parse_args()
    ensure_final_model(url=args.url)


if __name__ == "__main__":
    main()
