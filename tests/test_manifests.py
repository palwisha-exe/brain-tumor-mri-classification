import json

import pandas as pd

from src.config import DATA_DIR, METADATA_DIR, PROJECT_ROOT, SPLITS_DIR


def test_preparation_summary_declares_limitations():
    summary = json.loads((METADATA_DIR / "preparation_summary.json").read_text())
    assert summary["method"] == "image-level leakage-controlled"
    assert summary["patient_level_independence_verified"] is False
    assert summary["final_cross_split_exact_decoded_duplicates"] == 0


def test_final_manifests_are_disjoint_and_valid():
    frames = {name: pd.read_csv(SPLITS_DIR / f"{name}.csv") for name in ("train", "validation", "test")}
    for frame in frames.values():
        assert frame["relative_path"].is_unique
        assert frame["pixel_sha256"].is_unique
        if DATA_DIR.is_dir():
            assert all((PROJECT_ROOT / path).is_file() for path in frame["relative_path"])
        assert not frame["relative_path"].str.contains("aug", case=False).any()
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert set(frames[left]["pixel_sha256"]).isdisjoint(frames[right]["pixel_sha256"])
        assert set(frames[left]["relative_path"]).isdisjoint(frames[right]["relative_path"])
