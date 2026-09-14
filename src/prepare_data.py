from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ExifTags, Image
from scipy.fftpack import dct
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from src.config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    DATA_DIR,
    METADATA_DIR,
    PROJECT_ROOT,
    SEED,
    SPLITS_DIR,
    ensure_directories,
    save_class_mapping,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_statistics(image: Image.Image) -> dict[str, float]:
    gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
    border = np.concatenate((gray[:8, :].ravel(), gray[-8:, :].ravel(), gray[8:-8, :8].ravel(), gray[8:-8, -8:].ravel()))
    center = gray[32:96, 32:96]
    return {
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
        "border_mean": float(border.mean()),
        "center_mean": float(center.mean()),
        "border_to_center_ratio": float(border.mean() / max(center.mean(), 1e-6)),
        "dark_border_fraction": float((border < 0.03).mean()),
        "bright_border_fraction": float((border > 0.92).mean()),
    }


def perceptual_hash_256(image: Image.Image) -> str:
    gray = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32)
    coefficients = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")[:16, :16]
    values = coefficients.flatten()
    threshold = np.median(values[1:])
    bits = values > threshold
    bits[0] = False
    return np.packbits(bits.astype(np.uint8)).tobytes().hex()


def scan_dataset() -> pd.DataFrame:
    rows: list[dict] = []
    for original_split in ("Training", "Testing"):
        split_dir = DATA_DIR / original_split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing expected dataset folder: {split_dir}")
        discovered = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        if discovered != sorted(CLASS_NAMES):
            raise ValueError(f"Unexpected classes in {original_split}: {discovered}")
        for class_name in CLASS_NAMES:
            for path in sorted((split_dir / class_name).iterdir()):
                if not path.is_file():
                    continue
                relative_path = path.relative_to(PROJECT_ROOT).as_posix()
                row = {
                    "relative_path": relative_path,
                    "filename": path.name,
                    "class_name": class_name,
                    "class_index": CLASS_TO_IDX[class_name],
                    "original_split": original_split,
                    "is_pre_generated_augmented": "aug" in path.stem.lower(),
                    "file_size_bytes": path.stat().st_size,
                    "raw_sha256": sha256_file(path),
                    "corrupted": False,
                    "error": "",
                }
                try:
                    with Image.open(path) as probe:
                        probe.verify()
                    with Image.open(path) as image:
                        image.load()
                        rgb = image.convert("RGB")
                        pixel_payload = f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()
                        exif_tags = sorted(ExifTags.TAGS.get(key, str(key)) for key in image.getexif())
                        row.update(
                            {
                                "width": image.width,
                                "height": image.height,
                                "aspect_ratio": image.width / image.height,
                                "decoded_format": image.format,
                                "original_mode": image.mode,
                                "pixel_sha256": hashlib.sha256(pixel_payload).hexdigest(),
                                "phash256": perceptual_hash_256(image),
                                "has_exif": bool(exif_tags),
                                "exif_tags": "|".join(exif_tags),
                                **image_statistics(image),
                            }
                        )
                except Exception as exc:  # retained in the audit manifest
                    row.update({"corrupted": True, "error": repr(exc)})
                rows.append(row)
    return pd.DataFrame(rows)


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def gray_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32) / 255.0


def similarity_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_centered = left.ravel() - left.mean()
    right_centered = right.ravel() - right.mean()
    correlation = float(
        np.dot(left_centered, right_centered)
        / max(np.linalg.norm(left_centered) * np.linalg.norm(right_centered), 1e-8)
    )
    left_mean = gaussian_filter(left, 1.5)
    right_mean = gaussian_filter(right, 1.5)
    left_variance = gaussian_filter(left * left, 1.5) - left_mean * left_mean
    right_variance = gaussian_filter(right * right, 1.5) - right_mean * right_mean
    covariance = gaussian_filter(left * right, 1.5) - left_mean * right_mean
    ssim_map = (
        (2 * left_mean * right_mean + 0.01**2) * (2 * covariance + 0.03**2)
        / ((left_mean**2 + right_mean**2 + 0.01**2) * (left_variance + right_variance + 0.03**2))
    )
    return correlation, float(ssim_map.mean())


def build_duplicate_components(frame: pd.DataFrame, eligible_indices) -> tuple[dict, pd.DataFrame]:
    indices = list(eligible_indices)
    union = UnionFind(indices)
    for _, group in frame.loc[indices].groupby("pixel_sha256"):
        first = group.index[0]
        for other in group.index[1:]:
            union.union(first, other)

    bit_matrix = np.stack(
        [np.unpackbits(np.frombuffer(bytes.fromhex(value), dtype=np.uint8)) for value in frame.loc[indices, "phash256"]]
    )
    neighbor_model = NearestNeighbors(radius=4 / 256, metric="hamming", algorithm="brute", n_jobs=-1).fit(bit_matrix)
    distances, neighbors = neighbor_model.radius_neighbors(bit_matrix, sort_results=True)
    gray_cache = {}
    edges = []
    for local_left, (local_neighbors, local_distances) in enumerate(zip(neighbors, distances)):
        left = indices[local_left]
        for local_right, distance in zip(local_neighbors, local_distances):
            if local_right <= local_left:
                continue
            right = indices[int(local_right)]
            phash_distance = int(round(float(distance) * 256))
            if frame.at[left, "pixel_sha256"] == frame.at[right, "pixel_sha256"]:
                correlation = 1.0
                accepted = True
                method = "exact_decoded_rgb"
            else:
                if left not in gray_cache:
                    gray_cache[left] = gray_array(PROJECT_ROOT / frame.at[left, "relative_path"])
                if right not in gray_cache:
                    gray_cache[right] = gray_array(PROJECT_ROOT / frame.at[right, "relative_path"])
                correlation, structural_similarity = similarity_metrics(gray_cache[left], gray_cache[right])
                accepted = (
                    (phash_distance == 0 and correlation >= 0.98)
                    or (phash_distance <= 2 and correlation >= 0.975 and structural_similarity >= 0.82)
                    or (phash_distance <= 4 and correlation >= 0.96 and structural_similarity >= 0.84)
                )
                method = "perceptual_phash_ncc"
            if frame.at[left, "pixel_sha256"] == frame.at[right, "pixel_sha256"]:
                structural_similarity = 1.0
            if accepted:
                union.union(left, right)
                edges.append(
                    {
                        "path_a": frame.at[left, "relative_path"],
                        "class_a": frame.at[left, "class_name"],
                        "path_b": frame.at[right, "relative_path"],
                        "class_b": frame.at[right, "class_name"],
                        "method": method,
                        "phash_distance": phash_distance,
                        "normalized_pixel_correlation": correlation,
                        "structural_similarity": structural_similarity,
                        "label_conflict": frame.at[left, "class_name"] != frame.at[right, "class_name"],
                    }
                )
    components = defaultdict(list)
    for index in indices:
        components[union.find(index)].append(index)
    return components, pd.DataFrame(edges)


def assign_decisions(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["status"] = "candidate"
    frame["exclusion_reason"] = ""
    frame["duplicate_of"] = ""
    frame["final_split"] = "excluded"

    corrupt = frame["corrupted"].fillna(False)
    frame.loc[corrupt, ["status", "exclusion_reason"]] = ["excluded", "corrupted_or_unreadable"]

    pre_aug = frame["is_pre_generated_augmented"] & ~corrupt
    frame.loc[pre_aug, ["status", "exclusion_reason"]] = ["excluded", "pre_generated_augmentation_unknown_provenance"]

    eligible = frame.index[frame["status"].eq("candidate")]
    components, edges = build_duplicate_components(frame, eligible)
    frame["duplicate_component"] = ""
    for component_number, members in enumerate(sorted(components.values(), key=lambda values: min(values)), start=1):
        component_id = f"dup_{component_number:05d}"
        frame.loc[members, "duplicate_component"] = component_id
        if len(members) == 1:
            continue
        group = frame.loc[members].sort_values("relative_path")
        if group["class_name"].nunique() > 1:
            frame.loc[members, "status"] = "excluded"
            frame.loc[members, "exclusion_reason"] = "perceptual_duplicate_group_with_label_conflict"
            continue
        representative = group.iloc[0]
        duplicates = group.iloc[1:]
        frame.loc[duplicates.index, "status"] = "excluded"
        exact = duplicates["pixel_sha256"].eq(representative["pixel_sha256"])
        frame.loc[duplicates.index[exact], "exclusion_reason"] = "redundant_exact_decoded_rgb_duplicate"
        frame.loc[duplicates.index[~exact], "exclusion_reason"] = "redundant_high_similarity_perceptual_duplicate"
        frame.loc[duplicates.index, "duplicate_of"] = representative["relative_path"]

    included = frame[frame["status"].eq("candidate")].copy()
    train_indices, holdout_indices = train_test_split(
        included.index,
        test_size=0.30,
        random_state=seed,
        stratify=included["class_name"],
    )
    val_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=0.50,
        random_state=seed + 1,
        stratify=included.loc[holdout_indices, "class_name"],
    )
    frame.loc[train_indices, ["status", "final_split"]] = ["included", "train"]
    frame.loc[val_indices, ["status", "final_split"]] = ["included", "validation"]
    frame.loc[test_indices, ["status", "final_split"]] = ["included", "test"]
    return frame, edges


def cross_original_duplicate_summary(frame: pd.DataFrame) -> tuple[int, int]:
    groups = 0
    pairs = 0
    valid = frame[~frame["corrupted"]]
    for _, group in valid.groupby("pixel_sha256"):
        training = int(group["original_split"].eq("Training").sum())
        testing = int(group["original_split"].eq("Testing").sum())
        if training and testing:
            groups += 1
            pairs += training * testing
    return groups, pairs


def validate_final_splits(frame: pd.DataFrame) -> None:
    included = frame[frame["status"].eq("included")]
    assert not included["is_pre_generated_augmented"].any()
    assert included["relative_path"].is_unique
    assert included["pixel_sha256"].is_unique
    assert set(included["final_split"]) == {"train", "validation", "test"}
    split_hashes = {name: set(group["pixel_sha256"]) for name, group in included.groupby("final_split")}
    assert split_hashes["train"].isdisjoint(split_hashes["validation"])
    assert split_hashes["train"].isdisjoint(split_hashes["test"])
    assert split_hashes["validation"].isdisjoint(split_hashes["test"])


def save_outputs(frame: pd.DataFrame, edges: pd.DataFrame, seed: int) -> dict:
    ensure_directories()
    save_class_mapping()
    output = frame.sort_values(["status", "final_split", "class_name", "relative_path"]).reset_index(drop=True)
    output.to_csv(METADATA_DIR / "dataset_manifest.csv", index=False)
    exclusions = output[output["status"].eq("excluded")].copy()
    exclusions.to_csv(METADATA_DIR / "exclusions.csv", index=False)
    edges.to_csv(METADATA_DIR / "perceptual_duplicate_edges.csv", index=False)
    included = output[output["status"].eq("included")].copy()
    manifest_columns = [
        "relative_path", "class_name", "class_index", "final_split", "pixel_sha256",
        "width", "height", "aspect_ratio", "decoded_format", "original_mode", "original_split",
    ]
    for split_name in ("train", "validation", "test"):
        split = included[included["final_split"].eq(split_name)][manifest_columns]
        split.to_csv(SPLITS_DIR / f"{split_name}.csv", index=False)

    cross_groups, cross_pairs = cross_original_duplicate_summary(output)
    summary = {
        "method": "image-level leakage-controlled",
        "patient_level_independence_verified": False,
        "seed": seed,
        "original_images": int(len(output)),
        "original_class_counts": output["class_name"].value_counts().sort_index().to_dict(),
        "original_split_counts": output["original_split"].value_counts().sort_index().to_dict(),
        "corrupted_images": int(output["corrupted"].sum()),
        "pre_generated_augmented_excluded": int((output["exclusion_reason"] == "pre_generated_augmentation_unknown_provenance").sum()),
        "redundant_decoded_duplicates_excluded": int((output["exclusion_reason"] == "redundant_exact_decoded_rgb_duplicate").sum()),
        "redundant_perceptual_duplicates_excluded": int((output["exclusion_reason"] == "redundant_high_similarity_perceptual_duplicate").sum()),
        "label_conflict_group_images_excluded": int((output["exclusion_reason"] == "perceptual_duplicate_group_with_label_conflict").sum()),
        "accepted_duplicate_edges": int(len(edges)),
        "accepted_cross_label_edges": int(edges["label_conflict"].sum()) if not edges.empty else 0,
        "total_excluded": int(len(exclusions)),
        "total_included_unique": int(len(included)),
        "final_split_counts": included["final_split"].value_counts().sort_index().to_dict(),
        "final_split_class_counts": {
            split: group["class_name"].value_counts().sort_index().to_dict()
            for split, group in included.groupby("final_split")
        },
        "original_cross_split_decoded_duplicate_groups": cross_groups,
        "original_cross_split_decoded_duplicate_pairs": cross_pairs,
        "final_cross_split_exact_decoded_duplicates": 0,
        "classes": list(CLASS_NAMES),
        "class_to_index": CLASS_TO_IDX,
        "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "selection_rule": "Exclude all pre-generated augmented files. Build duplicate components from exact decoded-RGB equality and visually reviewed high-similarity rules: pHash distance 0 with correlation >=0.98; distance <=2 with correlation >=0.975 and SSIM >=0.82; or distance <=4 with correlation >=0.96 and SSIM >=0.84. Exclude every class-conflicting component; otherwise retain the lexicographically first path. Stratify representatives with fixed seeds.",
    }
    (METADATA_DIR / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-controlled image-level manifests without modifying source images.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    ensure_directories()
    scanned = scan_dataset()
    decided, edges = assign_decisions(scanned, args.seed)
    validate_final_splits(decided)
    summary = save_outputs(decided, edges, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
