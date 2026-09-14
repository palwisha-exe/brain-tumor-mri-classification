from __future__ import annotations

import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy.fftpack import dct
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors

from src.config import FIGURES_DIR, METADATA_DIR, PROJECT_ROOT, REPORTS_DIR, ensure_directories


def perceptual_hash(path, size: int = 64, low_frequency: int = 16) -> np.ndarray:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((size, size)), dtype=np.float32)
    coefficients = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")[:low_frequency, :low_frequency]
    values = coefficients.flatten()
    threshold = np.median(values[1:])
    bits = values > threshold
    bits[0] = False
    return bits.astype(np.uint8)


def similarity(path_a, path_b) -> tuple[float, float]:
    vectors = []
    for path in (path_a, path_b):
        with Image.open(path) as image:
            values = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32).ravel()
        vectors.append(values / 255.0)
    left, right = vectors
    correlation = float(np.corrcoef(left.ravel(), right.ravel())[0, 1])
    left_mean = gaussian_filter(left, 1.5); right_mean = gaussian_filter(right, 1.5)
    left_var = gaussian_filter(left * left, 1.5) - left_mean * left_mean
    right_var = gaussian_filter(right * right, 1.5) - right_mean * right_mean
    covariance = gaussian_filter(left * right, 1.5) - left_mean * right_mean
    ssim = float(np.mean(((2 * left_mean * right_mean + 0.01**2) * (2 * covariance + 0.03**2)) /
                         ((left_mean**2 + right_mean**2 + 0.01**2) * (left_var + right_var + 0.03**2))))
    return correlation, ssim


def create_montage(pairs: pd.DataFrame) -> None:
    shown = pairs.head(12)
    canvas = Image.new("RGB", (900, max(1, len(shown)) * 220), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (_, row) in enumerate(shown.iterrows()):
        for col, key in enumerate(("path_a", "path_b")):
            with Image.open(PROJECT_ROOT / row[key]) as source:
                tile = ImageOps.contain(source.convert("RGB"), (300, 180))
            x = 10 + col * 330
            y = row_index * 220
            canvas.paste(tile, (x + (300 - tile.width) // 2, y))
        label = (
            f"d={int(row['phash_distance'])}/256\n"
            f"{row['split_a']} {row['class_a']}\n{row['filename_a']}\n"
            f"{row['split_b']} {row['class_b']}\n{row['filename_b']}"
        )
        draw.multiline_text((670, row_index * 220 + 15), label, fill="black", spacing=4)
    canvas.save(FIGURES_DIR / "near_duplicate_candidates.jpg", quality=92)


def main() -> None:
    ensure_directories()
    manifest = pd.read_csv(METADATA_DIR / "dataset_manifest.csv")
    frame = manifest[manifest["status"].eq("included")].reset_index(drop=True)
    hashes = np.stack([perceptual_hash(PROJECT_ROOT / path) for path in frame["relative_path"]])
    pairs = []
    split_names = ("train", "validation", "test")
    for split_a in split_names:
        a_indices = frame.index[frame["final_split"].eq(split_a)].to_numpy()
        for split_b in split_names:
            if split_a >= split_b:
                continue
            b_indices = frame.index[frame["final_split"].eq(split_b)].to_numpy()
            model = NearestNeighbors(n_neighbors=1, metric="hamming").fit(hashes[b_indices])
            distances, neighbors = model.kneighbors(hashes[a_indices])
            for source_idx, distance, neighbor in zip(a_indices, distances[:, 0], neighbors[:, 0]):
                target_idx = b_indices[neighbor]
                a = frame.loc[source_idx]
                b = frame.loc[target_idx]
                correlation, structural_similarity = similarity(PROJECT_ROOT / a["relative_path"], PROJECT_ROOT / b["relative_path"])
                phash_distance = int(round(distance * hashes.shape[1]))
                meets_rule = (
                    (phash_distance == 0 and correlation >= 0.98)
                    or (phash_distance <= 2 and correlation >= 0.975 and structural_similarity >= 0.82)
                    or (phash_distance <= 4 and correlation >= 0.96 and structural_similarity >= 0.84)
                )
                pairs.append(
                    {
                        "split_a": split_a,
                        "class_a": a["class_name"],
                        "filename_a": a["filename"],
                        "path_a": a["relative_path"],
                        "split_b": split_b,
                        "class_b": b["class_name"],
                        "filename_b": b["filename"],
                        "path_b": b["relative_path"],
                        "phash_distance": phash_distance,
                        "same_class": bool(a["class_name"] == b["class_name"]),
                        "normalized_pixel_correlation": correlation,
                        "structural_similarity": structural_similarity,
                        "meets_exclusion_rule": bool(meets_rule),
                    }
                )
    result = pd.DataFrame(pairs).sort_values(["phash_distance", "path_a", "path_b"]).reset_index(drop=True)
    result.to_csv(REPORTS_DIR / "near_duplicate_nearest_pairs.csv", index=False)
    create_montage(result)
    summary = {
        "hash": "256-bit DCT perceptual hash",
        "minimum_cross_split_distance": int(result["phash_distance"].min()),
        "pairs_at_or_below_2_bits": int((result["phash_distance"] <= 2).sum()),
        "pairs_at_or_below_4_bits": int((result["phash_distance"] <= 4).sum()),
        "pairs_meeting_full_duplicate_rule": int(result["meets_exclusion_rule"].sum()),
        "note": "Perceptual similarity is a screening signal, not proof of patient identity; candidates require visual review.",
    }
    (REPORTS_DIR / "near_duplicate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
