# Dataset Statement

## Local dataset

The local, unmodified source tree contains 7,200 MRI image files:

| Source folder | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Training | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| Testing | 400 | 400 | 400 | 400 | 1,600 |

The dataset is not redistributed in this repository. Users must obtain any source data under terms applicable to their own download and place it under `data/Training/<class>/` and `data/Testing/<class>/`.

## Attribution verification status

The exact local download provenance is **unverified**.

The local directory structure, class names, counts, and filename patterns strongly match:

- Dataset: *Brain Tumor MRI Dataset*
- Kaggle uploader: Masoud Nickparvar
- Candidate source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

However, retained local records contain no Kaggle `dataset-metadata.json`, archive, source URL, version identifier, or receipt. macOS metadata establishes that files were obtained through Chrome but does not establish the source page. Therefore this project does not claim definitively that the local files are a particular Kaggle version.

The following fields could not be recovered and must not be invented:

- exact Kaggle dataset version or version number;
- exact download/access date;
- license applicable to the downloaded version;
- complete chain of custody from upstream sources to each local file.

The current candidate Kaggle page identifies CC BY 4.0 and describes the collection as combining Figshare, SARTAJ, and Br35H sources, with no-tumor images from Br35H and replacement glioma images from Figshare. Those are claims about the current hosting page, not independently verified facts about this local archive. Confirm the original download record before converting this candidate attribution into a definitive one.

## Empirical audit of the local files

The project's findings are based on the local files and remain valid independently of hosting-page claims:

- 7,200/7,200 files decoded successfully.
- 7,196 decoded as JPEG and four as PNG despite `.jpg` suffixes.
- 111 exact decoded-image duplicate groups crossed the supplied Training/Testing boundary, producing 121 cross-boundary pairs.
- 203 filenames explicitly indicated pre-generated augmentation, including 103 under Testing.
- Internal exact and high-similarity redundancy was present.
- Image modes, dimensions, aspect ratios, and source-related properties varied.
- No patient or study identifiers were available.

The current candidate hosting page may make different statements about deduplication or split overlap. This project's empirical results describe the exact locally audited tree and are not replaced by those page-level claims.

## Derived split

The deterministic preparation process excluded 1,232 files and retained 5,968 representative images in canonical train/validation/test manifests of 4,177/895/896. Exact decoded-image hashes do not cross these splits.

For V2 model development, only canonical train/validation rows originating in the source Training directory were eligible: 3,221 training and 690 validation images. The official test manifest remained 896 images.

This is an **image-level leakage-controlled** split. It is not patient-level independent because patient/study identifiers were unavailable.

## Public redistribution decision

Pending confirmation of the exact source version and license:

- exclude `data/` entirely;
- exclude MRI sample montages, error-case contact sheets, and saved Grad-CAM overlays derived from source images;
- retain numerical manifests, hashes, aggregate metrics, confusion matrices, ROC curves, and training curves that do not reproduce source MRI pixels;
- keep notebook code and statistical outputs, but omit embedded raster outputs from the public notebook.

If provenance is later confirmed, review the verified license and upstream attribution obligations before publishing any MRI-derived visual asset or model checkpoint. This statement is not legal advice.
