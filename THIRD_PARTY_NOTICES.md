# Rights and Third-Party Notices

## Project code

Original source code authored for this repository is licensed under the MIT License; see `LICENSE`.

The MIT License applies only to that original repository code. It does **not** relicense the source MRI dataset, upstream or third-party datasets and materials, pretrained third-party software or weights, or MRI-derived material whose rights are governed separately. Inclusion of metadata, references, interfaces, or factual analysis does not assert ownership of third-party material.

## Source MRI data and derived images

The source MRI dataset is third-party material and is not distributed. Exact provenance and licensing for the local downloaded version remain unverified; see `docs/DATASET.md`.

MRI sample grids, error-case contact sheets, and saved Grad-CAM overlays reproduce or transform source-image pixels. They are excluded from the proposed public release until the applicable source license and attribution obligations are confirmed.

## Pretrained model lineage

The final classifier uses a torchvision EfficientNet-B0 architecture with ImageNet-pretrained initialization. The repository's MIT License does not relicense torchvision, PyTorch, ImageNet source material, pretrained weights, or other third-party components. Users must comply with their applicable terms.

## Final trained checkpoint

The final checkpoint is planned as a separate GitHub Release asset after provenance and redistribution review. Its technical identity is SHA-256 `26e4b268c112533bf22b6e726625044b0d0ad774fa7d00bea2599e044c36e1d0`. Publishing that artifact is a separate rights decision from licensing the original source code.
