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

## BodyParts3D brain anatomy asset

The interactive empty-state brain viewer uses `app/assets/bodyparts3d_brain_prototype.glb`, a derivative of selected anatomical polygon meshes from BodyParts3D Release 4.0.

Required source attribution:

> BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International

- Source: [BodyParts3D, © The Database Center for Life Science](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/)
- Dataset DOI: [10.18908/lsdba.nbdc00837-000](https://doi.org/10.18908/lsdba.nbdc00837-000)
- Polygon archive DOI: [10.18908/lsdba.nbdc00837-007](https://doi.org/10.18908/lsdba.nbdc00837-007)
- Source archive: `isa_BP3D_4.0_obj_99.zip`
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Changes made: Forty-four cerebral, cerebellar, and brainstem element meshes were selected and combined. Skull, vessels, ventricles, nerves, labels, and unrelated anatomy were excluded. Coincident vertices were welded, unused vertices were removed, winding and normals were made consistent, and the Wavefront OBJ geometry was converted to glTF Binary (GLB). Geometry was quantized and compressed with `EXT_meshopt_compression` without polygon decimation. The viewer applies a neutral display material.

The derived anatomy asset remains licensed under CC BY 4.0 and is not covered by the MIT License for this repository's original source code. It is provided for research and educational visualization only and is not validated for diagnosis, surgical planning, segmentation, or other clinical use. BodyParts3D notes that its data may contain errors and should not be treated as a canonical anatomical model.

## Three.js

The local brain-viewer JavaScript bundle contains Three.js 0.180.0, including Three.js core, OrbitControls, GLTFLoader, and MeshoptDecoder.

- Source: [three.js](https://github.com/mrdoob/three.js)
- License: MIT
- Bundled license text: `app/assets/THREE_LICENSE.txt`

Three.js is third-party software and is not relicensed by this repository's MIT License.
