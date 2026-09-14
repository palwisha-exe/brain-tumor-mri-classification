# Model Artifacts

Experimental checkpoints are local reproducibility artifacts and are excluded from Git. The only checkpoint required by the Streamlit application is:

```text
models/v2/v2_b0_224_last3_unfreeze_best.pt
```

Expected SHA-256:

```text
26e4b268c112533bf22b6e726625044b0d0ad774fa7d00bea2599e044c36e1d0
```

The public distribution plan is a single GitHub Release asset. After an approved repository and release are created, use:

```bash
python scripts/fetch_final_model.py --url <final-checkpoint-release-url>
```

The helper downloads to a temporary file, verifies the checksum, and only then places the checkpoint at the required path. No external upload or release has been performed yet.

`selected_model_v2.json` is tracked because it records the frozen model identity, configuration, test-artifact paths, and checksum. `selected_model.json` is retained as historical V1 metadata.
