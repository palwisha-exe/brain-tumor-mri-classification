from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import METRICS_DIR, MODELS_DIR, PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a model using validation evidence only.")
    parser.add_argument("--candidate", action="append", required=True, help="run_name=checkpoint_path")
    args = parser.parse_args()
    candidates = []
    for value in args.candidate:
        run_name, checkpoint = value.split("=", 1)
        metrics_path = METRICS_DIR / f"{run_name}_validation_metrics.json"
        metrics = json.loads(metrics_path.read_text())
        candidates.append(
            {
                "run_name": run_name,
                "checkpoint": checkpoint,
                "architecture": metrics["architecture"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_accuracy": metrics["accuracy"],
            }
        )
    selected = max(candidates, key=lambda row: (row["validation_macro_f1"], row["validation_accuracy"]))
    payload = {
        "selection_metric": "validation macro-F1; validation accuracy as tie-breaker",
        "test_set_used_for_selection": False,
        "candidates": candidates,
        "selected": selected,
    }
    (MODELS_DIR / "selected_model.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
