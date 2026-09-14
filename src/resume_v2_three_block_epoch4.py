from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import CLASS_NAMES, PROJECT_ROOT, SEED, get_device, set_seed
from src.models import build_model
from src.train_v2 import (
    SplitDataset,
    V2_METRICS_DIR,
    configure_unfreezing,
    run_epoch,
    save_validation_artifacts,
    sha256,
)


RUN_NAME = "v2_b0_224_last3_unfreeze"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "v2" / f"{RUN_NAME}_best.pt"
CONFIGURATION_PATH = V2_METRICS_DIR / f"{RUN_NAME}_configuration.json"
HISTORY_PATH = V2_METRICS_DIR / f"{RUN_NAME}_history.csv"
SUMMARY_PATH = V2_METRICS_DIR / f"{RUN_NAME}_training_summary.json"
RESUME_NOTE_PATH = V2_METRICS_DIR / f"{RUN_NAME}_epoch4_resume_note.json"


def main() -> None:
    configuration = json.loads(CONFIGURATION_PATH.read_text())
    history = pd.read_csv(HISTORY_PATH)
    completed_epochs = history["epoch"].astype(int).tolist()
    if completed_epochs != [1, 2, 3]:
        raise ValueError(f"Expected exactly epochs 1-3 before bounded resume; found {completed_epochs}")
    if SUMMARY_PATH.exists():
        raise FileExistsError("Three-block experiment already has a final summary; refusing to rerun epoch 4.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if checkpoint["best_epoch"] != 3 or checkpoint["run_name"] != RUN_NAME:
        raise ValueError("The saved checkpoint is not the expected epoch 3 three-block checkpoint.")
    if checkpoint["input_size"] != 224 or checkpoint["unfrozen_blocks"] != 3:
        raise ValueError("Unexpected three-block checkpoint configuration.")

    # Optimizer and random-generator state were not included in the checkpoint.
    # Use a fixed, explicitly recorded continuation seed rather than pretending
    # this is a bit-exact continuation of the interrupted process.
    continuation_seed = SEED + 4
    set_seed(continuation_seed)
    device = get_device()
    model = build_model("efficientnet_b0", pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    configure_unfreezing(model, 3)
    model.to(device)

    train_dataset = SplitDataset("train", 224, True, configuration["augmentation"])
    validation_dataset = SplitDataset("validation", 224, False, configuration["augmentation"])
    generator = torch.Generator().manual_seed(continuation_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=configuration["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=configuration["batch_size"], shuffle=False, num_workers=0
    )
    weight_tensor = torch.tensor(configuration["class_weights"], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=configuration["learning_rate"],
        weight_decay=configuration["weight_decay"],
    )

    note = {
        "run_name": RUN_NAME,
        "completed_before_resume": completed_epochs,
        "resumed_epoch": 4,
        "resume_checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "resume_checkpoint_sha256_before_epoch4": sha256(CHECKPOINT_PATH),
        "checkpoint_best_epoch_before_resume": checkpoint["best_epoch"],
        "optimizer_state_restored": False,
        "scheduler_state_restored": False,
        "augmentation_rng_state_restored": False,
        "continuation_seed": continuation_seed,
        "reason": "The interrupted trainer checkpoint saved model weights but not optimizer, scheduler, or RNG state.",
        "test_set_read_or_used": False,
        "original_testing_source_used": False,
    }
    RESUME_NOTE_PATH.write_text(json.dumps(note, indent=2) + "\n")
    print(json.dumps(note, indent=2), flush=True)

    started = time.time()
    train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
    validation_metrics = run_epoch(model, validation_loader, criterion, device)
    epoch_seconds = time.time() - started
    row = {
        "epoch": 4,
        "epoch_seconds": epoch_seconds,
        "learning_rate": configuration["learning_rate"],
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_glioma_precision": validation_metrics["per_class"]["glioma"]["precision"],
        "validation_glioma_recall": validation_metrics["per_class"]["glioma"]["recall"],
        "validation_glioma_f1": validation_metrics["per_class"]["glioma"]["f1"],
        "validation_meningioma_precision": validation_metrics["per_class"]["meningioma"]["precision"],
        "validation_meningioma_recall": validation_metrics["per_class"]["meningioma"]["recall"],
        "validation_meningioma_f1": validation_metrics["per_class"]["meningioma"]["f1"],
        "validation_glioma_as_meningioma": validation_metrics["glioma_as_meningioma"],
        "validation_meningioma_as_glioma": validation_metrics["meningioma_as_glioma"],
        "validation_confusion_matrix": json.dumps(validation_metrics["confusion_matrix"]),
    }
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history.to_csv(HISTORY_PATH, index=False)
    print(json.dumps(row), flush=True)

    if validation_metrics["macro_f1"] > checkpoint["best_validation_macro_f1"] + 1e-5:
        checkpoint["model_state_dict"] = model.state_dict()
        checkpoint["best_validation_macro_f1"] = validation_metrics["macro_f1"]
        checkpoint["best_epoch"] = 4
        checkpoint["continuation_seed"] = continuation_seed
        checkpoint["optimizer_state_restored_for_epoch4"] = False
        torch.save(checkpoint, CHECKPOINT_PATH)

    best = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    summary = {
        **configuration,
        "epochs_completed": 4,
        "best_epoch": best["best_epoch"],
        "best_validation_macro_f1": best["best_validation_macro_f1"],
        "training_time_seconds": float(history["epoch_seconds"].sum()),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT_PATH),
        "interrupted_after_epoch_3": True,
        "epoch_4_optimizer_state_restored": False,
        "epoch_4_continuation_seed": continuation_seed,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    final_metrics = save_validation_artifacts(RUN_NAME, CHECKPOINT_PATH, configuration["batch_size"], device)
    print(json.dumps({"training_summary": summary, "validation_metrics": final_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
