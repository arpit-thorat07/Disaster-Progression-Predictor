
import csv
import json
import random
import hashlib
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#CONFIG

ROOT = Path(__file__).resolve().parent

SPLITS = ROOT / "data" / "sen12_splits.csv"

REPORTS = ROOT / "forecast_lab" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

CHECKPOINT = ROOT / "models" / "sen12_classifier_best.pt"
HISTORY = ROOT / "reports" / "sen12_classifier_history.json"

SEED = 42

IMAGE_SIZE = 160
BATCH_SIZE = 12
EPOCHS = 5

LEARNING_RATE = 0.0003

CACHE_DIR = (
    ROOT / "forecast_lab" / "sen12_cache_160"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#DATASET

class SEN12Dataset(Dataset):

    def __init__(self, split):

        with SPLITS.open(
            newline="",
            encoding="utf-8"
        ) as f:
            rows = list(csv.DictReader(f))

        self.rows = [
            row for row in rows
            if row["split"] == split
        ]

        if not self.rows:
            raise RuntimeError(
                f"No observations for split: {split}"
            )

        self.split = split

        CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def read_band(path):

        with rasterio.open(path) as src:
            image = src.read(
                1,
                out_shape=(IMAGE_SIZE, IMAGE_SIZE),
                resampling=Resampling.bilinear,
                masked=True
            )

        values = np.asarray(
            image.data,
            dtype=np.float32
        )

        valid = (
            ~np.ma.getmaskarray(image)
            & np.isfinite(values)
        )

        if valid.sum() < 100:
            raise ValueError(
                f"Insufficient valid pixels: {path}"
            )

        #Signed-log transformation
        values = (
            np.sign(values)
            * np.log1p(np.abs(values))
        )

        valid_values = values[valid]

        lower, upper = np.percentile(
            valid_values,
            [1, 99]
        )

        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
        ):
            raise ValueError(
                f"Invalid radar percentiles: {path}"
            )

        #Safe fallback for constant bands
        if upper <= lower + 1e-8:
            return np.zeros(
                (IMAGE_SIZE, IMAGE_SIZE),
                dtype=np.float32
            )

        values = (
            values - lower
        ) / (upper - lower)

        values = np.clip(
            values,
            0.0,
            1.0
        )

        values = values * 2.0 - 1.0
        values[~valid] = 0.0

        return values.astype(np.float32)

    def __getitem__(self, idx):

        row = self.rows[idx]

        identity = (
            row["vv_path"]
            + "|"
            + row["vh_path"]
            + f"|{IMAGE_SIZE}|signedlog_p1_p99_v1"
        )

        digest = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:24]

        cache_path = (
            CACHE_DIR / f"{digest}.npy"
        )

        if cache_path.exists():

            image = np.load(
                cache_path,
                allow_pickle=False
            ).astype(np.float32)

        else:

            vv = self.read_band(
                ROOT / row["vv_path"]
            )

            vh = self.read_band(
                ROOT / row["vh_path"]
            )

            image = np.stack(
                [vv, vh]
            ).astype(np.float32)

            tmp_path = (
                CACHE_DIR / f"{digest}.tmp.npy"
            )

            with tmp_path.open("wb") as f:
                np.save(
                    f,
                    image.astype(np.float16)
                )

            os.replace(
                tmp_path,
                cache_path
            )

        label = float(
            row["flooding"]
        )

        return (
            torch.from_numpy(
                image.copy()
            ),
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )

#RADAR CNN MODEL

class RadarCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                2, 16, 3, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32, 3, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64, 3, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128, 3, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(
            128, 1
        )

    def forward(self, x):

        features = self.encoder(x)

        return self.classifier(
            features.flatten(1)
        ).squeeze(1)

#CLASSIFICATION METRICS

def compute_metrics(tp, fp, fn, tn):

    def safe_div(a, b):
        return a / b if b else None

    return {
        "accuracy": safe_div(
            tp + tn,
            tp + fp + fn + tn
        ),
        "precision": safe_div(
            tp,
            tp + fp
        ),
        "recall": safe_div(
            tp,
            tp + fn
        ),
        "f1": safe_div(
            2 * tp,
            2 * tp + fp + fn
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }

#VALIDATION

@torch.inference_mode()
def validate(model, loader, criterion):

    model.eval()

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    loss_sum = 0.0
    observations = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        predicted = (
            torch.sigmoid(logits) >= 0.5
        )

        actual = labels.bool()

        tp += int(
            (predicted & actual).sum().item()
        )

        fp += int(
            (predicted & ~actual).sum().item()
        )

        fn += int(
            (~predicted & actual).sum().item()
        )

        tn += int(
            (~predicted & ~actual).sum().item()
        )

        count = labels.numel()

        loss_sum += loss.item() * count
        observations += count

    results = compute_metrics(
        tp, fp, fn, tn
    )

    results["loss"] = (
        loss_sum / observations
    )

    return results

#TRAINING

def main():

    # IMPORTANT FIX:
    # Check for existing results only when this file is executed for training
    # Importing RadarCNN from another script will NOT trigger this check

    if CHECKPOINT.exists() or HISTORY.exists():
        raise FileExistsError(
            "Existing SEN12 training outputs found. "
            "Your trained checkpoint is protected. "
            "Do not overwrite it."
        )

    if DEVICE.type != "cuda":
        raise RuntimeError(
            "CUDA is required for this run."
        )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_ds = SEN12Dataset("train")
    val_ds = SEN12Dataset("val")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = RadarCNN().to(
        DEVICE
    )

    train_positive = sum(
        int(row["flooding"])
        for row in train_ds.rows
    )

    train_negative = (
        len(train_ds) - train_positive
    )

    pos_weight = (
        train_negative / train_positive
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            device=DEVICE
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("SEN12-FLOOD RADAR CLASSIFICATION TRAINING")

    print("Device:", DEVICE)
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print("Train acquisitions:", len(train_ds))
    print("Validation acquisitions:", len(val_ds))

    print(
        "Input shape:",
        (2, IMAGE_SIZE, IMAGE_SIZE)
    )

    print("Epochs:", EPOCHS)

    print(
        "Positive weight:",
        round(pos_weight, 4)
    )

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    best_f1 = -1.0
    best_epoch = None

    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for batch_idx, (
            images,
            labels
        ) in enumerate(
            train_loader,
            start=1
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Nonfinite loss at epoch {epoch}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            count = labels.numel()

            train_loss_sum += (
                loss.item() * count
            )

            train_count += count

            if batch_idx % 20 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.5f}",
                    flush=True
                )

        train_loss = (
            train_loss_sum / train_count
        )

        val = validate(
            model,
            val_loader,
            criterion
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": val
        })

        print(f"\nEPOCH {epoch}/{EPOCHS}")

        print(
            "Train loss:",
            round(train_loss, 5)
        )

        print(
            "Validation loss:",
            round(val["loss"], 5)
        )

        for name in [
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]:

            value = val[name]

            print(
                name + ":",
                None if value is None
                else round(value, 5)
            )

        print(
            "TP/FP/FN/TN:",
            val["tp"],
            val["fp"],
            val["fn"],
            val["tn"]
        )

        current_f1 = (
            val["f1"]
            if val["f1"] is not None
            else -1.0
        )

        if current_f1 > best_f1:

            best_f1 = current_f1
            best_epoch = epoch

            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "validation": val,
                "input_channels": [
                    "VV",
                    "VH"
                ],
                "image_size": IMAGE_SIZE,
                "seed": SEED,
                "architecture": "RadarCNN",
                "preprocessing":
                    "per-image signed-log "
                    "percentile normalization",
            }, CHECKPOINT)

            print(
                "Best checkpoint saved."
            )

        HISTORY.write_text(
            json.dumps(
                history,
                indent=2
            ),
            encoding="utf-8"
        )

    print("SEN12 TRAINING COMPLETED")

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation F1:",
        best_f1
    )

    print(
        "Checkpoint:",
        CHECKPOINT
    )

    print(
        "History:",
        HISTORY
    )


if __name__ == "__main__":
    main()
