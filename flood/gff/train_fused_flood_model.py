
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader, Subset

import gff.dataloaders
import gff.models.creation
import gff.training
import gff.util

from test_fused_flood_model import (
    FusedFloodModel,
    load_models,
)

#CONFIG

ROOT = Path(__file__).resolve().parent

BASELINE_RUN = ROOT

CONFIG_FILE = ROOT / "configs" / "config_utae.yml"
SUBSET_FILE = ROOT / "data" / "subset_indices.json"

RUNS = ROOT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

EPOCHS = 3
BATCH_SIZE = 1
SEED = 6426452

#FUSION FINE-TUNING MODEL

class TrainableFusion(FusedFloodModel):

    def train(self, mode=True):
        super().train(mode)

        #Keep pretrained encoder statistics fixed
        self.sen12_encoder.eval()

        #Keep the rest of the pretrained GFF model in inference mode so its BatchNorm runningstatistics do not change
        self.gff.eval()

        #The new adapter remains trainable
        self.adapter.train(mode)

        return self

# FREEZE MOST PRETRAINED WEIGHTS

def configure_trainable_parameters(model):

    for parameter in model.parameters():
        parameter.requires_grad = False

    #New SEN12 projection
    for parameter in model.adapter.parameters():
        parameter.requires_grad = True

    #First U-TAE convolution, expanded from 5 to 6 input channels
    first_conv = (
        model.gff.local_embed
        .in_conv.conv.conv[0]
    )

    assert first_conv.in_channels == 6

    first_conv.weight.requires_grad_(True)

    #Bias stays frozen: original initialization was copied from the pretrained GFF model
    if first_conv.bias is not None:
        first_conv.bias.requires_grad_(False)

    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    print("\nTRAINABLE PARAMETERS")

    for name in trainable_names:
        print(" ", name)

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nTrainable:", f"{trainable:,}")
    print("Total:", f"{total:,}")

    assert trainable > 0
    assert trainable < total

    return first_conv

# DATASET

def create_dataloaders(config):

    folder = Path(config["data_folder"])

    train_files, val_files, _ = (
        gff.dataloaders.read_partitions(
            folder,
            config["fold"]
        )
    )

    #No test partition is loaded
    train_ds = gff.dataloaders.FloodForecastDataset(
        folder,
        config,
        meta_fnames=train_files,
        label_sources=config["train_label_sources"]
    )

    val_ds = gff.dataloaders.FloodForecastDataset(
        folder,
        config,
        meta_fnames=val_files
    )

    #Reuse exactly the selected tiles from the original GFF experiment
    with SUBSET_FILE.open(encoding="utf-8") as f:
        subset_info = json.load(f)

    train_indices = subset_info["train_indices"]
    val_indices = subset_info["val_indices"]

    assert len(train_indices) == 1024
    assert len(val_indices) == 256

    assert max(train_indices) < len(train_ds)
    assert max(val_indices) < len(val_ds)

    train_subset = Subset(
        train_ds,
        train_indices
    )

    val_subset = Subset(
        val_ds,
        val_indices
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    collate = (
        gff.dataloaders.sometimes_things_are_lists
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        generator=generator
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate
    )

    return train_loader, val_loader

# TRAINING

def main():

    print("GFF + SEN12 FUSION FINE-TUNING")

    assert CONFIG_FILE.is_file(), CONFIG_FILE
    assert SUBSET_FILE.is_file(), SUBSET_FILE

    with CONFIG_FILE.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["epochs"] = EPOCHS
    config["batch_size"] = BATCH_SIZE
    config["num_workers"] = 0

    config["cache_local_in_ram"] = False
    config["cache_context_in_ram"] = False

    config["scalar_freq"] = 32
    config["img_freq"] = 1000000

    config["name"] = "mini_project_sen12_fusion"

    gff.util.seed_packages(SEED)

    assert torch.cuda.is_available(), (
        "CUDA GPU required for this run"
    )

    device = torch.device(config["device"])

    print("GPU:", torch.cuda.get_device_name(0))
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)

    train_loader, val_loader = create_dataloaders(
        config
    )

    print("\nDATASET")
    print("Training tiles:", len(train_loader.dataset))
    print("Validation tiles:", len(val_loader.dataset))

    gff_model, sen12_model = load_models(
        config,
        device
    )

    model = TrainableFusion(
        gff_model,
        sen12_model
    ).to(device)

    first_conv = configure_trainable_parameters(
        model
    )

    #Confirm the pretrained five input weights and sixth fusion channel have expected shapes
    assert first_conv.weight.shape[1] == 6

    with torch.no_grad():
        assert torch.count_nonzero(
            first_conv.weight[:, 5:]
        ).item() == 0

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_folder = RUNS / (
        f"mini_project_sen12_fusion_{timestamp}"
    )

    run_folder.mkdir(
        parents=True,
        exist_ok=False
    )

    with (
        run_folder / "config.yml"
    ).open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    with (
        run_folder / "experiment.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "SEN12 feature fusion",
                "baseline_run": str(BASELINE_RUN),
                "epochs": EPOCHS,
                "seed": SEED,
                "train_tiles": 1024,
                "validation_tiles": 256,
                "sen12_encoder": "frozen",
                "gff_backbone": "frozen_except_input_conv",
                "trainable_parts": [
                    "fusion_adapter",
                    "utae_first_convolution"
                ],
                "note": (
                    "Original GFF and SEN12 checkpoints "
                    "are not overwritten."
                )
            },
            f,
            indent=2
        )

    print("\nOUTPUT FOLDER")
    print(run_folder)

    print("\nStarting official GFF training loop...")

    gff.training.training_loop(
        config,
        run_folder,
        model,
        (train_loader, val_loader)
    )

    checkpoints = sorted(
        run_folder.glob("checkpoint_*.th")
    )

    print("FUSION FINE-TUNING COMPLETED")

    print("Saved checkpoints:")

    for path in checkpoints:
        print(" ", path.name)

    print("Output:", run_folder)


if __name__ == "__main__":
    main()
