
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

#MINI-PROJECT CONFIG

ROOT = Path(__file__).resolve().parent

CONFIG_PATH = ROOT / "configs" / "mini_project_utae.yml"

TRAIN_TILES = 1024
VAL_TILES = 256
EPOCHS = 3

SEED = 6426452

RUNS = ROOT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

# SELECT TILES FROM MULTIPLE OBSERVATIONS

def select_tiles(dataset, maximum, seed):
    """
    Select a reproducible, approximately balanced
    sample across observations.

    Selection uses tile indices and observation
    identity, never flood labels.
    """

    groups = {}

    for index, item in enumerate(dataset.tiles):
        meta = item[0]
        observation = meta["floodmap"]

        groups.setdefault(
            observation, []
        ).append(index)

    rng = np.random.default_rng(seed)

    for indices in groups.values():
        rng.shuffle(indices)

    observations = sorted(groups)

    selected = []
    positions = {
        name: 0 for name in observations
    }

    while len(selected) < maximum:

        progressed = False

        for name in observations:
            indices = groups[name]
            position = positions[name]

            if position >= len(indices):
                continue

            selected.append(indices[position])
            positions[name] += 1

            progressed = True

            if len(selected) >= maximum:
                break

        if not progressed:
            break

    return sorted(selected)


def main():

    with CONFIG_PATH.open(
        encoding="utf-8"
    ) as f:
        C = yaml.safe_load(f)

    C["epochs"] = EPOCHS
    C["batch_size"] = 1
    C["num_workers"] = 0

    #Avoid caching every raster in RAM
    C["cache_local_in_ram"] = False
    C["cache_context_in_ram"] = False

    C["scalar_freq"] = 32

    #Avoid expensive image logging during the first short training experiment
    C["img_freq"] = 1000000

    C["name"] = "mini_project_utae_initial"

    gff.util.seed_packages(SEED)

    assert torch.cuda.is_available(), (
        "CUDA is required for this run."
    )

    device = torch.device(C["device"])

    print("OFFICIAL GFF U-TAE — TRAINING")

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print("Architecture:", C["backbone"])
    print("Input sources:", C["data_sources"])
    print("Epochs:", C["epochs"])
    print("Batch size:", C["batch_size"])

    folder = Path(
        C["data_folder"]
    ).expanduser()

    train_files, val_files, _ = (
        gff.dataloaders.read_partitions(
            folder,
            C["fold"]
        )
    )

    #Deliberately do not construct a test dataset

    train_ds = gff.dataloaders.FloodForecastDataset(
        folder,
        C,
        meta_fnames=train_files,
        label_sources=C["train_label_sources"]
    )

    val_ds = gff.dataloaders.FloodForecastDataset(
        folder,
        C,
        meta_fnames=val_files
    )

    print("\nFULL PARTITION SIZES")

    print("Training tiles:", len(train_ds))
    print("Validation tiles:", len(val_ds))

    train_indices = select_tiles(
        train_ds,
        TRAIN_TILES,
        SEED
    )

    val_indices = select_tiles(
        val_ds,
        VAL_TILES,
        SEED + 1
    )

    train_subset = Subset(
        train_ds,
        train_indices
    )

    val_subset = Subset(
        val_ds,
        val_indices
    )

    print("\nSELECTED SUBSET SIZES")

    print("Training tiles:", len(train_subset))
    print("Validation tiles:", len(val_subset))

    if not train_subset or not val_subset:
        raise RuntimeError(
            "Empty training or validation subset"
        )

    collate = (
        gff.dataloaders.sometimes_things_are_lists
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_dl = DataLoader(
        train_subset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        generator=generator
    )

    val_dl = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate
    )

    #CREATE OFFICIAL MODEL

    model = (
        gff.models.creation.create(C)
        .to(device)
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nMODEL")

    print("Parameters:", f"{parameters:,}")

    # SAVE CONFIG + SUBSET INDICES

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_folder = RUNS / (
        f"mini_project_utae_{timestamp}"
    )

    run_folder.mkdir(
        parents=True,
        exist_ok=False
    )

    with (
        run_folder / "config.yml"
    ).open("w", encoding="utf-8") as f:
        yaml.safe_dump(C, f)

    with (
        run_folder / "subset_indices.json"
    ).open("w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "train_indices": train_indices,
            "val_indices": val_indices,
        }, f)

    print("\nRUN FOLDER:", run_folder)

    print("\nSTARTING OFFICIAL TRAINING LOOP")

    #This calls the AUTHORS' training loop
    
    #Unlike train.py, this script does not perform automatic test-set inference

    gff.training.training_loop(
        C,
        run_folder,
        model,
        (train_dl, val_dl)
    )

    print("OFFICIAL U-TAE TRAINING COMPLETED")

    checkpoints = sorted(
        run_folder.glob("checkpoint_*.th")
    )

    print("Checkpoints:", len(checkpoints))

    for checkpoint in checkpoints:
        print(checkpoint.name)

    print("Results folder:", run_folder)

    print("\nNEXT: VALIDATION AND FLOOD MAP VISUALIZATION")


if __name__ == "__main__":
    main()
