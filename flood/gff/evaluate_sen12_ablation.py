
from pathlib import Path
import json

import torch
import yaml

from torch.utils.data import DataLoader, Subset

import gff.dataloaders
import gff.training

from test_fused_flood_model import (
    FusedFloodModel,
    load_models,
)


ROOT = Path(__file__).resolve().parent

BASELINE_RUN = (
    ROOT / "gff_official" / "runs"
    / "mini_project_utae_20261022_170820"
)

FUSION_RUN = (
    ROOT / "gff_official" / "runs"
    / "mini_project_sen12_fusion_20261022_220212"
)

CHECKPOINT = ROOT / "models" / "gff_sen12_fusion_best.th"
CONFIG_FILE = ROOT / "configs" / "config_sen12.yml"

SUBSET_FILE = ROOT / "data" / "subset_indices.json"

OUTPUT_FILE = ROOT / "reports" / "sen12_ablation_results.json"


def make_validation_loader(config):

    folder = Path(config["data_folder"])

    _, val_files, _ = (
        gff.dataloaders.read_partitions(
            folder,
            config["fold"]
        )
    )

    dataset = gff.dataloaders.FloodForecastDataset(
        folder,
        config,
        meta_fnames=val_files
    )

    with SUBSET_FILE.open(
        encoding="utf-8"
    ) as f:
        indices = json.load(f)["val_indices"]

    assert len(indices) == 256

    assert max(indices) < len(dataset)

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=(
            gff.dataloaders
            .sometimes_things_are_lists
        )
    )

    return loader


def evaluate(model, loader, criterion, classes):

    model.eval()

    loss, f1s, confusion = (
        gff.training.val_epoch(
            model,
            loader,
            criterion,
            classes,
            limit=256
        )
    )

    f1s = [
        float(value)
        for value in f1s
    ]

    return {
        "loss": float(loss),
        "background_f1": f1s[0],
        "flood_water_f1": f1s[1],
        "overall_f1": sum(f1s) / len(f1s),
        "confusion_matrix": (
            confusion.tolist()
            if hasattr(confusion, "tolist")
            else str(confusion)
        )
    }


def show_results(name, result):

    print("\n" + name)

    print(
        "Validation loss:",
        round(result["loss"], 6)
    )

    print(
        "Background F1:",
        round(result["background_f1"], 6)
    )

    print(
        "Flood/water F1:",
        round(result["flood_water_f1"], 6)
    )

    print(
        "Overall F1:",
        round(result["overall_f1"], 6)
    )


def main():

    print("SEN12 FEATURE ABLATION")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    assert CHECKPOINT.is_file(), CHECKPOINT
    assert SUBSET_FILE.is_file(), SUBSET_FILE

    with CONFIG_FILE.open(
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    loader = make_validation_loader(
        config
    )

    print(
        "Validation tiles:",
        len(loader.dataset)
    )

    gff_model, radar_model = load_models(
        config, device
    )

    model = FusedFloodModel(
        gff_model,
        radar_model
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    weights = torch.tensor(
        config.get(
            "class_weights",
            (1.0,) * 3
        ),
        dtype=torch.float32
    )[:config["n_classes"]]

    criterion = (
        gff.training.DeterministicCrossEntropyLoss(
            weight=weights,
            ignore_index=-100
        )
    )

    #FULL FUSION MODEL
    print(
        "\nEvaluating full GFF + SEN12 model...",
        flush=True
    )

    with torch.inference_mode():
        full_results = evaluate(
            model,
            loader,
            criterion,
            config["n_classes"]
        )

    show_results(
        "GFF + SEN12 ENABLED",
        full_results
    )

    #DISABLE SEN12 FEATURE CONTRIBUTION

    first_conv = (
        model.gff.local_embed
        .in_conv.conv.conv[0]
    )

    assert first_conv.in_channels == 6

    with torch.no_grad():

        original_sen12_weights = (
            first_conv.weight[:, 5:].clone()
        )

        print(
            "\nSEN12 channel weight norm:",
            original_sen12_weights.norm().item()
        )

        first_conv.weight[:, 5:].zero_()

    try:

        print(
            "\nEvaluating with SEN12 channel disabled...",
            flush=True
        )

        with torch.inference_mode():
            ablated_results = evaluate(
                model,
                loader,
                criterion,
                config["n_classes"]
            )

    finally:

        with torch.no_grad():
            first_conv.weight[:, 5:].copy_(
                original_sen12_weights
            )

    show_results(
        "SAME MODEL - SEN12 DISABLED",
        ablated_results
    )

    #COMPARE

    print("SEN12 CONTRIBUTION ON VALIDATION SET")

    for metric in (
        "overall_f1",
        "flood_water_f1",
        "background_f1"
    ):

        enabled = full_results[metric]
        disabled = ablated_results[metric]

        difference = (
            enabled - disabled
        ) * 100

        print(
            f"{metric}: "
            f"{difference:+.4f} percentage points"
        )

    results = {
        "checkpoint": str(CHECKPOINT),
        "validation_tiles": 256,
        "sen12_enabled": full_results,
        "sen12_disabled": ablated_results
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            results,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
