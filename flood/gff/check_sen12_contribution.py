
from pathlib import Path

import torch
import yaml

import gff.dataloaders
import gff.util

from test_fused_flood_model import (
    FusedFloodModel,
    load_models,
)


ROOT = Path(__file__).resolve().parent

OFFICIAL = ROOT / "gff_official"

FUSION_RUN = (
    OFFICIAL / "runs"
    / "mini_project_sen12_fusion_20261022_220212"
)

CHECKPOINT = ROOT / "models" / "gff_sen12_fusion_best.th"
CONFIG = ROOT / "configs" / "config_sen12.yml"


def main():
    print("SEN12 FEATURE CONTRIBUTION CHECK")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    with CONFIG.open(encoding="utf-8") as f:
        C = yaml.safe_load(f)

    gff_model, radar_model = load_models(C, device)

    model = FusedFloodModel(
        gff_model,
        radar_model
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    conv = (
        model.gff.local_embed
        .in_conv.conv.conv[0]
    )

    original_weights = (
        conv.weight[:, :5]
    )

    sen12_weights = (
        conv.weight[:, 5:]
    )

    print("\nFIRST CONVOLUTION")

    print(
        "Original channel weight norm:",
        original_weights.norm().item()
    )

    print(
        "SEN12 channel weight norm:",
        sen12_weights.norm().item()
    )

    print(
        "SEN12 channel max absolute weight:",
        sen12_weights.abs().max().item()
    )

    train_files, _, _ = (
        gff.dataloaders.read_partitions(
            Path(C["data_folder"]),
            C["fold"]
        )
    )

    dataset = gff.dataloaders.FloodForecastDataset(
        Path(C["data_folder"]),
        C,
        meta_fnames=train_files[:1],
        label_sources=C["train_label_sources"]
    )

    example = dataset[0]

    batch = (
        gff.dataloaders.sometimes_things_are_lists(
            [example]
        )
    )

    batch.pop("floodmap", None)

    batch = gff.util.recursive_todevice(
        batch,
        device
    )

    with torch.inference_mode():
        full_output = model(batch).clone()

        saved_sen12_weights = (
            conv.weight[:, 5:].clone()
        )

        #Temporarily remove the contribution of the SEN12 feature channel
        conv.weight[:, 5:].zero_()

        without_sen12 = model(batch).clone()

        #Restore the original trained weights
        conv.weight[:, 5:].copy_(
            saved_sen12_weights
        )

    difference = (
        full_output - without_sen12
    ).abs()

    print("\nOUTPUT ABLATION")

    print(
        "Maximum prediction-logit difference:",
        difference.max().item()
    )

    print(
        "Mean prediction-logit difference:",
        difference.mean().item()
    )

    full_prediction = full_output.argmax(dim=1)
    ablated_prediction = without_sen12.argmax(dim=1)

    changed_pixels = (
        full_prediction != ablated_prediction
    ).sum().item()

    total_pixels = full_prediction.numel()

    print(
        "Changed predicted pixels:",
        changed_pixels
    )

    print(
        "Percentage of changed pixels:",
        round(100 * changed_pixels / total_pixels, 4)
    )

    print("\nSEN12 CONTRIBUTION CHECK COMPLETED")


if __name__ == "__main__":
    main()
