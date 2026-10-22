
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import gff.dataloaders
import gff.models.creation
import gff.util


ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))

from train_sen12_classifier import RadarCNN

CONFIG = ROOT / "configs" / "config_utae.yml"

GFF_CHECKPOINT = ROOT / "models" / "gff_utae_best.th"
SEN12_CHECKPOINT = ROOT / "models" / "sen12_classifier_best.pt"

#SEN12-COMPATIBLE RADAR PREPROCESSING

def preprocess_radar(radar):
    """
    radar: Raw GFF Sentinel-1 tensor [B, 2, H, W].

    Reproduces SEN12's signed-log and per-image
    percentile scaling, at 160x160 resolution.

    Does not use labels or future observations.
    """

    radar = F.interpolate(
        radar.float(),
        size=(160, 160),
        mode="bilinear",
        align_corners=False
    )

    result = torch.zeros_like(radar)

    for b in range(radar.shape[0]):
        for c in range(2):
            values = radar[b, c]

            valid = torch.isfinite(values)

            if int(valid.sum()) < 100:
                continue

            clean = torch.where(
                valid,
                values,
                torch.zeros_like(values)
            )

            transformed = (
                torch.sign(clean)
                * torch.log1p(clean.abs())
            )

            valid_values = transformed[valid]

            lower = torch.quantile(
                valid_values, 0.01
            )
            upper = torch.quantile(
                valid_values, 0.99
            )

            if (
                not torch.isfinite(lower)
                or not torch.isfinite(upper)
                or (upper - lower).item() <= 1e-8
            ):
                continue

            scaled = (
                (transformed - lower)
                / (upper - lower)
            ).clamp(0, 1)

            result[b, c] = torch.where(
                valid,
                scaled * 2 - 1,
                torch.zeros_like(scaled)
            )

    return result

#COMBINED MODEL

class FusedFloodModel(nn.Module):

    def __init__(self, gff_model, radar_model):
        super().__init__()

        self.gff = gff_model

        #Keep pretrained SEN12 feature extractor
        self.sen12_encoder = radar_model.encoder[:-1]

        for p in self.sen12_encoder.parameters():
            p.requires_grad = False

        self.sen12_encoder.eval()

        #Learn a 128-channel -> 1-channel projection
        self.adapter = nn.Conv2d(
            128, 1, kernel_size=1
        )

        #Locate the official first convolution
        old_conv = (
            self.gff.local_embed
            .in_conv.conv.conv[0]
        )

        assert old_conv.in_channels == 5

        #Expand 5 input channels to 6
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None
        )

        with torch.no_grad():
            new_conv.weight.zero_()

            #Preserve the original five input weights
            new_conv.weight[:, :5].copy_(
                old_conv.weight
            )

            #The SEN12 input channel initially has no effect on the GFF predictions
            if old_conv.bias is not None:
                new_conv.bias.copy_(
                    old_conv.bias
                )

        self.gff.local_embed.in_conv.conv.conv[0] = (
            new_conv
        )

        self.extra_feature = None

        #Inject the SEN12 feature after the official GFF model has normalized its local inputs
        self.hook = (
            self.gff.local_embed
            .register_forward_pre_hook(
                self.add_feature
            )
        )

    def train(self, mode=True):
        super().train(mode)

        #Keep SEN12 batch-normalization statistics fixed during GFF fine-tuning
        self.sen12_encoder.eval()

        return self

    def add_feature(self, module, args):
        local_input = args[0]

        if self.extra_feature is None:
            raise RuntimeError(
                "SEN12 features were not prepared"
            )

        # local_input: [B, 1, 5, H, W]
        assert local_input.ndim == 5
        assert local_input.shape[2] == 5

        feature = self.extra_feature.unsqueeze(1)

        assert feature.shape[:2] == (
            local_input.shape[0], 1
        )

        assert feature.shape[-2:] == (
            local_input.shape[-2:]
        )

        combined = torch.cat(
            [local_input, feature],
            dim=2
        )

        assert combined.shape[2] == 6

        return (combined,) + args[1:]

    def forward(self, example):

        #Raw GFF Sentinel-1 inputs, before the official model's own normalization
        raw_radar = example["s1"]

        with torch.no_grad():
            radar = preprocess_radar(
                raw_radar
            )

            features = self.sen12_encoder(
                radar
            )

        projected = self.adapter(features)

        self.extra_feature = F.interpolate(
            projected,
            size=raw_radar.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        try:
            output = self.gff(example)
        finally:
            self.extra_feature = None

        return output

#CHECKPOINT LOADING

def load_models(config, device):

    gff_model = (
        gff.models.creation.create(config)
        .to(device)
    )

    gff_checkpoint = torch.load(
        GFF_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    gff_model.load_state_dict(
        gff_checkpoint["model"]
    )

    sen12_model = RadarCNN()

    sen12_checkpoint = torch.load(
        SEN12_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    sen12_model.load_state_dict(
        sen12_checkpoint["model_state"]
    )

    sen12_model.to(device)

    print(
        "Loaded GFF checkpoint epoch:",
        gff_checkpoint["epoch"]
    )
    print(
        "Loaded SEN12 checkpoint epoch:",
        sen12_checkpoint["epoch"]
    )

    return gff_model, sen12_model


#TEST WITH REAL GFF DATA

def main():

    torch.manual_seed(42)

    with CONFIG.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("GFF + SEN12 FUSED FORECAST TEST")

    assert GFF_CHECKPOINT.is_file()
    assert SEN12_CHECKPOINT.is_file()

    print("Device:", device)

    train_files, _, _ = (
        gff.dataloaders.read_partitions(
            Path(config["data_folder"]),
            config["fold"]
        )
    )

    dataset = gff.dataloaders.FloodForecastDataset(
        Path(config["data_folder"]),
        config,
        meta_fnames=train_files[:1]
    )

    example = dataset[0]

    batch = gff.dataloaders.sometimes_things_are_lists(
        [example]
    )

    batch.pop("floodmap", None)

    batch = gff.util.recursive_todevice(
        batch, device
    )

    #Baseline and fused model must start from identical pretrained GFF parameters
    baseline, _ = load_models(config, device)
    baseline.eval()

    with torch.inference_mode():
        baseline_output = baseline(batch)

    gff_model, sen12_model = load_models(
        config, device
    )

    fused = FusedFloodModel(
        gff_model,
        sen12_model
    ).to(device)

    fused.eval()

    with torch.inference_mode():
        fused_output = fused(batch)

    print("\nBASELINE OUTPUT")
    print("Shape:", tuple(baseline_output.shape))

    print("\nFUSED OUTPUT")
    print("Shape:", tuple(fused_output.shape))

    max_difference = (
        baseline_output - fused_output
    ).abs().max().item()

    print(
        "\nMaximum output difference:",
        max_difference
    )

    print(
        "Finite:",
        bool(torch.isfinite(fused_output).all())
    )

    assert fused_output.shape == (
        baseline_output.shape
    )

    assert torch.isfinite(fused_output).all()

    #With the sixth-channel weights initialized to zero, both models should be equivalent
    assert torch.allclose(
        baseline_output,
        fused_output,
        atol=1e-5,
        rtol=1e-5
    ), "Fusion unexpectedly changed baseline predictions"

    print("\nFUSED FORECAST TEST PASSED")
    print(
        "Original GFF predictions preserved."
    )
    print(
        "SEN12 spatial features successfully injected."
    )


if __name__ == "__main__":
    main()
