# Disaster Progression Predictor

A deep-learning mini-project for predicting how **floods** and **wildfires** evolve from satellite observations. The two disaster types have separate datasets, models, and evaluation protocols; they can eventually share a user interface without sharing incompatible labels.

**Current status (October 2026):** The flood branch has a trained GFF U-TAE baseline, a trained SEN12-FLOOD radar classifier, and a fine-tuned GFF + SEN12 feature-fusion model. A fixed-weight feature ablation has been evaluated on the GFF development-validation subset. **Held-out test evaluation, qualitative flood-map examples, a matched-budget training control, and the wildfire branch remain outstanding.**

## Project at a glance

| Branch | Dataset | Task | Status |
|---|---|---|---|
| Flood forecast baseline | Global Flood Forecasting (GFF) | Predict a future two-class flood/water map from pre-event radar, terrain, and lead time | Trained and validated |
| Radar feature pretraining | SEN12-FLOOD | Classify whether a Sentinel-1 acquisition is associated with flooding | Trained and validated |
| Flood forecast with transfer learning | GFF + pretrained SEN12-FLOOD encoder | Predict a future flood/water map using auxiliary radar features | Trained, validated, and ablated |
| Wildfire forecast | TS-SatFire | Predict future fire/burn extent from earlier observations | Assigned to wildfire workstream; implementation pending |

## Flood models and results

### 1. GFF U-TAE baseline

The baseline adapts the **official GFF U-TAE implementation** using the local modalities available in this project:

- Pre-event Sentinel-1 **VV and VH** radar channels.
- Local digital elevation model (**DEM**) and height above nearest drainage (**HAND**).
- A land/sea indicator derived from DEM by the official model.
- Forecast lead-time conditioning.

The GFF implementation assembles five spatial input channels and produces a **224 × 224 two-class prediction**. The reduced-input experiment does **not** use the paper's full environmental feature set, particularly ERA5/ERA5-Land weather inputs; its results should not be presented as a reproduction of the full published benchmark.

The original experiment ran for three epochs on a reproducibly sampled **1,024 training tiles** and **256 validation tiles**. The best checkpoint was from epoch 2 (stored with the zero-based checkpoint suffix `001`).

### 2. SEN12-FLOOD radar classifier

A lightweight RadarCNN learns from paired Sentinel-1 VV/VH acquisitions. A verified manifest contains **3,062 VV/VH pairs** across 234 location folders. The split is location-folder-disjoint: **2,134 train**, **486 validation**, and **442 test** acquisitions. The last set remains held out.

The classifier has approximately **97,905 parameters**, uses per-band signed-log compression and per-image 1st/99th-percentile scaling, and was trained for five epochs. Its best checkpoint is from epoch 2.

**Important distinction:** SEN12's `FLOODING` label is **acquisition-level**. The classifier is **not** itself a pixel-wise flood forecasting model. Its encoder supplies auxiliary radar features to GFF.

### 3. GFF + SEN12 feature fusion

The combined approach loads both trained models, takes spatial features from the SEN12 encoder **before global pooling**, and uses a learnable `128 → 1` convolutional adapter. The adapted map is upsampled and appended as a sixth local input channel to GFF U-TAE.

The first five channels of the expanded U-TAE convolution are initialized from the trained GFF baseline; the new sixth-channel weights are initialized to zero. Consequently, before fusion fine-tuning, the combined model's output exactly matched the GFF baseline on the tested tile.

During the three-epoch fusion experiment, the SEN12 encoder and most pretrained GFF weights were frozen. **513 of 1,399,171 parameters** were trainable: the adapter and the expanded first convolution's weights. This first convolution also allowed the original five channels' weights to change.

### Development-validation results

The following results are from the **same sampled 256 GFF validation tiles**, not from the untouched held-out test partition. GFF scores are pixel-level class-wise F1 as computed by the official validation pipeline. The separate SEN12 score below is acquisition-level F1 and is **not directly comparable**.

| GFF model | Overall F1 (macro across two classes) | Flood/water F1 | Background F1 | Validation loss |
|---|---:|---:|---:|---:|
| U-TAE baseline, best epoch | **0.7902** | **0.6620** | 0.9185 | 0.21329 |
| Fine-tuned fusion model, SEN12 channel disabled | **0.8072** | **0.6888** | 0.9256 | 0.18569 |
| Fine-tuned fusion model, SEN12 channel enabled | **0.9066** | **0.8417** | 0.9716 | 0.14797 |

**SEN12 classifier:** Best validation acquisition-level F1 **0.6596**, accuracy **0.6708**, precision **0.5657**, recall **0.7908** (486 validation acquisitions).

The within-checkpoint **SEN12-channel ablation** decreased GFF overall F1 by **9.94 percentage points** and flood/water F1 by **15.29 percentage points**. This shows that the trained fusion model uses the feature channel and that disabling it hurts this model on the sampled validation set. It does **not**, by itself, establish that SEN12 pretraining is better than a GFF-only model given an equivalent extra training budget: post-training removal of an input also changes the model's input distribution.

**Experiment limitations:** The initial GFF-only model was trained for three epochs, while the fusion model was initialized from its best checkpoint and trained for three **additional** epochs. A matched-budget GFF-only fine-tuning control is therefore required for a stronger comparison. None of the GFF figures above are final held-out test scores or published-paper benchmark scores. Generalization to unseen events, spatial overlap beyond the existing partition design, calibration, and performance across forecast horizons still require assessment.

## Repository organization

The intended layout for the cleaned GitHub flood branch is:

```text
Disaster-Progression-Predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── flood/
│   ├── configs/
│   │   ├── mini_project_utae.yml
│   │   ├── config_utae.yml
│   │   └── config_sen12.yml
│   ├── data/
│   │   ├── sen12_manifest.csv
│   │   ├── sen12_splits.csv
│   │   ├── subset_indices.json
│   │   ├── GFF_official_view/       # external dataset, not committed
│   │   └── SEN12FLOOD/             # external dataset, not committed
│   ├── models/
│   │   ├── gff_utae_best.th
│   │   ├── gff_sen12_fusion_best.th
│   │   └── sen12_classifier_best.pt
│   ├── reports/
│   │   ├── experiment.json
│   │   ├── sen12_ablation_results.json
│   │   └── sen12_classifier_history.json
│   ├── gff/                       # official GFF source package
│   ├── sen12_build_manifest.py
│   ├── sen12_create_splits.py
│   ├── train_sen12_classifier.py
│   ├── train_mini_project.py
│   ├── test_fused_flood_model.py
│   ├── train_fused_flood_model.py
│   ├── check_sen12_contribution.py
│   └── evaluate_sen12_ablation.py
└── wildfire/
```

This is the **target layout**, not confirmation that all moved scripts have been tested in place. Source code may still need imports and config references updated after relocation. Preserve the official GFF source structure and its license/attribution; do not silently present it as original team code.

The two GFF checkpoint files in the original run directories were named `checkpoint_001.th` (baseline) and `checkpoint_002.th` (fusion). When moving them into `flood/models/`, use the clearer filenames shown above and update all loading paths accordingly. Original training configs, manifests, and subset indices should be retained to document provenance.

## Installation and data setup

### Python environment

Python, Git, and the packages in `requirements.txt` are required. The flood experiments were developed with Python 3.13 and CUDA-enabled PyTorch on an NVIDIA RTX 3050 Laptop GPU; equivalent hardware is **not** guaranteed, and CUDA/PyTorch compatibility is machine-dependent.

```bash
git clone <repository-url>
cd Disaster-Progression-Predictor
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment if desired (`.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on Linux/macOS), or run scripts via the virtual environment's Python executable. Install a suitable CPU or CUDA PyTorch build for the target machine. The original official GFF package may have additional dependencies; validate the environment with the retained `gff/` source rather than assuming installation is sufficient.

### External datasets

**Raw satellite imagery is intentionally not stored in Git.** Cloning the repository is sufficient to obtain the source, manifests, and any committed model checkpoints, but **not** to retrain or run an inference example that requires external rasters.

- **GFF:** Obtain the GFF dataset and prepare the layout expected by the official loader: `flood/data/GFF_official_view/rois/`, `partitions/`, and `normalisation/`. The local training dataset used an official-loader-compatible view of 298 GFF observations; this view must be recreated for a new machine.
- **SEN12-FLOOD:** Obtain the raster dataset and place location folders under `flood/data/SEN12FLOOD/`. Build or reuse the manifest and folder-disjoint split files, checking that all CSV-relative paths resolve correctly.
- **TS-SatFire:** The wildfire teammate should document their independently acquired dataset and target folder in `wildfire/README.md`.

The checked-in paths should be resolved relative to the script or the repository directory, **not** to a developer-specific `C:\...` location. GFF YAML `data_folder` settings must be resolved consistently with Python path handling. Do not assume an arbitrary working directory. The prepared data and original runs are not redistributed as part of this README.

### Flood commands

From the repository root (after setting up the required data and Python environment):

```bash
# Build the SEN12 manifest and location-folder splits
python flood/sen12_build_manifest.py
python flood/sen12_create_splits.py

# Train from scratch only when output checkpoints do not already exist
python flood/train_sen12_classifier.py
python flood/train_mini_project.py

# Check integration or evaluate an already trained fused model
python flood/test_fused_flood_model.py
python flood/check_sen12_contribution.py
python flood/evaluate_sen12_ablation.py
```

`train_fused_flood_model.py` performs additional GFF fine-tuning after loading the original GFF and SEN12 checkpoints. It must be run only after checking its output directory, checkpoint references, and GFF configuration. The original SEN12 training script deliberately prevents overwriting an existing model checkpoint. These commands describe the intended workflow; **end-to-end execution from a freshly cloned, reorganized repository has not yet been verified**.

### Checkpoints and portability

Pretrained weights can be versioned separately when they are too large for normal Git hosting. If a checkpoint is distributed externally, document the source URL, expected filename, checksum, and destination rather than claiming it is provided in the clone. A checkpoint alone cannot reproduce the dataset or its preprocessing.

## Wildfire workstream

The wildfire branch is a **separate implementation**, using **TS-SatFire** rather than GFF or SEN12-FLOOD. Its output should ultimately be consistent with the flood branch at the application-interface level, not by forcing both disasters into the same network architecture.

### Dataset and target definition

1. Obtain TS-SatFire from its official source: https://github.com/zhaoyutim/TS-SatFire. Keep the raw imagery outside Git.
2. Inventory the available events, dates, bands, coordinate systems, raster sizes, NoData masks, and class values. The initial local selection contained approximately **30 event folders** and imagery including **FirePred**, **VIIRS_Day**, **VIIRS_Night**, and **ESRI_LULC**; treat these as a starting inventory, not a validated ready-made sequence dataset.
3. Read the source documentation and inspect actual labels to establish whether `FirePred` means active fire, newly burned area, cumulative burn extent, or something else. **Do not assume the target semantics.** Record the unit of prediction and the intended lead time.
4. Construct sequences using only timestamps **before** each target date. Ensure that VIIRS and land-cover observations are available by the prediction cutoff. Handle irregular observation gaps explicitly.
5. Split by **whole fire event or geographic area** before creating overlapping windows; avoid placing nearby frames of the same fire into both training and validation/test sets.

### Suggested implementation order

1. Implement a minimal dataset loader and inspect real input/target image pairs.
2. Train a straightforward baseline (persistence, where appropriate, or a compact CNN/U-Net). Report class imbalance and valid-pixel coverage.
3. Add a temporal model (ConvLSTM or another justified sequence model) only once the baseline and temporal labels are sound. GNN work is **optional research extension**, not a prerequisite for delivery.
4. Choose the best model on a fixed validation split; run the test set once for final reporting.
5. Produce at least one side-by-side visualization of historical inputs, ground truth and forecast, including the date and forecast horizon.
6. Record model configuration, random seed, dataset/split manifests, training history, metrics, and limitations.

### Expected files from the wildfire teammate

```text
wildfire/
├── README.md                 # data source, target semantics, setup
├── configs/                 # reproducible experiment parameters
├── data/                    # small manifests/split files only
├── models/                  # model code; checkpoints if distributable
├── reports/                 # validation/test metrics and examples
├── prepare_data.py
├── train.py
├── evaluate.py
└── predict.py
```

The exact Python filenames are flexible; the **interfaces and documentation** matter. Make sure the entry points accept configurable dataset/model paths and do not reference personal drives.

### Integration contract

Once both branches are ready, each should expose an inference path accepting an observation or prepared tensor and forecast horizon, and returning:

- A **spatial prediction map** (with class/probability semantics documented).
- Geospatial metadata and target horizon where supported.
- Model identifier and preprocessing version.
- A clear indication of whether the model is a flood or wildfire forecaster.

The application can offer a disaster-type selector, but should **not** combine flood F1 and wildfire F1 into a single model-quality number. Keep model-specific evaluation methods, test sets, and claims separate.

### Git collaboration

Develop wildfire changes in a branch, e.g. `feature/wildfire-pipeline`, and open a pull request. Avoid changing the flood branch without coordination. Add installation notes, a minimal smoke-test command, and sample output to `wildfire/README.md`. Do not commit downloaded imagery, caches, virtual environments, sensitive machine paths, or massive intermediary files.

## Evaluation protocol and remaining work

- **GFF:** Report per-class F1, macro-F1, official validation loss, and ideally flood IoU, per-event results, and representative predicted/actual maps. Respect the official ignored-pixel handling.
- **SEN12 classifier:** Report acquisition-level accuracy, precision, recall, F1, and class balance separately from GFF segmentation/forecast scores.
- **Wildfire:** Select metrics after verifying label semantics; likely spatial IoU/F1 and horizon-specific evaluation, alongside persistence comparison where meaningful.
- **Comparability:** Compare models on the same split and tiles, with matching label masks, forecast horizons, and training budgets.
- **Outstanding flood experiments:** Matched-extra-epochs GFF-only control, true validation contribution check on several locations beyond the current subset, output visualizations, untouched held-out test evaluation, and a clean-clone reproducibility test.
- **Potential leakage:** Folder-disjoint SEN12 acquisitions do not automatically prove different spatial events; check event/region overlap explicitly. GFF's original published setup may use retrospective weather reanalysis, but the reduced-input mini-project configuration excludes weather channels, limiting this particular risk. Forecast-time assumptions still need documentation.
