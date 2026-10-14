# Disaster Progression Predictor

**A deep-learning framework for forecasting the spatiotemporal progression of floods and wildfires using remote-sensing data.**

> **Project status:** Dataset inventory and initial raster validation completed. Dataset alignment, flood-label verification, and temporal sequence preparation are in progress. Model training has not yet been documented as completed.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [Proposed Model Architecture](#3-proposed-model-architecture)
4. [Datasets](#4-datasets)
5. [Current Project Progress](#5-current-project-progress)
6. [Repository Structure](#6-repository-structure)
7. [Technology Stack](#7-technology-stack)
8. [Environment Setup](#8-environment-setup)
9. [Dataset Organization](#9-dataset-organization)
10. [Data Processing Pipeline](#10-data-processing-pipeline)
11. [Experimental Design](#11-experimental-design)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Next Plan of Action](#13-next-plan-of-action)
14. [Team Collaboration](#14-team-collaboration)
15. [Reproducibility and Data Leakage Prevention](#15-reproducibility-and-data-leakage-prevention)
16. [Data Storage and Version Control](#16-data-storage-and-version-control)
17. [Known Issues and Limitations](#17-known-issues-and-limitations)
18. [References and Dataset Documentation](#18-references-and-dataset-documentation)

---

## 1. Project Overview

Natural disasters such as floods and wildfires evolve over time and affect spatially connected regions. Predicting their future extent can support disaster monitoring, emergency planning, resource allocation, and risk assessment.

The **Disaster Progression Predictor** aims to develop a spatiotemporal deep-learning framework that learns from historical observations and predicts future disaster-related spatial patterns.

The project focuses on two separate prediction pipelines:

- **Flood progression prediction:** Learning from flood-related observations, satellite imagery, terrain, weather, and hydrological information.
- **Wildfire progression prediction:** Learning from chronological fire observations and associated environmental or land-cover information.

The intended system combines convolutional neural networks, recurrent spatial-temporal modeling, and graph neural networks.

The immediate priority is to validate the datasets, establish reliable target labels, construct temporally aligned training sequences, and establish baseline models before implementing the complete architecture.

### Prediction objective

Given observations from previous time steps,

\[
X_{t-k+1}, X_{t-k+2}, \ldots, X_t
\]

the model aims to predict a future spatial target:

\[
\hat{Y}_{t+h}=f(X_{t-k+1:t})
\]

where:

- \(X\) represents the available input modalities.
- \(k\) is the number of historical observations.
- \(h\) is the forecast horizon.
- \(\hat{Y}_{t+h}\) is the predicted future disaster-related spatial map.

The precise prediction target—such as binary flood extent, multiclass flood-map labels, newly burned area, or cumulative burned area—must be established separately for each dataset.

---

## 2. Objectives

### Primary objectives

1. Build a reliable dataset pipeline for flood and wildfire progression.
2. Analyze the temporal and spatial properties of the available observations.
3. Integrate relevant satellite and environmental modalities.
4. Develop CNN-based spatial feature extraction.
5. Model temporal dependencies using ConvLSTM.
6. Represent spatial relationships using graph neural networks.
7. Predict future disaster-related spatial maps.
8. Evaluate the model using appropriate spatial and classification metrics.
9. Compare the proposed model against simpler baseline approaches.
10. Produce reproducible experiments and visualizations for research and team collaboration.

### Research questions

- Can historical observations predict future flood or wildfire extent?
- Does temporal modeling improve performance over single-observation baselines?
- Does adding terrain, weather, hydrological, or land-cover information improve predictions?
- Can graph-based spatial relationships improve modeling of disaster propagation?
- How does prediction performance change with forecast horizon?
- How well does the model generalize to unseen disaster events or regions?

---

## 3. Proposed Model Architecture

The planned architecture is:

**CNN → ConvLSTM → GNN → Prediction Decoder**

This architecture is a design target, not a claim that the complete model has already been implemented or trained.

### 3.1 CNN — Spatial Feature Extraction

The CNN encoder extracts spatial representations from input imagery and raster modalities.

Potential inputs include:

- Sentinel-1 radar imagery.
- Flood-related spatial maps.
- Elevation and terrain information.
- Weather and hydrological features.
- Land-cover information.

The CNN converts each observation into a spatial feature representation suitable for downstream temporal modeling.

### 3.2 ConvLSTM — Temporal Modeling

ConvLSTM models the evolution of spatial features over successive observations.

It is intended to capture:

- Changes in disaster extent.
- Spatial propagation patterns.
- Temporal dependencies.
- Relationships between previous and subsequent observations.

Irregular observation intervals must be handled explicitly through timestamp-aware sequence construction, interval features, or carefully defined sampling rules.

### 3.3 GNN — Spatial Relationship Modeling

The graph neural network represents relationships between spatial regions or cells.

Possible graph constructions include:

- Adjacent raster patches.
- Spatial grid neighborhoods.
- Hydrologically connected regions.
- Regions linked through geographic proximity.

Node features, graph edges, and graph construction rules must be defined and validated before the GNN stage is implemented.

### 3.4 Prediction Decoder

The decoder transforms the learned representation into the desired future spatial output.

Depending on the verified target format, this may be:

- A binary flood-probability map.
- A multiclass flood-extent map.
- A newly burned-area map.
- A cumulative burned-area map.

The prediction head and loss function will depend on the selected target.

### 3.5 Training strategy

The proposed development sequence is:

1. Validate data and labels.
2. Implement simple baselines.
3. Train the CNN-based baseline.
4. Add temporal modeling with ConvLSTM.
5. Add graph-based spatial modeling.
6. Evaluate the combined architecture.
7. Perform ablation studies and error analysis.

This staged approach will make it easier to determine which components contribute to performance.

---

## 4. Datasets

The project currently contains three dataset collections.

### 4.1 GFF — Flood Data

**Local directory:** `GFF/`

The GFF collection contains flood-related observations and environmental modalities.

| Component | Contents |
|---|---|
| `base/` | Flood-related raster observations, metadata JSON files, and GeoPackages |
| `dem/` | Local and contextual elevation rasters |
| `era5/` | ERA5 and ERA5-Land raster products |
| `glofas/` | Hydrological NetCDF files |
| `s1/` | Sentinel-1 raster observations |

Current inventory:

| Component | Files |
|---|---:|
| Base | 927 |
| DEM | 596 |
| ERA5 | 596 |
| GLOFAS | 298 |
| Sentinel-1 | 298 |
| **Total** | **2,715** |

The base directory contains 298 TIFF observations and 298 corresponding metadata JSON files. The TIFFs include two metadata formats:

- `kurosiwo`: 40 observations.
- `generated`: 258 observations.

The metadata includes fields such as `pre1_date`, `pre2_date`, `post_date`, `flooding`, `flooded_tiles`, and, for the richer format, additional flood-related metadata.

The labels and temporal semantics are still being verified.

### 4.2 SEN12-FLOOD

**Local directory:** `SEN12FLOOD/`

SEN12-FLOOD contains Sentinel-1 and Sentinel-2 imagery organized into sequence directories.

Initial inventory:

| Metric | Result |
|---|---:|
| Files | 28,035 |
| TIFF files | 27,981 |
| Other files | JSON, PNG, XML, and extensionless files |

The dataset requires acquisition-level grouping of individual raster bands into coherent observations. Individual band files must not be treated as separate time steps when they belong to the same acquisition.

Before using SEN12-FLOOD for future flood-extent forecasting, the availability and semantics of its labels must be checked. Dataset suitability for pixel-wise future prediction has not yet been established.

#### Quarantined files

Thirty-six TIFFs failed pixel-read validation and were moved to:

`audit/quarantine/sen12flood_bad_tiffs/`

Verification confirmed:

- Flagged files: 36.
- Quarantined: 36.
- Remaining in original locations: 0.
- Missing from both locations: 0.

These files should remain quarantined until their source or recovery options are understood. They should not be used for training.

### 4.3 TS-SatFire

**Local directory:** `TS-SatFire-selected/`

The selected TS-SatFire collection contains approximately 30 event directories.

The initial inventory reported:

| Metric | Result |
|---|---:|
| Files | 1,893 |
| File type | TIFF |
| Approximate size | 12.985 GB |

The event directories contain products such as:

- `FirePred`
- `VIIRS_Day`
- `VIIRS_Night`
- `ESRI_LULC`

The FirePred products are candidate targets for progression prediction. The exact label semantics must be verified before training.

In particular, the pipeline must establish whether the target represents newly burned area between observations or cumulative burned area. These are different prediction tasks and must not be mixed.

Dataset source repository:

https://github.com/zhaoyutim/TS-SatFire

---

## 5. Current Project Progress

This section distinguishes completed work from preliminary findings and remaining tasks.

### Phase 1 — Dataset Inventory

**Status: Completed**

The inventory script was executed successfully.

Reports:

- `audit/reports/step1_file_inventory.csv`
- `audit/reports/step1_inventory_summary.json`

Initial dataset inventory:

| Dataset | Files | Approximate size |
|---|---:|---:|
| GFF | 2,715 | 28.03 GB |
| SEN12-FLOOD | 28,035 | 13.789 GB |
| TS-SatFire-selected | 1,893 | 12.985 GB |
| **Total** | **32,643** | **54.80 GB** |

These are the initial inventory figures; they are not the amount of free or occupied storage currently available on the machine.

### Phase 2 — Raster Validation

**Status: Initial validation completed**

The raster validation script inspected 31,662 TIFF files.

| Dataset | TIFFs inspected | Open errors | Pixel-sample errors |
|---|---:|---:|---:|
| GFF | 1,788 | 0 | 0 |
| SEN12-FLOOD | 27,981 | 0 | 36 |
| TS-SatFire-selected | 1,893 | 0 | 0 |
| **Total** | **31,662** | **0** | **36** |

All 31,662 TIFFs opened sufficiently for the initial metadata inspection. However, 36 SEN12-FLOOD files failed pixel sampling.

A successful metadata open does not guarantee that the raster's pixel data can be read successfully.

Reports:

- `audit/reports/step2_raster_validation.csv`
- `audit/reports/step2_raster_validation_summary.txt`
- `audit/reports/step2_raster_errors.csv`
- `audit/reports/step2_raster_recheck.csv`

### Phase 3 — Failed TIFF Quarantine

**Status: Completed**

The 36 failed SEN12-FLOOD TIFFs were quarantined and the move was verified.

No raw files outside the flagged list were intentionally removed.

### Phase 4 — GFF Alignment Audit

**Status: Partial; the initial report is incomplete**

The first alignment script recognized 40 base observations using a filename pattern that did not cover the other 258 base observations.

For those 40 recognized observations, the script found matching files for the selected DEM, ERA5, ERA5-Land, GLOFAS, and prior Sentinel-1 inputs.

This does **not** establish complete alignment for all 298 base observations. The filename parser must be corrected before the alignment report can be considered complete.

The current report is:

`audit/reports/step4_gff_alignment.csv`

### Phase 5 — GFF Metadata Audit

**Status: Completed for metadata parsing**

The metadata audit examined all 298 base TIFFs and their JSON files.

| Check | Result |
|---|---:|
| Base TIFFs | 298 |
| Metadata files found | 298 |
| Metadata parsing errors | 0 |
| `kurosiwo` observations | 40 |
| `generated` observations | 258 |
| `flooding: true` | 88 |
| `flooding: false` | 210 |

Report:

`audit/reports/step6_gff_base_metadata.csv`

The `flooding` counts represent metadata fields, not yet independently verified pixel-level ground-truth counts.

### Phase 6 — Initial Label Inspection

**Status: Preliminary investigation**

Small windows were sampled from six representative GFF rasters.

Observed characteristics:

- The sampled rasters use `uint8` pixel data.
- The inspected rasters report `255` as NoData.
- Some valid sampled windows contain values such as `0`, `2`, and `3`.
- Some windows contain no valid pixels, while other windows of the same raster contain valid data.
- At least one generated observation returned no valid pixels in three sampled windows.

These results do not prove that the corresponding complete rasters are corrupt or empty. The meaning of the pixel values and the relationship between raster masks and metadata labels remain to be verified.

The following scripts support this investigation:

- `audit/scripts/gff_label_sample.py`
- `audit/scripts/gff_masked_window_check.py`

### Overall status

The project has completed its initial dataset inventory, raster metadata inspection, failed-file quarantine, and GFF metadata parsing.

The next milestone is to produce a complete, trustworthy alignment and label audit before constructing training sequences.

No completed model training, final evaluation metrics, or validated model-ready temporal dataset is claimed in this README.

---

## 6. Repository Structure

The intended repository structure is:

```text
Disaster-Progression-Predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── audit/
│   ├── scripts/
│   │   ├── inventory.py
│   │   ├── raster_validation.py
│   │   ├── recheck_failed_rasters.py
│   │   ├── diagnose_tiff_failures.py
│   │   ├── gff_alignment_audit.py
│   │   ├── gff_base_metadata_audit.py
│   │   ├── gff_label_sample.py
│   │   └── gff_masked_window_check.py
│   ├── reports/
│   └── quarantine/
│
├── configs/
│   ├── flood.yaml
│   └── wildfire.yaml
│
├── manifests/
│   ├── gff_manifest.csv
│   ├── wildfire_manifest.csv
│   └── split_assignments.csv
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── sequence_builder.py
│   ├── models/
│   │   ├── cnn_encoder.py
│   │   ├── convlstm.py
│   │   ├── gnn.py
│   │   └── predictor.py
│   └── training/
│       ├── train.py
│       └── evaluate.py
│
├── tests/
├── notebooks/
├── docs/
└── artifacts/
    └── README.md
```

**Note:** this is the target structure, not a claim that every directory and source file already exists. The audit scripts and reports listed in the current progress section are the existing work; the model implementation and manifests are planned additions.

---

## 7. Technology Stack

The planned technology stack is:

| Component | Technology |
|---|---|
| Programming language | Python |
| Numerical computing | NumPy |
| Tabular data and manifests | Pandas |
| Raster reading and geospatial metadata | Rasterio, GDAL |
| Scientific data | xarray and/or NetCDF4, subject to availability |
| Deep learning | PyTorch |
| Spatial feature extraction | CNN |
| Temporal modeling | ConvLSTM |
| Graph modeling | PyTorch Geometric or another compatible graph framework |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib |
| Experiment tracking | Weights & Biases, optional |
| Version control | Git and GitHub |

Only the dependencies required by the current audit should be installed initially. The full model dependencies can be added as implementation progresses.

---

## 8. Environment Setup

### Prerequisites

- Python compatible with the selected dependency versions.
- Git.
- Sufficient disk space for the selected dataset and preprocessing outputs.
- Optional CUDA-compatible GPU and supported PyTorch/CUDA configuration.

The current development machine has been using Python 3.13. Model and geospatial dependency compatibility should be tested before fixing the project's final Python version.

### Create a virtual environment

Windows PowerShell:

```powershell
cd "C:\Disaster Progression Dataset"

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

If PowerShell blocks activation, use the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### Install audit dependencies

Create or update `requirements.txt` with the dependencies currently needed:

```text
numpy>=1.26,<3.0
pandas>=2.1,<4.0
rasterio>=1.3.9,<2.0
```

Then install:

```powershell
python -m pip install -r requirements.txt
```

Verify the installation:

```powershell
python -c "import numpy, pandas, rasterio; print('Dependencies OK')"
```

These are initial audit dependencies, not a finalized model-training requirements file. Additional packages will be added and pinned after testing the model environment.

---

## 9. Dataset Organization

The raw datasets currently reside outside the proposed Git repository structure.

Example local organization:

```text
C:\Disaster Progression Dataset\
├── GFF\
├── SEN12FLOOD\
├── TS-SatFire-selected\
├── audit\
│   ├── scripts\
│   ├── reports\
│   └── quarantine\
└── requirements.txt
```

The exact project-root arrangement should be standardized before the preprocessing pipeline is finalized.

### Path configuration

Training scripts should not hard-code a developer-specific absolute path.

Instead, configure a dataset root through a local configuration file or environment variable.

For example:

```text
DATA_ROOT=C:\Disaster Progression Dataset
```

Do not commit machine-specific paths if the repository is intended for multiple teammates.

### Manifests

Each dataset manifest should include, where applicable:

- Observation or event identifier.
- ROI or spatial region identifier.
- Acquisition timestamp.
- Target timestamp.
- Input modality paths.
- Target-map path.
- Dimensions, band count, CRS, and spatial bounds.
- Label metadata and NoData definition.
- Temporal gap between observations.
- Dataset split.
- Validation status.
- Preprocessing version.

Manifest paths should be relative to the configured dataset root.

---

## 10. Data Processing Pipeline

The intended processing workflow is:

### Stage 1 — Inventory and integrity checks

- Enumerate the dataset files.
- Identify file types and sizes.
- Validate raster metadata.
- Test representative pixel reads.
- Record unreadable or suspicious files.

### Stage 2 — Metadata and label validation

- Parse metadata JSON files.
- Identify target dates and historical observation dates.
- Verify flood and wildfire label semantics.
- Investigate NoData values.
- Determine whether labels are binary, categorical, or continuous.
- Document invalid and excluded observations.

### Stage 3 — Temporal and spatial alignment

- Match observations by event or ROI.
- Associate each target with the correct historical inputs.
- Verify spatial bounds, CRS, resolution, and pixel-grid alignment.
- Record acquisition intervals.
- Confirm that all required modalities are available.

A matching ROI identifier alone is not sufficient to establish spatial or temporal alignment.

### Stage 4 — Preprocessing

- Apply consistent raster transformations.
- Normalize continuous input features using training data only.
- Preserve categorical label values and NoData masks.
- Define a common spatial grid or an appropriate patch-based representation.
- Record every transformation in the preprocessing configuration.

### Stage 5 — Sequence generation

For each valid event or ROI, construct sequences such as:

```text
Historical observation 1
        ↓
Historical observation 2
        ↓
Historical observation 3
        ↓
Target future observation
```

The number of historical observations and forecast horizon will be selected based on the dataset's temporal structure.

### Stage 6 — Training-ready artifacts

Generate:

- Dataset manifests.
- Train/validation/test split assignments.
- Small sanity-check samples.
- Sequence indexes.
- Preprocessing metadata.

Large intermediate rasters should be stored outside normal Git history.

---

## 11. Experimental Design

Flood and wildfire experiments will be developed separately.

### 11.1 Flood pipeline

Candidate inputs:

- Sentinel-1 observations.
- DEM and terrain features.
- ERA5 and ERA5-Land weather features.
- GLOFAS hydrological features.
- Historical flood-related maps, where valid and appropriate.

Candidate target:

- Future flood extent or a verified flood-related spatial label.

The exact target definition, temporal horizon, and suitable subset of GFF observations remain to be finalized.

SEN12-FLOOD may provide complementary data, but its suitability for pixel-wise future flood prediction must be verified rather than assumed.

### 11.2 Wildfire pipeline

Candidate inputs:

- Historical FirePred products.
- VIIRS Day and Night products.
- Land-cover information.
- Additional environmental features where available and appropriately aligned.

Candidate targets:

- Newly burned area.
- Cumulative burned area.

The two targets must be treated as separate prediction formulations.

### 11.3 Baseline models

Before training the full architecture, establish baselines such as:

1. Persistence baseline using the most recent valid observation.
2. CNN-based spatial baseline.
3. CNN + ConvLSTM temporal baseline.
4. CNN + ConvLSTM + GNN full model.

The persistence baseline is important because disaster extent can be spatially persistent over short forecast horizons.

### 11.4 Ablation studies

After the full model is operational, evaluate:

- CNN only.
- CNN + ConvLSTM.
- CNN + GNN.
- CNN + ConvLSTM + GNN.
- Multimodal versus reduced-modality input configurations.

These experiments will help determine whether temporal and graph components provide meaningful improvements.

---

## 12. Evaluation Metrics

Metrics will be selected according to the target representation and the intended use.

### For binary spatial maps

- Intersection over Union (IoU).
- Dice coefficient.
- Precision.
- Recall.
- F1-score.
- Precision-recall curves.

### For probabilistic predictions

- Binary cross-entropy.
- Brier score.
- Calibration analysis.

### For multiclass maps

- Per-class IoU.
- Macro-F1.
- Confusion matrix.
- Class-wise precision and recall.

### For progression prediction

- Metrics at each forecast horizon.
- Spatial error visualizations.
- Comparison against the persistence baseline.
- Performance on unseen events or regions.

Overall accuracy should not be used alone when flood or burned pixels are rare. Metrics must be reported on held-out data and should reflect class imbalance and spatial prediction quality.

No model performance values are reported yet because a completed, validated training and evaluation run has not been established in the current project status.

---

## 13. Next Plan of Action

The following roadmap prioritizes data correctness before computationally expensive training.

### Phase A — Complete dataset auditing

**Status: In progress**

- [x] Inventory all three dataset collections.
- [x] Validate raster opening and basic metadata.
- [x] Identify and quarantine 36 unreadable SEN12-FLOOD TIFFs.
- [x] Parse metadata for all 298 GFF base observations.
- [x] Inspect representative GFF label windows.
- [ ] Correct the GFF filename parser to support both `kurosiwo` and `generated` observations.
- [ ] Re-run GFF alignment for all 298 base observations.
- [ ] Investigate missing georeferencing warnings.
- [ ] Verify pixel-label semantics and NoData handling.
- [ ] Produce a final dataset audit summary.

**Deliverable:** A validated dataset manifest and documented exclusions.

### Phase B — Finalize prediction targets

- [ ] Define the flood target representation.
- [ ] Decide which GFF observations are valid training candidates.
- [ ] Verify whether SEN12-FLOOD supplies suitable future pixel-level labels.
- [ ] Determine whether TS-SatFire targets represent new burns or cumulative burned area.
- [ ] Define forecast horizons and acceptable temporal gaps.

**Deliverable:** A target specification for each dataset.

### Phase C — Build aligned temporal sequences

- [ ] Match historical observations with future targets.
- [ ] Align spatial grids and verify CRS and bounds.
- [ ] Define a consistent tensor representation.
- [ ] Preserve NoData masks.
- [ ] Generate sequence manifests.
- [ ] Split by event or ROI before generating overlapping windows.
- [ ] Inspect sample sequences visually.

**Deliverable:** Reproducible, model-ready sequence indexes and preprocessing outputs.

### Phase D — Establish baselines

- [ ] Implement the persistence baseline.
- [ ] Train and evaluate a CNN baseline.
- [ ] Add ConvLSTM temporal modeling.
- [ ] Record experiments and configurations.
- [ ] Evaluate using appropriate spatial metrics.

**Deliverable:** Baseline results for comparison with the full model.

### Phase E — Implement the proposed architecture

- [ ] Define the graph representation.
- [ ] Implement the GNN module.
- [ ] Integrate CNN spatial features, ConvLSTM temporal features, and graph features.
- [ ] Implement the prediction decoder.
- [ ] Define losses and training configuration.
- [ ] Validate model shapes and gradient flow.
- [ ] Train on the training split.
- [ ] Select configurations using validation data.

**Deliverable:** A working CNN → ConvLSTM → GNN prediction model.

### Phase F — Evaluate and improve

- [ ] Evaluate on held-out events or regions.
- [ ] Compare against baseline models.
- [ ] Perform modality and architecture ablations.
- [ ] Investigate false positives and false negatives.
- [ ] Analyze performance across forecast horizons.
- [ ] Document limitations and failure cases.

**Deliverable:** Reproducible evaluation results and visualizations.

### Phase G — Collaboration and research documentation

- [ ] Finalize the README and dataset documentation.
- [ ] Version the manifests and split assignments.
- [ ] Store model configurations and checkpoints appropriately.
- [ ] Document dataset licensing and access instructions.
- [ ] Prepare research figures and a progress presentation.

**Deliverable:** A reproducible project repository suitable for collaborative development and research reporting.

---

## 14. Team Collaboration

The project is intended to support parallel development after the data interface and target definitions have been stabilized.

Possible work allocation:

| Workstream | Responsibilities |
|---|---|
| Dataset and preprocessing | Data validation, manifests, alignment, sequence generation |
| Spatial encoder | CNN implementation and spatial feature extraction |
| Temporal modeling | ConvLSTM implementation and temporal experiments |
| Graph modeling | Graph construction and GNN implementation |
| Training and evaluation | Training loops, metrics, visualization, experiment tracking |
| Integration | Model interfaces, configuration, testing, documentation |

These are suggested responsibilities, not a record of assigned team members.

### Collaboration conventions

- Use Git branches for individual features.
- Open pull requests for substantial changes.
- Keep data paths configurable.
- Record preprocessing and training configurations.
- Avoid committing raw datasets or large generated files.
- Share dataset manifests and reproducible instructions.
- Use the same split assignments across model comparisons.

---

## 15. Reproducibility and Data Leakage Prevention

Data leakage is a major risk in disaster-progression forecasting because many observations can originate from the same event or spatial region.

The following rules will be applied:

1. Split by whole event or ROI before generating overlapping temporal windows.
2. Do not allow windows from the same held-out event to appear in training.
3. Fit normalization parameters using the training split only.
4. Use validation data for model selection.
5. Keep the test split untouched until final evaluation.
6. Record random seeds, configuration files, dependency versions, and split assignments.
7. Track the source and timestamps of every input and target.
8. Ensure that no future observation is inadvertently included in historical input features.
9. Keep flood and wildfire evaluation protocols separate unless a shared evaluation is explicitly justified.

Because observation timestamps may be irregular, the pipeline should record the time difference between successive observations and the target horizon.

---

## 16. Known Issues and Limitations

The following issues remain open:

1. **Incomplete GFF alignment report:** the current script recognized only 40 of 298 base observations. The filename parser must be corrected before the alignment report is trusted.
2. **Unverified label semantics:** sampled GFF pixel values include `0`, `2`, and `3`; their exact meanings need confirmation.
3. **NoData interpretation:** `255` is reported as NoData in sampled GFF rasters. Masked windows do not establish that an entire raster is empty.
4. **Spatial consistency:** CRS, bounds, resolution, and pixel-grid alignment need to be verified across the selected modalities.
5. **Temporal consistency:** historical inputs and targets may have different acquisition dates. Forecast horizons must be explicitly defined.
6. **SEN12-FLOOD labels:** suitability for pixel-wise future flood forecasting remains unverified.
7. **Wildfire target semantics:** newly burned area versus cumulative burned area must be resolved before training.
8. **Model implementation:** the full CNN → ConvLSTM → GNN architecture has not yet been documented as completed.
9. **Performance evaluation:** no validated baseline or full-model results are currently reported.
10. **Compute and storage:** large-scale preprocessing and model training must account for available SSD capacity and hardware.

These limitations will be resolved in the order described in the roadmap.

---

## 17. References and Dataset Documentation

- **TS-SatFire repository:** https://github.com/zhaoyutim/TS-SatFire
- **SEN12-FLOOD:** Consult the original dataset documentation for acquisition structure, labels, permitted use, and redistribution terms.
- **GFF:** Consult the original dataset documentation and metadata schema to verify flood-map values, observation semantics, and licensing.

Dataset descriptions in this README reflect the files inspected locally. Source-specific definitions and license terms should be confirmed against the authoritative dataset documentation before publication or redistribution.

---

## Project Status Summary

The project has completed the initial dataset inventory, raster metadata checks, quarantine of 36 unreadable SEN12-FLOOD TIFFs, and parsing of all 298 GFF base metadata files.

The immediate priority is to complete the GFF alignment and label audit, define valid prediction targets, and construct leakage-free temporal sequences. Baseline experiments will follow, before the full CNN → ConvLSTM → GNN architecture is trained and evaluated.

**The next major milestone is a validated, reproducible dataset pipeline—not a claimed model accuracy.**
