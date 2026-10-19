# Federated Fundus Classification Reproduction

Auditable, executable reproduction and modernization of the computational protocol described in:

> Jiang, R., Kumar, Y., and Kruger, D. (2025). *Enhanced Privacy-Preserving Architecture for Fundus Disease Diagnosis with Federated Learning*. Applied Sciences, 15(6), 3004. https://doi.org/10.3390/app15063004

The released source at [rayzjiang/Enhanced-Privacy-Preserving-Architecture-for-Fundus-Disease-Diagnosis-with-Federated-Learning](https://github.com/rayzjiang/Enhanced-Privacy-Preserving-Architecture-for-Fundus-Disease-Diagnosis-with-Federated-Learning) was inspected as primary implementation evidence at commit `6a2c21ce4b6dfbb3f2f9cb170041c198ef8ee3c8`. It has no declared software license, so this package is an independently written implementation rather than a copy. It does **not** claim to reproduce the paper's reported 84.88% or 86.63% accuracies. Those values remain historical paper results until a run produces traceable predictions from the original data selection.

## What is implemented

- ODIR metadata ingestion using eye-specific filenames and diagnostic keywords.
- Two explicit endpoints: the released code's three-output multi-label task and a corrected exclusive three-class task derived from official eye-specific keywords.
- Patient-separated 64/16/20 train/validation/test splits.
- Patient-separated assignment to four federated clients.
- RGB CLAHE, non-background bounding-box crop, 224x224 resizing, train-only 50% horizontal and vertical flips, and ImageNet normalization.
- ImageNet-pretrained VGG-19 with frozen feature layers and a three-output classifier.
- Centralized, local-only, and sample-weighted FedAvg training.
- Four clients, two reproducibly selected per round, five rounds, one local epoch, batch size 16, Adam, and learning rate 0.001.
- Accuracy, balanced accuracy, macro-F1, one-vs-rest AUROC/AUPRC, per-class sensitivity/specificity, confusion matrix, prediction ledger, client schedule, processed-example count, elapsed time, and transmitted-byte accounting.
- Audit commands that fail on patient leakage, missing files, duplicate image assignments, or non-normalized aggregation weights.

## Reproduction boundaries

The publication leaves several choices unspecified. This implementation never silently invents them:

1. The exact rule that produced 3,402 images is unknown. The manifest records the rule used and reports the resulting count.
2. The released code uses unseeded `random_split`; therefore its original assignments and five-round schedule are not recoverable. This implementation makes both deterministic and saves them.
3. The paper describes random, even client splits but does not establish patient separation. The default corrected configuration groups both eyes from a patient in one split and one client.
4. Sample-size weighting is canonical FedAvg, not a new aggregation algorithm.
5. The released code contains no clipping, noise injection, privacy accountant, or secure aggregation. This package does not label ordinary FL as DP. Privacy experiments must be added with clipping, noise, protected unit, accountant, epsilon, and delta explicitly recorded.

## Data placement

Obtain ODIR under its applicable terms. Put the official metadata workbook/CSV and images outside Git, for example:

```text
retina_fl_reproduction/data/odir/
  ODIR-5K_Training_Annotations(Updated)_V2.xlsx
  Training Images/
    0_left.jpg
    0_right.jpg
```

Raw medical images, manifests containing sensitive identifiers, model checkpoints, and predictions are ignored by Git.

## Environment

Python 3.11 or 3.12 is recommended. PyTorch currently has limited support for Python 3.13 on some platforms.

```powershell
cd retina_fl_reproduction
python -m venv .venv
.venv\Scripts\python -m pip install -e .
```

## Prepare and audit the real manifest

```powershell
retina-fl prepare --config configs/corrected_odir.yaml
retina-fl audit --manifest runs/odir_corrected/manifest.csv
```

`prepare` writes a transparent selection report containing accepted images, exclusion reasons, patient counts, class counts, split counts, and client counts.

## Run the baselines

```powershell
retina-fl train centralized --config configs/corrected_odir.yaml
retina-fl train local --config configs/corrected_odir.yaml
retina-fl train federated --config configs/corrected_odir.yaml
```

Use `configs/paper_hyperparameters_patient_safe.yaml` to study the reported hyperparameters while retaining the minimum patient-separation safeguard. The paper does not provide the original patient assignments, so an exact historical split cannot be recovered from the manuscript. A run directory is immutable unless `--overwrite` is explicitly supplied.

## Evidence produced by every real run

```text
runs/<run_id>/
  resolved_config.yaml
  environment.json
  manifest.csv
  manifest_audit.json
  client_schedule.json
  round_metrics.csv
  predictions.csv
  metrics.json
  communication.json
  model.pt
```

No metric enters a paper until the associated configuration, manifest hash, predictions, and model checksum exist.

## Extension path

The next study can add source/device partitions, BRSET and mBRSET adapters, patient-level DP accounting, frozen-encoder versus compact-adapter ablations, and held-out-source evaluation without changing the evidence contract above.

