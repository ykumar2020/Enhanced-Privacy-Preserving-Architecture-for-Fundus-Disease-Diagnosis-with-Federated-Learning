# Federated Fundus Disease Diagnosis

[![Open the A100 trial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ykumar2020/Enhanced-Privacy-Preserving-Architecture-for-Fundus-Disease-Diagnosis-with-Federated-Learning/blob/main/colab/ODIR_Federated_A100.ipynb)

This repository preserves the original implementation accompanying:

> Jiang, R., Kumar, Y., and Kruger, D. (2025). *Enhanced Privacy-Preserving Architecture for Fundus Disease Diagnosis with Federated Learning*. Applied Sciences, 15(6), 3004. https://doi.org/10.3390/app15063004

It also contains an independently written, auditable reproduction under [`reproduction/`](reproduction/). The historical files remain in `src/` and `train.py`; they are retained for provenance, not presented as corrected production code.

## Run a real ODIR trial on Google Colab

Open [`colab/ODIR_Federated_A100.ipynb`](colab/ODIR_Federated_A100.ipynb) in Colab, select an A100 GPU runtime, and run the cells in order. The notebook:

1. clones this repository;
2. installs the reproduction package;
3. downloads the public ODIR-5K distribution from Kaggle;
4. locates the official annotation workbook and training images;
5. creates patient-separated train/validation/test and client assignments;
6. audits the manifest for leakage and missing files;
7. runs centralized, local-only, or sample-weighted FedAvg training;
8. saves configurations, predictions, metrics, schedules, model hashes, and communication accounting.

The dataset is downloaded directly into the Colab runtime and is never committed to Git. Kaggle documents that public datasets can be downloaded without credentials; if access policy changes, store `KAGGLE_API_TOKEN` as a Colab secret rather than putting a token in the notebook.

## Local setup

```bash
git clone https://github.com/ykumar2020/Enhanced-Privacy-Preserving-Architecture-for-Fundus-Disease-Diagnosis-with-Federated-Learning.git
cd Enhanced-Privacy-Preserving-Architecture-for-Fundus-Disease-Diagnosis-with-Federated-Learning/reproduction
python -m pip install -e ".[colab]"
retina-fl acquire-odir
retina-fl prepare --config configs/runtime_colab_odir.yaml
retina-fl train federated --config configs/runtime_colab_odir.yaml
```

## What was corrected

- deterministic patient-level splitting rather than unseeded image-level `random_split`;
- both eyes from one patient remain in one evaluation split and one federated client;
- train-only random augmentation;
- exact example counts rather than batch counts for FedAvg weighting;
- valid client evaluation loaders and correct partial-batch denominators;
- explicit task and class ordering;
- per-class and imbalance-aware metrics in addition to accuracy;
- immutable evidence directories containing predictions and hashes;
- no claim that ordinary federated averaging is differential privacy.

The A100 notebook is a multi-client federated **simulation on one accelerator**. It executes real local client updates and aggregation but is not evidence of cross-hospital networking or a differential-privacy guarantee. Those require a separately specified deployment and privacy accountant.

## Data source

ODIR-5K was created from 5,000 patients with bilateral fundus photographs, diagnostic keywords, and eight patient-level disease labels. See the [official ODIR challenge dataset page](https://odir2019.grand-challenge.org/dataset/). The acquisition command uses the [public Kaggle distribution](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k), while recording the resolved local paths in `dataset_location.json`.

## Reproducibility boundary

The original paper's exact 3,402-image selection, historical random split, client schedule, and `labels_perEye.csv` were not published. Therefore 84.88% federated accuracy and 86.63% centralized accuracy remain historical paper values until independently regenerated from traceable predictions. The corrected configuration deliberately refuses to invent those missing artifacts.

See [`reproduction/docs/PROTOCOL_AND_GAPS.md`](reproduction/docs/PROTOCOL_AND_GAPS.md) for the source audit and extension plan.

## Upstream provenance

Forked from [`rayzjiang/Enhanced-Privacy-Preserving-Architecture-for-Fundus-Disease-Diagnosis-with-Federated-Learning`](https://github.com/rayzjiang/Enhanced-Privacy-Preserving-Architecture-for-Fundus-Disease-Diagnosis-with-Federated-Learning) at commit `6a2c21ce4b6dfbb3f2f9cb170041c198ef8ee3c8`.

The upstream repository does not declare a software license. The new `reproduction/` implementation is independently written; do not assume redistribution rights beyond GitHub's fork functionality without confirming them with the authors.
