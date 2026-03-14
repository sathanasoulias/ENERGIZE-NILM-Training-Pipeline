# ENERGIZE NILM

<p align="center">
  <img src="docs/energise_banner.png" alt="ENERGIZE Project Banner" width="350"/>
</p>

<p align="center">
  <b>ENERGIZE Project</b> — DNN-based Non-Intrusive Load Monitoring
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

---

## Overview

**ENERGIZE NILM** is the baseline deep learning code for appliance-level energy disaggregation developed within the **ENERGIZE** project. Given a single whole-home power signal from the **PLEGMA** dataset, the models learn to estimate the individual power consumption of target appliances — without any additional hardware sensors.

Two PyTorch architectures are provided:

| Model | Architecture | Strategy | Input Window |
|-------|-------------|----------|-------------|
| **CNN** | 1-D Convolutional Network | Seq2Point | 299 samples |
| **GRU** | Gated Recurrent Unit | Seq2Point | 199 samples |
| **TCN** | Temporal Convolutional Network | Seq2Seq | 600 samples |

---

## Dataset — PLEGMA

The **PLEGMA** dataset is a Greek residential smart meter dataset recorded at **10-second** intervals. It covers multiple households and includes sub-metered appliance readings used as ground truth for NILM training and evaluation.

**Supported appliances**

| Appliance | Threshold | Cutoff |
|-----------|-----------|--------|
| `boiler` | 800 W | 4000 W |
| `ac_1` | 50 W | 2300 W |
| `washing_machine` | 15 W | 2600 W |

**House splits** — models are evaluated on fully unseen houses:

| Appliance | Train | Validation | Test |
|-----------|-------|------------|------|
| `boiler` | 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13 | 10 | 2 |
| `ac_1` | 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13 | 5 | 1 |
| `washing_machine` | 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13 | 10 | 2 |

Download the PLEGMA dataset from the [official source](https://pureportal.strath.ac.uk/en/datasets/plegma-dataset) and place it under `data/PlegmaDataset_Clean/`.

---

## Project Structure

```
ENERGIZE-NILM/
├── main.py
├── requirements.txt
├── src_pytorch/          # Core library (models, training, evaluation, pipeline)
├── data/                 # Data pre-processing scripts and processed CSVs
├── outputs/              # Checkpoints, TensorBoard logs, metrics, predictions
├── notebooks/            # Colab-ready walkthroughs (data prep, training, evaluation)
└── docs/
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/energize-nilm.git
cd energize-nilm

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Step 1 — Prepare the data

```bash
cd data
python data.py --dataset plegma --appliance boiler
```

This reads the raw PLEGMA data, applies normalisation, and writes three CSV files to `data/processed/plegma/boiler/`:

```
training_.csv     validation_.csv     test_.csv
```

Repeat for any other appliance:

```bash
python data.py --dataset plegma --appliance ac_1
python data.py --dataset plegma --appliance washing_machine
```

### Step 2 — Train and evaluate

```bash
# Default experiment: boiler / TCN
python main.py

# Custom model
python main.py --dataset plegma --appliance boiler --model tcn

# Evaluate only (load an existing checkpoint)
python main.py --eval-only --checkpoint outputs/tcn_boiler/checkpoint/model.pt
```

Results are written to `outputs/<model>_<appliance>/metrics/`.

### Step 3 — Interactive notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_data_prep_training.ipynb` | Data preparation, normalisation, model training, and live training curves |
| `notebooks/02_evaluation.ipynb` | Load a trained checkpoint, run inference, compute metrics and generate visualisations |
| `notebooks/03_visualization.ipynb` | Extended result visualisations and multi-appliance comparison plots |

---

## Configuration

All static hyperparameters are defined in [src_pytorch/config.py](src_pytorch/config.py).
Only three values need to be set per experiment:

```python
DATASET_NAME   = 'plegma'
APPLIANCE_NAME = 'boiler'   # boiler | ac_1 | washing_machine
MODEL_NAME     = 'tcn'      # cnn | gru | tcn
```

**Key training parameters**

| Parameter | TCN | CNN | GRU | Description |
|-----------|-----|-----|-----|-------------|
| Epochs | 100 | 50 | 100 | Maximum training epochs |
| Early stopping patience | 20 | 10 | 20 | Epochs without val_loss improvement before stopping |
| Optimizer | Adam | Adam | Adam | β₁=0.9, β₂=0.999, ε=1e-8 |
| Loss | MSE | MSE | MSE | Mean squared error on normalised targets |

**Learning rate:** `0.001` (Adam) — shared across all appliances and models.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error in Watts — primary regression metric |
| **F1** | Harmonic mean of Precision and Recall computed on duration-filtered ON/OFF status: `F1 = 2 · (Precision · Recall) / (Precision + Recall)`. The duration filter is applied to both ground truth and predictions before scoring, so the metric reflects appliance-cycle detection quality rather than sample-level jitter (see [Postprocessing](#postprocessing)) |
| **Accuracy** | Overall ON/OFF classification accuracy |
| **Energy Error %** | Absolute relative error on total energy consumption (Wh) |

---

## Postprocessing

Raw predictions are converted to a binary ON/OFF status using a power threshold. To account for minimum-duration constraints of real appliances, `compute_status` in [src_pytorch/evaluator.py](src_pytorch/evaluator.py) applies a multi-pass duration filter:

1. **Remove short ON events** — runs shorter than `min_on` samples are forced to OFF.
2. **Fill short OFF gaps** — gaps of `≤ min_off` samples between two ON runs are bridged (merged into one event).
3. **Remove short ON events again** — catches boundary artefacts created by gap-filling.
4. **Optional stricter minimum committed duration** — removes ON events shorter than `min_committed_duration`. Used for the washing machine to reject spurious short activations: a real wash cycle always lasts at least ~30 minutes (180 samples at 10 s), so any ON event below that threshold is treated as a false detection even if it already passed the `min_on` gate.

The same filter is applied to both ground truth and predictions. The **F1** score is computed on the filtered status, providing a more realistic measure of appliance-cycle detection quality that is independent of within-cycle sample-level jitter.

**Per-appliance postprocessing parameters**

| Appliance | `min_on` | `min_off` | `min_committed_duration` |
|-----------|---------|----------|--------------------------|
| `boiler` | 30 samples (5 min) | 6 samples (1 min) | — |
| `ac_1` | 100 samples (≈17 min) | 50 samples (≈8 min) | — |
| `washing_machine` | 2 samples | 100 samples (≈17 min) | 180 samples (≈30 min) |

---

## Baseline Results — PLEGMA

All models are evaluated on fully held-out test houses.

### TCN (Seq2Seq)

| Appliance | MAE (W) | F1 | Accuracy | Energy Error % |
|-----------|---------|-----|----------|---------------|
| `boiler` | 13.02 | 0.9197 | 0.9960 | 4.94 |
| `ac_1` | 11.72 | 0.9554 | 0.9934 | 6.03 |
| `washing_machine` | 3.20 | 0.8627 | 0.9860 | 9.40 |

### CNN (Seq2Point)

| Appliance | MAE (W) | F1 | Accuracy | Energy Error % |
|-----------|---------|-----|----------|---------------|
| `boiler` | 9.84 | 0.9084 | 0.9964 | 8.87 |
| `ac_1` | 17.10 | 0.9460 | 0.9888 | 17.63 |
| `washing_machine` | 3.22 | 0.8294 | 0.9859 | 4.29 |

### GRU (Seq2Point)

| Appliance | MAE (W) | F1 | Accuracy | Energy Error % |
|-----------|---------|-----|----------|---------------|
| `boiler` | 7.99 | 0.9320 | 0.9975 | 6.66 |
| `ac_1` | 23.29 | 0.9178 | 0.9912 | 28.87 |
| `washing_machine` | 3.37 | 0.8542 | 0.9871 | 10.07 |

---

## Normalisation

- **Aggregate signal** — z-score: `(x − mean) / std`
- **Appliance signal** — cutoff scaling: `y / cutoff`
- During evaluation, predictions are denormalised and clipped to `[0, cutoff]` before metric calculation. Samples below the appliance threshold are zeroed out.

---

## Funding
<p align="center">
  <img src="docs/daiedge_logo.png" alt="dAIEDGE Logo" width="200"/>
</p>
<p align="center">

This project has received funding from the European Union's Horizon Europe programme **dAIEDGE** under grant agreement No. **101120726**. The work was carried out within the **ENERGIZE** project (sub-grant agreement dAI1OC1).



---

## Citation

If you use this code in your research, please cite the ENERGIZE project:

```bibtex
@misc{energize-nilm,
  title  = {ENERGIZE NILM: DNN-based Non-Intrusive Load Monitoring},
  year   = {2024},
  url    = {https://github.com/sathanasoulias/ENERGIZE}
}
```

---

## License

This project is released under the MIT License.
