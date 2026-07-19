# Cross-Attentive Dual-Graph Network for Pedestrian Crossing Intention Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Master's thesis — Computer Engineering (AI curriculum)  
> Target venues: IEEE ITSC · IV Conference · IET ITS

---

## Overview

This repository contains the implementation of **CA-DGN** (Cross-Attentive Dual-Graph Network), a novel architecture for pedestrian crossing intention prediction (PCIP) in autonomous driving scenarios.

**Key contributions:**
1. **CGAM** (Cross-Graph Attention Module) — bidirectional node-level cross-attention between the skeleton graph and the heterogeneous scene graph. First paper to do this in PCIP.
2. **Causally-aware ego-vehicle modeling** — only initial speed used as input, avoiding the documented causal confusion from full speed sequences (Azarmi et al., 2024; Ling et al., 2024).
3. **Explicit traffic light state** (R/G/Y) in a semantically sparse scene graph with attention-weighted edges.

Evaluated on **PIE** and **JAAD** benchmarks (Kotseruba et al., WACV 2021).

---

## Architecture

```
Branch A (Skeleton GCN)          Branch B (Scene GAT)
   17 AlphaPose keypoints            pedestrian + vehicles +
   [x, y, confidence] × 16f         traffic light + crosswalk
           │                                  │
           ▼                                  ▼
    ST-GCN layers                    GAT layers (sparse)
   {h_j1,...,h_j17}              {h_ped, h_tl, h_veh, h_cw}
           │                                  │
           └──────────── CGAM ───────────────┘
                  (bidirectional cross-attention)
                           │
                    GRU temporal
                           │
                      FC → P(crossing)
```

---

## Results

### PIE Dataset

| Model | Acc | F1 | AUC | Recall |
|-------|-----|----|-----|--------|
| Baseline GRU (ours) | - | - | - | - |
| CA-DGN (ours) | - | - | - | - |
| RAIDN (Yang et al., T-ITS 2024) | 0.92 | 0.85 | 0.89 | 0.89 |
| Dual-STGAT (Lian et al., T-ITS 2025) | 0.86 | 0.91 | 0.87 | 0.90 |

*Results will be filled in after training.*

---

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### PIE Dataset
Download from [PIE official repo](https://github.com/aras62/PIE).  
Place annotations under `data/PIE/annotations/` with the following structure:
```
data/PIE/annotations/
    set01/video_0001/annotations.xml
    set01/video_0002/annotations.xml
    ...
    set06/...
```

Video frames (needed for AlphaPose):
```
data/PIE/images/
    set01/video_0001/
        00000.png
        00001.png
        ...
```

---

## Usage

### 1. Baseline (annotations only, no frames needed)
```bash
python train.py --config configs/baseline.yaml
```

### 2. Build pose cache (requires AlphaPose + video frames)
```bash
python scripts/build_pose_cache.py \
    --pie_root data/PIE \
    --output_dir cache/alphapose \
    --set_ids set02
```

### 3. Full model
```bash
python train.py --config configs/cadgn.yaml
```

### 4. Evaluate
```bash
python evaluate.py \
    --config configs/baseline.yaml \
    --checkpoint checkpoints/baseline/best_model.pt \
    --split test
```

---

## Repository Structure

```
├── configs/             # YAML configuration files
│   ├── baseline.yaml    # Baseline GRU
│   └── cadgn.yaml       # Full CA-DGN (coming)
├── data/
│   └── pie_dataset.py   # PIE parser + PyTorch Dataset
├── models/
│   └── models.py        # BaselineGRU, BranchA/B, CGAM, CA-DGN
├── notebooks/
│   └── 01_data_exploration.ipynb
├── scripts/
│   └── build_pose_cache.py   # AlphaPose preprocessing
├── train.py
├── evaluate.py
└── requirements.txt
```

---

## Citation

*Paper under submission — citation will be added after publication.*

---

## Acknowledgements

- PIE dataset: [Rasouli et al., ICCV 2019](https://github.com/aras62/PIE)
- JAAD dataset: [Rasouli et al., ICCVW 2017](https://github.com/ykotseruba/JAAD)
- Benchmark protocol: [Kotseruba et al., WACV 2021](https://github.com/ykotseruba/PedestrianActionBenchmark)
