"""
Training + Evaluation — Baseline GRU geometry-only (bbox + delta)
=================================================================
File unico: training, validation ed evaluation finale.

Uso:
    python train.py --config configs/baseline_base.yaml

Per ogni epoca calcola su TRAIN e VAL:
    loss, accuracy, f1, auc, precision, recall

Al termine valuta il best model su train/val/test e salva:
    - history.json   : tutte le metriche per epoca (train + val)
    - test_results.json
    - predictions.json : label/pred/prob per train, val, test (per le
                         confusion matrix e i plot)
Poi basta lanciare:
    python plot_metrics.py --exp_dir checkpoints/<nome_esperimento>
"""

import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    recall_score, precision_score, balanced_accuracy_score,
)

from data.pie_dataset import PIEDataset
from data.val_split import load_or_create_split, split_samples, describe_split
from models.models import BaselineGRU, TransformerModalityNet
from models.models_benchmark import BenchmarkSingleRNN
from plot_metrics import plot_metric_curves, plot_confusion_matrices
from losses import (
    build_criterion, resolve_loss_name, num_outputs_for, calibration_report,
)


# ─── Utility ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
    return device


def load_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def collate_fn(batch):
    """Collate: bbox, bbox_displacement, bbox_delta, ego_speed, label (+ ped_id meta)."""
    # cls/roi sono presenti solo se le feature visive sono attive
    keys_tensor = ["bbox", "bbox_displacement", "bbox_delta",
                   "pdm", "polar", "keypoints", "ego_speed", "label"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys_tensor}
    for opt in ("cls", "roi"):
        if opt in batch[0]:
            out[opt] = torch.stack([b[opt] for b in batch])
    out["ped_id"] = [b["ped_id"] for b in batch]
    return out


def _pdm_dim(cfg: Dict) -> int:
    """Dimensione delle feature PDM, coerente con la config del dataset."""
    return (2
            + int(cfg["data"].get("pdm_use_area_ratio", True))
            + 2 * int(cfg["data"].get("pdm_keep_absolute", False)))


_TRACK_DIMS = {"corners": 4, "center": 2, "center_size": 4, "bottom_center": 2}


def _track_dim(cfg: Dict, key: str) -> int:
    """Dimensione di displacement/delta secondo il punto di riferimento scelto."""
    return _TRACK_DIMS[cfg["data"].get(key, "corners")]


def _polar_dim(cfg: Dict) -> int:
    """Dimensione delle feature quasi-polari, coerente con la config."""
    d = 9 if cfg["data"].get("polar_include_sin", False) else 6
    return d * 2 if cfg["data"].get("polar_add_deltas", False) else d


_JOINTS_N = {"none": 17, "openpose18": 18, "gtranspdm": 20}


def _joints_mode(cfg: Dict) -> str:
    """
    Modalita' dei giunti. Se non specificata, il default segue l'encoder:
      stgat     -> openpose18 (18 kp: 17 COCO + collo, topologia Dual-STGAT)
      gtranspdm -> gtranspdm  (20 kp: + anca e centro corpo)
    Impostandola esplicitamente si puo' confrontare i due encoder a PARITA'
    di input, che per un'ablation pulita e' preferibile.
    """
    m = cfg["data"].get("skeleton_joints_mode")
    if m is not None:
        assert m in _JOINTS_N, f"skeleton_joints_mode non valido: {m}"
        return m
    enc = cfg["model"].get("skeleton_encoder", "gtranspdm")
    return "openpose18" if enc == "stgat" else "gtranspdm"


def _skeleton_joints(cfg: Dict) -> int:
    """Numero di giunti coerente con la config."""
    return _JOINTS_N[_joints_mode(cfg)]


class _SampleView(torch.utils.data.Dataset):
    """
    Vista su un PIEDataset con un sottoinsieme dei sample.

    Serve a dividere train/val senza rigenerare le sequenze PIE (operazione
    lenta): il dataset si costruisce una volta sola e poi si creano due viste
    che condividono lo stesso preprocessing.
    """

    def __init__(self, base: PIEDataset, samples):
        self.base = base
        self.samples = samples
        # eredita le dimensioni delle feature, servono a build_model
        for attr in ("pdm_dim", "polar_dim", "displacement_dim",
                     "delta_dim", "skeleton_joints"):
            if hasattr(base, attr):
                setattr(self, attr, getattr(base, attr))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # riusa __getitem__ del dataset base scambiando temporaneamente la lista
        real, self.base.samples = self.base.samples, self.samples
        try:
            return self.base[idx]
        finally:
            self.base.samples = real

    def get_class_weights(self):
        labels = np.array([s["label"] for s in self.samples])
        n_pos, n_neg, n = (labels == 1).sum(), (labels == 0).sum(), len(labels)
        w_pos = n / (2.0 * n_pos) if n_pos > 0 else 1.0
        w_neg = n / (2.0 * n_neg) if n_neg > 0 else 1.0
        print(f"  Class weights — crossing: {w_pos:.3f}, non-crossing: {w_neg:.3f}")
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)


def build_model(cfg: Dict, num_outputs: int = None) -> nn.Module:
    """Costruisce il modello.

    `num_outputs` (1 o 2) e' deciso dalla loss scelta in config: la testa e il
    tipo di loss devono essere coerenti, altrimenti il forward produce logit
    della forma sbagliata. Se None si usa il default storico di ogni modello.
    """
    name = cfg["model"]["name"]
    if name == "BaselineGRU":
        return BaselineGRU(
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"]["num_layers"],
            dropout=cfg["model"]["dropout"],
            use_bbox=cfg["model"].get("use_bbox", True),
            use_bbox_displacement=cfg["model"].get("use_bbox_displacement", False),
            use_bbox_delta=cfg["model"].get("use_bbox_delta", True),
            use_ego_speed=cfg["model"].get("use_ego_speed", False),
            **({} if num_outputs is None else {"num_outputs": num_outputs}),
        )
    if name == "BenchmarkSingleRNN":
        if num_outputs is not None and num_outputs != 1:
            raise ValueError(
                "BenchmarkSingleRNN ha la testa a 1 logit. Per usare "
                "training.loss='ce'/'mcel_ce' aggiungi il parametro "
                "num_outputs anche a models_benchmark.py, oppure usa "
                "loss='bce'/'mcel_bce'.")
        return BenchmarkSingleRNN(
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"].get("num_layers", 1),
            dropout=cfg["model"].get("dropout", 0.0),
            use_bbox=cfg["model"].get("use_bbox", True),
            use_bbox_displacement=cfg["model"].get("use_bbox_displacement", False),
            use_bbox_delta=cfg["model"].get("use_bbox_delta", False),
            use_ego_speed=cfg["model"].get("use_ego_speed", True),
        )
    if name == "TransformerModalityNet":
        # obs_len effettivo: con drop_first_frame la sequenza perde 1 frame.
        # Serve al pooling "flatten", la cui testa dipende da T fisso.
        _obs_len = cfg["data"]["obs_len"]
        if cfg["data"].get("drop_first_frame", False):
            _obs_len -= 1
        return TransformerModalityNet(
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"]["num_layers"],
            nhead=cfg["model"].get("nhead", 8),
            dropout=cfg["model"].get("dropout", 0.1),
            pooling=cfg["model"].get("pooling", "cls"),
            head_layers=cfg["model"].get("head_layers", 1),
            head_hidden=cfg["model"].get("head_hidden", None),
            head_dropout=cfg["model"].get("head_dropout", 0.1),
            obs_len=_obs_len,
            separate_encoder_speed_kinematics=cfg["model"].get(
                "separate_encoder_speed_kinematics", False),
            norm_first=cfg["model"].get("norm_first", False),
            use_input_layernorm=cfg["model"].get("use_input_layernorm", False),
            use_bbox=cfg["model"].get("use_bbox", True),
            # use_pdm / use_polar vivono sotto `data:` perche' le feature sono
            # calcolate nel dataset: unica fonte di verita', niente disallineamenti.
            use_pdm=cfg["data"].get("use_pdm", False),
            use_polar=cfg["data"].get("use_polar", False),
            pdm_dim=_pdm_dim(cfg),
            polar_dim=_polar_dim(cfg),
            displacement_dim=_track_dim(cfg, "displacement_type"),
            delta_dim=_track_dim(cfg, "delta_type"),
            use_bbox_displacement=cfg["model"].get("use_bbox_displacement", False),
            use_bbox_delta=cfg["model"].get("use_bbox_delta", False),
            use_ego_speed=cfg["model"].get("use_ego_speed", True),
            # skeleton: use_skeleton vive sotto `data:` (le pose nascono nel
            # dataset), i parametri architetturali sotto `model:`
            use_skeleton=cfg["data"].get("use_skeleton", False),
            skeleton_encoder=cfg["model"].get("skeleton_encoder", "gtranspdm"),
            skeleton_n_joints=_skeleton_joints(cfg),
            skeleton_hidden=cfg["model"].get("skeleton_hidden", 64),
            skeleton_layers=cfg["model"].get("skeleton_layers", 4),
            stgat_channels=tuple(cfg["model"].get("stgat_channels", [32, 64, 64])),
            stgat_kt=cfg["model"].get("stgat_kt", 9),
            stgat_velocity=cfg["model"].get("stgat_velocity", True),
            use_global_context=cfg["data"].get("use_global_context", False),
            use_roi=cfg["data"].get("use_roi", False),
            visual_dim=cfg["data"].get("visual_dim", 768),
            roi_size=cfg["data"].get("roi_size", 3),
            roi_reduce=cfg["model"].get("roi_reduce", "flatten"),
            visual_dropout=cfg["model"].get("visual_dropout", 0.1),
            cross_modal=cfg["model"].get("cross_modal", "off"),
            cross_modal_heads=cfg["model"].get("cross_modal_heads", 4),
            **({} if num_outputs is None else {"num_outputs": num_outputs}),
        )
    raise ValueError(f"Modello non supportato: {name}.")


# ─── Metriche ─────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray,
                    probs: np.ndarray) -> Dict[str, float]:
    """Calcola tutte le metriche a partire da label/pred/prob."""
    acc          = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    f1           = f1_score(labels, preds, pos_label=1, zero_division=0)
    recall       = recall_score(labels, preds, pos_label=1, zero_division=0)
    precision    = precision_score(labels, preds, pos_label=1, zero_division=0)
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs)
    else:
        auc = 0.0
    return {
        "acc":          float(acc),
        "balanced_acc": float(balanced_acc),
        "f1":           float(f1),
        "auc":          float(auc),
        "precision":    float(precision),
        "recall":       float(recall),
    }


# ─── Train / Eval di una epoca ────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None,
              grad_clip=1.0, desc="train") -> Tuple[Dict[str, float], Dict[str, list]]:
    """
    Esegue una passata completa sul loader.
    Se optimizer è passato -> training (backward + step), altrimenti eval.

    Returns:
        metrics : dict con loss, acc, f1, auc, precision, recall
        outputs : dict con liste labels/preds/probs (per confusion matrix)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total = 0.0, 0
    all_labels, all_preds, all_probs = [], [], []
    # Fase 1 — diagnostica: logit GREZZI (pre-clamping, pre-margine) e norma
    # del gradiente PRIMA del clipping.
    all_logits = []
    gradnorm_sum, gradnorm_n, gradnorm_clipped = 0.0, 0, 0

    pbar = tqdm(loader, desc=desc, leave=False, ncols=100)
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()

    with grad_ctx:
        for batch in pbar:
            bbox              = batch["bbox"].to(device)
            bbox_displacement = batch["bbox_displacement"].to(device)
            bbox_delta        = batch["bbox_delta"].to(device)
            ego_speed         = batch["ego_speed"].to(device)
            pdm               = batch["pdm"].to(device)
            polar             = batch["polar"].to(device)
            keypoints         = batch["keypoints"].to(device)
            cls               = batch["cls"].to(device) if "cls" in batch else None
            roi               = batch["roi"].to(device) if "roi" in batch else None
            labels            = batch["label"].to(device)

            if is_train:
                optimizer.zero_grad()

            if isinstance(model, TransformerModalityNet):
                logits = model(bbox, bbox_displacement, bbox_delta, ego_speed,
                               pdm=pdm, polar=polar, keypoints=keypoints,
                               cls=cls, roi=roi)
            else:
                logits = model(bbox, bbox_displacement, bbox_delta, ego_speed)

            if logits.dim() == 2 and logits.shape[-1] == 1:
                # testa a 1 logit: BCEWithLogitsLoss pesata / MCEL binaria
                logits1 = logits.squeeze(-1)
                loss = criterion(logits1, labels.float())
                probs = torch.sigmoid(logits1)
                preds = (probs >= 0.5).long()
                raw_logits = logits1
            else:
                # testa a 2 logit: CrossEntropyLoss pesata / MCEL (Eq. 10)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                preds = logits.argmax(dim=-1)
                raw_logits = logits

            if is_train:
                loss.backward()
                # clip_grad_norm_ restituisce la norma PRIMA del taglio: se resta
                # stabilmente sopra grad_clip, e' il clip a dominare il passo,
                # non la loss. Sintomo tipico di MCEL senza warmup del margine.
                _max_norm = (float(grad_clip)
                             if grad_clip is not None and float(grad_clip) > 0
                             else float("inf"))
                _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), _max_norm)
                _gn = float(_gn)
                gradnorm_sum += _gn
                gradnorm_n   += 1
                if _max_norm != float("inf") and _gn > _max_norm:
                    gradnorm_clipped += 1
                optimizer.step()

            total_loss += loss.item() * len(labels)
            total      += len(labels)
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_logits.extend(raw_logits.detach().float().cpu().numpy().tolist())

            running_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_acc:.3f}")

    labels_np = np.array(all_labels)
    preds_np  = np.array(all_preds)
    probs_np  = np.array(all_probs)
    logits_np = np.array(all_logits, dtype=np.float32)

    metrics = compute_metrics(labels_np, preds_np, probs_np)
    metrics["loss"] = total_loss / total

    # ── Fase 1: calibrazione + geometria dei logit ────────────────────────
    #   loss_plain     CE/BCE NON pesata sui logit grezzi. E' l'unica loss
    #                  confrontabile tra run con criterion diverse.
    #   ece            Expected Calibration Error.
    #   mlm            Mean Logit Margin (Eq. 1 del paper MCEL).
    #   logit_absmean  |logit| medio -> serve a tarare mcel_logit_scale.
    metrics.update(calibration_report(logits_np, probs_np, labels_np))

    if gradnorm_n > 0:
        metrics["gradnorm"] = gradnorm_sum / gradnorm_n
        metrics["gradclip_frac"] = gradnorm_clipped / gradnorm_n

    outputs = {
        "labels": all_labels,
        "preds":  all_preds,
        "probs":  all_probs,
        # i logit servono per il temperature scaling post-hoc (Fase 4)
        "logits": all_logits,
    }
    return metrics, outputs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="sovrascrive experiment.seed del config (per run multi-seed)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # seed: da riga di comando se fornito, altrimenti dal config
    seed = args.seed if args.seed is not None else cfg["experiment"]["seed"]
    cfg["experiment"]["seed"] = seed
    set_seed(seed)
    device = get_device()

    # se il seed e' passato esplicitamente, isola l'output in una sottocartella per-seed
    _exp_name = cfg["experiment"]["name"]
    if args.seed is not None:
        _exp_name = f"{_exp_name}/seed_{seed}"
    out_dir = Path(cfg["output"]["checkpoint_dir"]) / _exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    print(f"\nEsperimento: {cfg['experiment']['name']}")
    print(f"Config: {args.config}")
    print(f"Output: {out_dir}\n")

    # -- Dataset --------------------------------------------------------------
    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]

    _st = data_cfg.get("sample_type", "beh")
    bbox_norm  = data_cfg.get("bbox_normalization", "raw")
    speed_norm = data_cfg.get("speed_normalization", "raw")
    drop_ff    = data_cfg.get("drop_first_frame", False)
    # Modalita' dei giunti: dataset e modello devono concordare.
    _jmode = _joints_mode(cfg)
    if data_cfg.get("use_skeleton", False):
        print(f"\n[INFO] skeleton_joints_mode={_jmode} "
              f"({_JOINTS_N[_jmode]} keypoint)")

    _geom = dict(
        use_pdm=data_cfg.get("use_pdm", False),
        use_polar=data_cfg.get("use_polar", False),
        pdm_use_area_ratio=data_cfg.get("pdm_use_area_ratio", True),
        pdm_keep_absolute=data_cfg.get("pdm_keep_absolute", False),
        polar_include_sin=data_cfg.get("polar_include_sin", False),
        polar_add_deltas=data_cfg.get("polar_add_deltas", False),
        geom_anchor=data_cfg.get("geom_anchor", "center"),
        displacement_type=data_cfg.get("displacement_type", "corners"),
        delta_type=data_cfg.get("delta_type", "corners"),
        use_skeleton=data_cfg.get("use_skeleton", False),
        pose_dir=data_cfg.get("pose_dir", "data/poses"),
        skeleton_norm=data_cfg.get("skeleton_norm", "bbox_topleft"),
        skeleton_joints_mode=_jmode,
        pose_interpolate=data_cfg.get("pose_interpolate", False),
        pose_max_gap=data_cfg.get("pose_max_gap", 2),
        pose_interp_confidence=data_cfg.get("pose_interp_confidence", 0.5),
        pose_conf_threshold=data_cfg.get("pose_conf_threshold", 0.0),
        use_global_context=data_cfg.get("use_global_context", False),
        use_roi=data_cfg.get("use_roi", False),
        visual_feat_dir=data_cfg.get("visual_feat_dir", "features/dinov3"),
        roi_scale=data_cfg.get("roi_scale", 1.5),
        roi_size=data_cfg.get("roi_size", 3),
        visual_dim=data_cfg.get("visual_dim", 768),
    )
    
    # ── Modalita' di validation ed evaluation test ───────────────────────
    #   use_val_from_train: true  -> val = fetta stratificata dal TRAIN (split pedone)
    #   use_validation: true      -> val = split "val" ufficiale PIE
    #   eval_test_every_epoch: true -> valuta il TEST set ad ogni epoca (oltre a train/val)
    use_val_from_train = train_cfg.get("use_val_from_train", False)
    use_validation_cfg = train_cfg.get("use_validation", True)
    eval_test_every_epoch = train_cfg.get("eval_test_every_epoch", False)

    if use_val_from_train and use_validation_cfg:
        print("\n[INFO] Sia use_val_from_train che use_validation sono True. "
              "Prevale use_val_from_train (validation estratto dal train set).")

    if use_val_from_train:
        val_type = "val_from_train"
    elif use_validation_cfg:
        val_type = "val_pie"
    else:
        val_type = None

    full_train_ds = PIEDataset(data_cfg["annotation_root"], split="train",
                               obs_len=data_cfg["obs_len"],
                               bbox_normalization=bbox_norm,
                               speed_normalization=speed_norm,
                               drop_first_frame=drop_ff,
                               **_geom,
                               sample_type=_st)

    if val_type == "val_from_train":
        print("\n[INFO] Validation set estratto dal train (split per pedone)")
        val_ids = load_or_create_split(
            full_train_ds.samples,
            path=train_cfg.get("val_split_file", "data/val_split.json"),
            n_val=train_cfg.get("val_n_peds", 80),
            seed=train_cfg.get("val_split_seed", 42),
            stratified=train_cfg.get("val_stratified", True),
        )
        tr_samples, va_samples = split_samples(full_train_ds.samples, val_ids)
        describe_split(tr_samples, va_samples)

        train_ds = _SampleView(full_train_ds, tr_samples)
        val_ds = _SampleView(full_train_ds, va_samples)
    else:
        train_ds = full_train_ds
        val_ds = None

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True, num_workers=0, collate_fn=collate_fn)
    train_eval_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                                   shuffle=False, num_workers=0, collate_fn=collate_fn)

    if val_type == "val_from_train":
        val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"],
                                shuffle=False, num_workers=0, collate_fn=collate_fn)
        _vlab = "VAL-T"
    elif val_type == "val_pie":
        val_ds = PIEDataset(data_cfg["annotation_root"], split="val",
                            obs_len=data_cfg["obs_len"],
                            bbox_normalization=bbox_norm,
                            speed_normalization=speed_norm,
                            drop_first_frame=drop_ff,
                            **_geom,
                            sample_type=_st)
        val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"],
                                shuffle=False, num_workers=0, collate_fn=collate_fn)
        _vlab = "VAL  "
    else:
        val_loader = None
        _vlab = None
        print("\n[INFO] Nessun validation set attivo.")

    # Pre-carica il TEST set per l'evaluation
    test_ds = PIEDataset(data_cfg["annotation_root"], split="test",
                         obs_len=data_cfg["obs_len"],
                         bbox_normalization=bbox_norm,
                         speed_normalization=speed_norm,
                         drop_first_frame=drop_ff,
                         **_geom,
                         sample_type=_st)
    test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"],
                             shuffle=False, num_workers=0, collate_fn=collate_fn)

    if eval_test_every_epoch:
        print("[INFO] eval_test_every_epoch=True -> Il TEST set verra' valutato ad ogni epoca.")

    # -- Loss: decide anche la forma della testa ------------------------------
    #   training.loss:  bce | ce | mcel_bce | mcel_ce
    #     bce      -> 1 logit + BCEWithLogits pesata  (default storico per
    #                 TransformerModalityNet / BenchmarkSingleRNN)
    #     ce       -> 2 logit + CrossEntropyLoss(weight=[w_neg, w_pos])
    #     mcel_bce -> 1 logit + MCEL binaria
    #     mcel_ce  -> 2 logit + MCEL Eq. 10 (versione del paper)
    #   Se la chiave manca, si replica il comportamento originale.
    loss_name   = resolve_loss_name(train_cfg, cfg["model"]["name"])
    n_outputs   = num_outputs_for(loss_name)

    # -- Modello --------------------------------------------------------------
    model = build_model(cfg, num_outputs=n_outputs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Feature attive: bbox=True, "
          f"bbox_delta={getattr(model, 'use_bbox_delta', '?')}, "
          f"ego_speed={getattr(model, 'use_ego_speed', '?')} "
          f"-> input_dim={getattr(model, 'input_dim', '?')}")
    print(f"Parametri: {n_params:,}\n")

    # -- Loss / optim / scheduler --------------------------------------------
    cw_cfg = train_cfg.get("class_weights", None)
    if cw_cfg is not None:
        class_weights = torch.tensor([float(cw_cfg[0]), float(cw_cfg[1])],
                                     dtype=torch.float32, device=device)
        print(f"  Class weights (da config) -> neg={cw_cfg[0]}, pos={cw_cfg[1]}")
    else:
        class_weights = train_ds.get_class_weights().to(device)

    criterion, _loss_desc = build_criterion(loss_name, train_cfg,
                                            class_weights, device)
    print(f"  Loss: {_loss_desc}")
    if n_outputs == 2 and cfg["model"]["name"] == "TransformerModalityNet":
        print("        (testa del Transformer a 2 logit invece di 1)")

    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    wd = float(train_cfg.get("weight_decay", 0.0))
    lr = float(train_cfg["lr"])
    if opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    print(f"  Optimizer: {opt_name}  lr={lr}  weight_decay={wd}")

    sched_name = str(train_cfg.get("scheduler", "none")).lower()
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"], eta_min=1e-5)
    else:
        scheduler = None
        print("  Scheduler: nessuno")

    grad_clip_cfg = train_cfg.get("grad_clip", 1.0)
    if grad_clip_cfg is False or grad_clip_cfg is None or str(grad_clip_cfg).lower() in ("none", "false", "0"):
        grad_clip = None
        print("  Gradient Clipping: disattivato")
    else:
        grad_clip = float(grad_clip_cfg)
        print(f"  Gradient Clipping: max_norm={grad_clip}")

    # -- Resume ---------------------------------------------------------------
    start_epoch = 1
    best_val_f1 = 0.0
    best_test_f1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("val_metrics", {}).get("f1", 0.0) if ckpt.get("val_metrics") else 0.0
        print(f"Ripreso da epoch {start_epoch} (best_f1={best_val_f1:.4f})")

    # -- Training loop --------------------------------------------------------
    patience_counter = 0
    history = []
    epochs = train_cfg["epochs"]

    print("=== Training ===")
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_m, _ = run_epoch(model, train_loader, criterion, device,
                               optimizer=optimizer, grad_clip=grad_clip,
                               desc=f"Epoch {epoch}/{epochs} [train]")

        if val_loader is not None:
            val_m, _ = run_epoch(model, val_loader, criterion, device,
                                 optimizer=None,
                                 desc=f"Epoch {epoch}/{epochs} [{_vlab.strip()}]  ")
        else:
            val_m = None

        if eval_test_every_epoch:
            test_m, _ = run_epoch(model, test_loader, criterion, device,
                                  optimizer=None,
                                  desc=f"Epoch {epoch}/{epochs} [test] ")
        else:
            test_m = None

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0] if scheduler is not None else lr

        def _fmt(tag: str, m: Dict) -> str:
            s = (f"  {tag} loss={m['loss']:.4f} acc={m['acc']:.4f} b_acc={m['balanced_acc']:.4f} "
                 f"f1={m['f1']:.4f} auc={m['auc']:.4f} "
                 f"P={m['precision']:.4f} R={m['recall']:.4f}\n")
            # riga di diagnostica (Fase 1)
            s += (f"  {' ' * len(tag)} plain={m['loss_plain']:.4f} ece={m['ece']:.4f} "
                  f"mlm={m['mlm']:.2f} |logit|={m['logit_absmean']:.2f} "
                  f"max={m['logit_absmax']:.1f}")
            if "gradnorm" in m:
                s += (f" gnorm={m['gradnorm']:.2f} "
                      f"clip={100 * m['gradclip_frac']:.0f}%")
            return s + "\n"

        out_msg = f"Epoch {epoch:3d}/{epochs} ({elapsed:.0f}s) lr={lr_now:.2e}\n"
        out_msg += _fmt("TRAIN", train_m)
        if val_m is not None:
            out_msg += _fmt(_vlab, val_m)
        if test_m is not None:
            out_msg += _fmt("TEST ", test_m)
        print(out_msg.rstrip())

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}}
        if val_m is not None:
            row.update({f"val_{k}": v for k, v in val_m.items()})
        if test_m is not None:
            row.update({f"test_{k}": v for k, v in test_m.items()})
        history.append(row)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Gestione Checkpoint / Early stopping
        if val_m is not None:
            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                patience_counter = 0
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "val_metrics": val_m, "config": cfg},
                           out_dir / "best_model.pt")
                print(f"  ✓ Best model salvato (val_f1={best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= train_cfg.get("patience", 10**9):
                    print(f"\nEarly stopping (patience={train_cfg['patience']})")
                    break
        elif test_m is not None:
            if test_m["f1"] > best_test_f1:
                best_test_f1 = test_m["f1"]
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "val_metrics": test_m, "config": cfg},
                           out_dir / "best_model.pt")
                print(f"  ✓ Best model salvato (test_f1={best_test_f1:.4f})")

    # Salva sempre il modello finale dell'ultima epoca
    torch.save({"epoch": epochs, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_m or test_m, "config": cfg},
               out_dir / "final_model.pt")
    print("  Modello finale (ultima epoca) salvato come final_model.pt")

    # -- Evaluation finale del best model ------------------------------------
    print("\n=== Evaluation finale (best model) ===")
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    eval_splits = [("train", train_eval_loader)]
    if val_loader is not None:
        eval_splits.append((_vlab.strip().lower(), val_loader))
    eval_splits.append(("test", test_loader))

    final_metrics = {}
    predictions   = {}
    for split, loader in eval_splits:
        m, out = run_epoch(model, loader, criterion, device,
                           optimizer=None, desc=f"eval [{split}]")
        final_metrics[split] = m
        predictions[split]   = out
        print(f"\n[{split.upper()}] "
              f"acc={m['acc']:.4f} b_acc={m['balanced_acc']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f}")
        print(f"          plain_loss={m['loss_plain']:.4f} ece={m['ece']:.4f} "
              f"mlm={m['mlm']:.2f} |logit|={m['logit_absmean']:.2f}")

    with open(out_dir / "test_results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    with open(out_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nRisultati salvati in: {out_dir}")

    # -- Plot automatici ------------------------------------------------------
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== Generazione grafici ===")
    plot_metric_curves(history, plots_dir)
    plot_confusion_matrices(predictions, plots_dir)
    print(f"Grafici salvati in: {plots_dir}")


if __name__ == "__main__":
    main()