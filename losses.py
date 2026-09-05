"""
Loss functions e diagnostica di calibrazione
============================================

Implementa la Margin Cross-Entropy Loss (MCEL) di
    Yayla & Kumar, "MCEL: Margin-Based Cross-Entropy Loss for Error-Tolerant
    Quantized Neural Networks", arXiv:2603.05048

Formulazione del paper (Eq. 7, 9, 10):

    y~_k        = L * tanh(y^_k / L)            # smooth clamping   (Eq. 7)
    y~_i       <- y~_i - m                      # margine sul target (Eq. 9)
    loss        = -log softmax(y~^(m))_i        # MCEL              (Eq. 10)

Perche' serve l'Eq. 7 e non basta l'Eq. 5 (sottrazione secca del margine):
il softmax e' invariante per traslazione (Eq. 6), quindi senza un bound sui
logit la rete soddisfa il margine abbassando TUTTI i logit insieme, senza
aumentare la separazione relativa. Il tanh fissa il range a [-L, L] e rende
il margine m interpretabile come frazione del dynamic range:

    RLS = m / (2L)        (Relative Logit Separation, Eq. 8)

Il margine e' applicato SOLO dentro la loss: il forward del modello non
cambia, quindi a inferenza la rete non vede m e non puo' aggirarlo.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════
# MCEL — versione a K logit + softmax (fedele all'Eq. 10 del paper)
# ══════════════════════════════════════════════════════════════════════════
class MCEL(nn.Module):
    """Drop-in replacement di nn.CrossEntropyLoss.

    Args:
        margin:       m > 0. Griglia del paper: 1,2,4,...,128,192.
        L:            bound di saturazione del tanh. Il paper usa 100.
        logit_scale:  costante moltiplicativa sui logit grezzi, per portarli
                      nella regione lineare del tanh (dove L*tanh(z/L) ~= z).
                      Il paper scala i logit a magnitudine ~1 con L=100.
        warmup_steps: se > 0, il margine cresce linearmente da 0 a `margin`
                      nei primi N step di training. NON e' nel paper: serve
                      a convivere col gradient clipping (vedi note sotto).
        weight:       pesi per classe, come in nn.CrossEntropyLoss.

    Nota sul gradient clipping:
        alla prima iterazione il logit target vale y~_i - m ~= -m, quindi la
        loss iniziale e' enorme e la norma del gradiente sta molto sopra le
        soglie tipiche di clip (1.0). Il clip la riscala a ogni step e il
        passo effettivo diventa costante e piccolo: sembra che MCEL non
        converga, ma sta solo strozzando il clip. Il warmup evita il picco.
    """

    def __init__(
        self,
        margin: float = 16.0,
        L: float = 100.0,
        logit_scale: float = 1.0,
        warmup_steps: int = 0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        assert margin > 0, "margin deve essere > 0"
        assert L > 0, "L deve essere > 0"
        self.margin = float(margin)
        self.L = float(L)
        self.logit_scale = float(logit_scale)
        self.warmup_steps = int(warmup_steps)
        self.reduction = reduction
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    @property
    def rls(self) -> float:
        """Relative Logit Separation, Eq. 8: frazione di dynamic range."""
        return self.margin / (2.0 * self.L)

    def current_margin(self) -> float:
        if self.warmup_steps <= 0:
            return self.margin
        frac = min(1.0, float(self._step.item()) / self.warmup_steps)
        return self.margin * frac

    def clamp_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Eq. 7: y~ = L * tanh(scale * y^ / L)."""
        z = logits * self.logit_scale
        return self.L * torch.tanh(z / self.L)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        z = self.clamp_logits(logits)                              # Eq. 7

        m = self.current_margin()
        if torch.is_grad_enabled():
            self._step += 1

        if m > 0.0:
            one_hot = F.one_hot(target.long(), num_classes=z.size(1)).to(z.dtype)
            z = z - m * one_hot                                    # Eq. 9

        # F.cross_entropy usa log_softmax: stabile anche con m=192, L=100
        return F.cross_entropy(z, target.long(), weight=self.weight,
                               reduction=self.reduction)           # Eq. 10


# ══════════════════════════════════════════════════════════════════════════
# MCEL — versione a logit singolo (equivalente, per non toccare la testa)
# ══════════════════════════════════════════════════════════════════════════
class MCELBinary(nn.Module):
    """MCEL per un output singolo + sigmoide.

    Con logit singolo z (che E' gia' la differenza y_1 - y_0) e s = 2y - 1:

        loss = BCEWithLogits(L*tanh(z/L) - s*m,  y)

    richiede z > +m per la classe 1 e z < -m per la classe 0.
    Qui il clamping agisce sulla DIFFERENZA, quindi il dynamic range e' 2L
    e RLS = m/(2L) mantiene lo stesso significato.

    Il peso e' applicato per-sample (stile Keras), come nel train.py originale.
    """

    def __init__(
        self,
        margin: float = 16.0,
        L: float = 100.0,
        logit_scale: float = 1.0,
        warmup_steps: int = 0,
        w_neg: float = 1.0,
        w_pos: float = 1.0,
    ) -> None:
        super().__init__()
        assert margin > 0 and L > 0
        self.margin = float(margin)
        self.L = float(L)
        self.logit_scale = float(logit_scale)
        self.warmup_steps = int(warmup_steps)
        self.w_neg = float(w_neg)
        self.w_pos = float(w_pos)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    @property
    def rls(self) -> float:
        return self.margin / (2.0 * self.L)

    def current_margin(self) -> float:
        if self.warmup_steps <= 0:
            return self.margin
        frac = min(1.0, float(self._step.item()) / self.warmup_steps)
        return self.margin * frac

    def clamp_logits(self, logit: torch.Tensor) -> torch.Tensor:
        z = logit * self.logit_scale
        return self.L * torch.tanh(z / self.L)

    def forward(self, logit1: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float().view_as(logit1)
        z = self.clamp_logits(logit1)

        m = self.current_margin()
        if torch.is_grad_enabled():
            self._step += 1

        s = 2.0 * target - 1.0            # +1 classe 1, -1 classe 0
        z = z - s * m

        per = F.binary_cross_entropy_with_logits(z, target, reduction="none")
        w = torch.where(target > 0.5,
                        torch.full_like(per, self.w_pos),
                        torch.full_like(per, self.w_neg))
        return (per * w).mean()


# ══════════════════════════════════════════════════════════════════════════
# Costruzione della criterion dalla config
# ══════════════════════════════════════════════════════════════════════════
_LOSS_NUM_OUTPUTS = {
    "bce":      1,   # 1 logit + BCEWithLogits pesata per-sample
    "ce":       2,   # 2 logit + CrossEntropyLoss(weight=...)
    "mcel_bce": 1,   # 1 logit + MCEL binaria
    "mcel_ce":  2,   # 2 logit + MCEL Eq. 10 (versione del paper)
}


def resolve_loss_name(train_cfg: Dict, model_name: str) -> str:
    """Nome della loss dalla config, con fallback al comportamento storico.

    Se `training.loss` non e' specificato, si replica esattamente la logica
    originale di train.py: BCE per BenchmarkSingleRNN/TransformerModalityNet,
    CE per tutti gli altri. Cosi' i config vecchi girano invariati.
    """
    name = train_cfg.get("loss", None)
    if name is None:
        legacy_bce = model_name in ("BenchmarkSingleRNN", "TransformerModalityNet")
        return "bce" if legacy_bce else "ce"
    name = str(name).lower()
    if name not in _LOSS_NUM_OUTPUTS:
        raise ValueError(
            f"training.loss='{name}' non valido. "
            f"Valori ammessi: {sorted(_LOSS_NUM_OUTPUTS)}")
    return name


def num_outputs_for(loss_name: str) -> int:
    """Numero di logit richiesto dalla loss (1 o 2)."""
    return _LOSS_NUM_OUTPUTS[loss_name]


def build_criterion(loss_name: str, train_cfg: Dict,
                    class_weights: torch.Tensor,
                    device: torch.device) -> Tuple[nn.Module, str]:
    """Costruisce la criterion. Ritorna (criterion, descrizione stampabile)."""
    w_neg, w_pos = float(class_weights[0]), float(class_weights[1])

    margin = float(train_cfg.get("mcel_margin", 16.0))
    L = float(train_cfg.get("mcel_L", 100.0))
    logit_scale = float(train_cfg.get("mcel_logit_scale", 1.0))
    warmup = int(train_cfg.get("mcel_warmup_steps", 0))

    if loss_name == "bce":
        _bce_none = nn.BCEWithLogitsLoss(reduction="none")

        class _WeightedBCE(nn.Module):
            def forward(self, logit1, target):
                target = target.float().view_as(logit1)
                per = _bce_none(logit1, target)
                w = torch.where(target > 0.5,
                                torch.full_like(per, w_pos),
                                torch.full_like(per, w_neg))
                return (per * w).mean()

        crit = _WeightedBCE()
        desc = f"BCE per-sample pesata (neg={w_neg}, pos={w_pos}) — 1 logit"

    elif loss_name == "ce":
        crit = nn.CrossEntropyLoss(weight=class_weights.to(device))
        desc = f"CrossEntropy pesata (neg={w_neg}, pos={w_pos}) — 2 logit"

    elif loss_name == "mcel_bce":
        crit = MCELBinary(margin=margin, L=L, logit_scale=logit_scale,
                          warmup_steps=warmup, w_neg=w_neg, w_pos=w_pos)
        desc = (f"MCEL binaria m={margin} L={L} RLS={crit.rls:.3f} "
                f"scale={logit_scale} warmup={warmup} — 1 logit")

    elif loss_name == "mcel_ce":
        crit = MCEL(margin=margin, L=L, logit_scale=logit_scale,
                    warmup_steps=warmup, weight=class_weights.to(device))
        desc = (f"MCEL (Eq. 10) m={margin} L={L} RLS={crit.rls:.3f} "
                f"scale={logit_scale} warmup={warmup} — 2 logit")

    else:
        raise ValueError(f"loss sconosciuta: {loss_name}")

    return crit.to(device), desc


# ══════════════════════════════════════════════════════════════════════════
# Diagnostica — Fase 1: capire cosa fanno i logit
# ══════════════════════════════════════════════════════════════════════════
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> float:
    """ECE con binning uniforme sulla confidence.

    probs = P(classe 1). confidence = max(p, 1-p).
    In ogni bin si confronta la confidence media dichiarata con l'accuracy
    reale; ECE e' la media pesata dei |gap|. Piu' basso = meglio calibrato.
    Un modello over-confident sta tipicamente sopra 0.08-0.10.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.size == 0:
        return 0.0

    conf = np.maximum(probs, 1.0 - probs)
    preds = (probs >= 0.5).astype(labels.dtype)
    correct = (preds == labels).astype(np.float64)

    edges = np.linspace(0.5, 1.0, n_bins + 1)     # conf binaria vive in [0.5, 1]
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def logit_diagnostics(logits: np.ndarray) -> Dict[str, float]:
    """Statistiche sui logit GREZZI (pre-clamping, pre-margine).

    mlm           : Mean Logit Margin, Eq. 1 del paper. Media di
                    (logit piu' alto - secondo piu' alto). Nel caso binario a
                    1 logit e' |z|, che e' la stessa quantita'.
                    Serve a verificare che il margine stia agendo: nel paper
                    MCEL alza l'MLM di un fattore 3-30 rispetto a CE.
    logit_absmean : |logit| medio. Serve a tarare `mcel_logit_scale`:
                    il paper vuole magnitudini ~1 con L=100. Se qui leggi 3,
                    imposta logit_scale ~ 0.3.
    logit_absmax  : |logit| massimo, per vedere le code.
    """
    arr = np.asarray(logits, dtype=np.float64)
    if arr.size == 0:
        return {"mlm": 0.0, "logit_absmean": 0.0, "logit_absmax": 0.0}

    if arr.ndim == 1:
        mlm = float(np.abs(arr).mean())
    else:
        part = np.sort(arr, axis=1)
        mlm = float((part[:, -1] - part[:, -2]).mean())

    return {
        "mlm":           mlm,
        "logit_absmean": float(np.abs(arr).mean()),
        "logit_absmax":  float(np.abs(arr).max()),
    }


def plain_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """CE/BCE standard NON pesata sui logit grezzi.

    E' la metrica da confrontare fra run diverse: il valore restituito da
    MCEL non e' comparabile con quello della CE (scale diverse, floor > 0).
    Questa invece e' sempre la stessa quantita', qualunque loss usi in
    training, ed e' quella che nel tuo caso "sale mentre le metriche salgono".
    """
    t_logits = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    t_labels = torch.from_numpy(np.asarray(labels))
    if t_logits.numel() == 0:
        return 0.0
    if t_logits.dim() == 1:
        return float(F.binary_cross_entropy_with_logits(
            t_logits, t_labels.float()).item())
    return float(F.cross_entropy(t_logits, t_labels.long()).item())


def calibration_report(logits: np.ndarray, probs: np.ndarray,
                       labels: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    """Blocco completo di diagnostica per una epoca/split."""
    out = {"loss_plain": plain_loss(logits, labels),
           "ece": expected_calibration_error(probs, labels, n_bins=n_bins)}
    out.update(logit_diagnostics(logits))
    return out


# ══════════════════════════════════════════════════════════════════════════
# Fase 4 — temperature scaling post-hoc (Guo et al., 2017)
# ══════════════════════════════════════════════════════════════════════════
class TemperatureScaler(nn.Module):
    """Divide i logit per una singola T > 0 fittata sul validation set.

    Non cambia l'argmax, quindi accuracy / F1 / AUC restano IDENTICI:
    sistema solo le probabilita'. E' il fix diretto per la calibrazione,
    complementare a MCEL (che agisce in training).
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> float:
        return float(self.log_T.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_T.exp()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            max_iter: int = 100) -> "TemperatureScaler":
        logits = logits.detach().float()
        labels = labels.detach()
        binary = (logits.dim() == 1)
        opt = torch.optim.LBFGS([self.log_T], lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            z = self(logits)
            loss = (F.binary_cross_entropy_with_logits(z, labels.float())
                    if binary else F.cross_entropy(z, labels.long()))
            loss.backward()
            return loss

        opt.step(closure)
        return self
