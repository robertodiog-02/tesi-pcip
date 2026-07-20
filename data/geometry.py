"""
Feature geometriche per la posizione del pedone
================================================

Due rappresentazioni alternative alla bbox assoluta, entrambe pensate per
rendere la posizione meno dipendente dal moto dell'ego-veicolo.

1) PDM — Position Decoupling Module (GTransPDM, eq. 2)
   Misura la posizione rispetto a due linee di riferimento che delimitano
   la ROI di attraversamento, piu' l'area ratio come proxy di profondita'.

2) Quasi-polare a tre poli (PCIP, sezione III-D — variante senza selezione)
   Riparametrizza la posizione in (cos theta, distanza) rispetto a tre poli
   sul bordo inferiore dell'immagine.

   NOTA — differenza voluta rispetto a PCIP: il paper seleziona dinamicamente
   il polo piu' vicino frame per frame. Questo introduce DISCONTINUITA': quando
   il polo cambia tra due frame consecutivi, (dist, theta) saltano di colpo
   senza che il pedone si sia mosso davvero. Qui calcoliamo le coordinate
   rispetto a TUTTI E TRE i poli (6 feature invece di 2): nessuna selezione,
   nessuna discontinuita', e il modello puo' imparare da solo a quale polo
   dare peso.

IMPORTANTE: entrambe le rappresentazioni vanno calcolate su coordinate in
PIXEL GREZZI, perche' linee di riferimento e poli sono definiti sul piano
immagine (1920x1080). La normalizzazione avviene dopo, internamente.
"""

from typing import Tuple

import numpy as np

IMG_W = 1920.0
IMG_H = 1080.0
IMG_DIAG = float(np.sqrt(IMG_W ** 2 + IMG_H ** 2))

PDM_ALPHA = 100.0        # fattore di scala dell'area ratio (GTransPDM: alpha=100)
PDM_R_CLIP = 500.0       # clamp di R, evita esplosioni su box minuscole/rumorose
EPS = 1e-6


# ─── utility ──────────────────────────────────────────────────────────────────

def bbox_anchor(bboxes: np.ndarray, anchor: str = "center") -> np.ndarray:
    """
    Estrae il punto di riferimento del pedone dalle bbox.

    Args:
        bboxes: [T, 4] in formato [x1, y1, x2, y2], PIXEL GREZZI
        anchor: "center"        -> centro della box (GTransPDM usa questo)
                "bottom_center" -> piedi, contatto col piano stradale
                "top_left"      -> angolo superiore sinistro
    Returns:
        [T, 2] coordinate (x, y)
    """
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    if anchor == "center":
        return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0], axis=-1)
    if anchor == "bottom_center":
        return np.stack([(x1 + x2) / 2.0, y2], axis=-1)
    if anchor == "top_left":
        return np.stack([x1, y1], axis=-1)
    raise ValueError(f"anchor sconosciuto: {anchor}")


def bbox_area(bboxes: np.ndarray) -> np.ndarray:
    """Area delle bbox: [T, 4] -> [T]"""
    w = np.clip(bboxes[:, 2] - bboxes[:, 0], 0.0, None)
    h = np.clip(bboxes[:, 3] - bboxes[:, 1], 0.0, None)
    return w * h


def compute_area_ratio(bboxes: np.ndarray, alpha: float = PDM_ALPHA) -> np.ndarray:
    """
    Area ratio come proxy di profondita' (GTransPDM eq. 2):
        R = (A_t / A_{t-1} - 1) * alpha

    Se il pedone si avvicina la box cresce (R > 0), se si allontana R < 0.
    Il primo frame non ha precedente -> 0.

    Returns: [T, 1]
    """
    A = bbox_area(bboxes)
    R = np.zeros(len(A), dtype=np.float32)
    R[1:] = (A[1:] / (A[:-1] + EPS) - 1.0) * alpha
    R = np.clip(R, -PDM_R_CLIP, PDM_R_CLIP)
    return R[:, None].astype(np.float32)


# ─── PDM (GTransPDM) ──────────────────────────────────────────────────────────

def _line_y_at_x(x: np.ndarray,
                 p0: Tuple[float, float],
                 p1: Tuple[float, float]) -> np.ndarray:
    """y della retta passante per p0 e p1, valutata in x (retta non verticale)."""
    (x0, y0), (x1, y1) = p0, p1
    slope = (y1 - y0) / ((x1 - x0) + EPS)
    return y0 + slope * (x - x0)


def compute_pdm(
    bboxes: np.ndarray,
    y_min: float,
    anchor: str = "center",
    use_area_ratio: bool = True,
    keep_absolute: bool = False,
    img_w: float = IMG_W,
    img_h: float = IMG_H,
) -> np.ndarray:
    """
    Position Decoupling Module (GTransPDM, eq. 2).

    La ROI di attraversamento e' delimitata da due linee di riferimento:
        A = (0, H)      ->  B = (W/2, Y_min)      [linea sinistra]
        B = (W/2, Y_min) ->  C = (W, H)           [linea destra]

    Per ogni frame si calcola la disparita' tra il pedone e la linea di
    riferimento (quella dal lato in cui si trova), poi se ne prendono le
    DIFFERENZE TEMPORALI — cosi' la feature descrive come il pedone si sta
    muovendo rispetto alla ROI, non dove si trova in assoluto.

        P_i = { dx_t - dx_{t-1},  dy_t - dy_{t-1},  R }

    NOTA su `keep_absolute`: GTransPDM deriva le disparita' e butta via i
    valori assoluti, quindi il PDM dice "mi sto avvicinando alla linea" ma non
    "quanto sono vicino". Con keep_absolute=True si tengono anche dx e dy
    assoluti (normalizzati), portando la feature da 3 a 5 componenti:
        { dx_t, dy_t, dx_t - dx_{t-1}, dy_t - dy_{t-1}, R }
    E' una deviazione dal paper, ma recupera l'informazione posizionale
    rispetto alla ROI. Ablabile.

    Args:
        bboxes: [T, 4] PIXEL GREZZI
        y_min:  vertice superiore B delle linee (minimo y del dataset)
        use_area_ratio: se includere R (proxy di profondita')
        keep_absolute: se includere anche dx, dy assoluti oltre alle differenze
    Returns:
        [T, F] con F = 2 (base) +1 (area_ratio) +2 (keep_absolute)
    """
    pts = bbox_anchor(bboxes, anchor)          # [T, 2]
    x, y = pts[:, 0], pts[:, 1]

    A = (0.0, img_h)
    B = (img_w / 2.0, float(y_min))
    C = (img_w, img_h)

    # Il pedone e' a sinistra o a destra del vertice B? La linea di riferimento
    # e' quella del suo lato.
    on_left = x <= B[0]
    y_line = np.where(
        on_left,
        _line_y_at_x(x, A, B),
        _line_y_at_x(x, B, C),
    )
    # x della linea alla quota y del pedone (disparita' orizzontale)
    slope_left = (B[1] - A[1]) / ((B[0] - A[0]) + EPS)
    slope_right = (C[1] - B[1]) / ((C[0] - B[0]) + EPS)
    x_line = np.where(
        on_left,
        A[0] + (y - A[1]) / (slope_left + EPS),
        B[0] + (y - B[1]) / (slope_right + EPS),
    )

    dx = x - x_line
    dy = y - y_line

    # Differenze temporali (il primo frame non ha precedente -> 0)
    ddx = np.zeros_like(dx)
    ddy = np.zeros_like(dy)
    ddx[1:] = dx[1:] - dx[:-1]
    ddy[1:] = dy[1:] - dy[:-1]

    # Normalizzazione: le disparita' sono in pixel, le portiamo su scala immagine
    ddx = ddx / img_w
    ddy = ddy / img_h

    feats = []
    if keep_absolute:
        # disparita' assolute, normalizzate su scala immagine
        feats.append((dx / img_w)[:, None])
        feats.append((dy / img_h)[:, None])
    feats.append(ddx[:, None])
    feats.append(ddy[:, None])
    if use_area_ratio:
        feats.append(compute_area_ratio(bboxes))

    return np.concatenate(feats, axis=-1).astype(np.float32)


# ─── Quasi-polare a tre poli (PCIP, variante senza selezione) ─────────────────

def compute_tripolar(
    bboxes: np.ndarray,
    anchor: str = "center",
    include_sin: bool = False,
    add_deltas: bool = False,
    img_w: float = IMG_W,
    img_h: float = IMG_H,
) -> np.ndarray:
    """
    Coordinate quasi-polari rispetto a TUTTI E TRE i poli (PCIP sez. III-D,
    variante senza selezione dinamica).

    Poli sul bordo inferiore dell'immagine:
        O1 = (0, H)      O2 = (W/2, H)      O3 = (W, H)

    Per ciascun polo si calcola il vettore polo->pedone e se ne estraggono:
        dist      : norma, normalizzata sulla diagonale immagine -> ~[0, 1]
        cos_theta : coseno dell'angolo con l'asse polare (orizzontale, verso
                    destra), gia' in [-1, 1]
        sin_theta : opzionale — cos da solo e' ambiguo sul verso verticale

    Args:
        bboxes: [T, 4] PIXEL GREZZI
        include_sin: aggiunge sin_theta per ogni polo (3 feature in piu')
        add_deltas: aggiunge le differenze temporali di tutte le feature
                    (raddoppia la dimensione)
    Returns:
        [T, F] con F = 6 base, 9 con sin, e il doppio con add_deltas
    """
    pts = bbox_anchor(bboxes, anchor)          # [T, 2]
    x, y = pts[:, 0], pts[:, 1]

    poles = [(0.0, img_h), (img_w / 2.0, img_h), (img_w, img_h)]

    feats = []
    for (ox, oy) in poles:
        vx = x - ox
        vy = y - oy
        dist = np.sqrt(vx ** 2 + vy ** 2)
        cos_t = vx / (dist + EPS)
        feats.append((dist / IMG_DIAG)[:, None])   # distanza normalizzata
        feats.append(cos_t[:, None])
        if include_sin:
            feats.append((vy / (dist + EPS))[:, None])

    out = np.concatenate(feats, axis=-1).astype(np.float32)

    if add_deltas:
        d = np.zeros_like(out)
        d[1:] = out[1:] - out[:-1]
        out = np.concatenate([out, d], axis=-1)

    return out


def tripolar_dim(include_sin: bool = False, add_deltas: bool = False) -> int:
    """Dimensione dell'output di compute_tripolar (per costruire il modello)."""
    d = 9 if include_sin else 6
    return d * 2 if add_deltas else d


def pdm_dim(use_area_ratio: bool = True, keep_absolute: bool = False) -> int:
    """Dimensione dell'output di compute_pdm."""
    return 2 + int(use_area_ratio) * 1 + int(keep_absolute) * 2


# ─── Displacement / delta su punto di riferimento configurabile ───────────────

def bbox_track_feature(
    bboxes: np.ndarray,
    mode: str = "corners",
) -> np.ndarray:
    """
    Estrae la rappresentazione della bbox su cui calcolare displacement e delta.

    Args:
        bboxes: [T, 4] formato [x1, y1, x2, y2]
        mode:
            "corners"       -> [T, 4] i due angoli, invariato (comportamento
                               storico: contiene implicitamente sia traslazione
                               sia cambio di scala)
            "center"        -> [T, 2] centro della box. E' la definizione di
                               GTransPDM per D_i e V_i: la traiettoria e' il
                               moto del centro, non degli angoli.
            "center_size"   -> [T, 4] (cx, cy, w, h): traslazione e scala
                               separate esplicitamente, invece che mescolate
                               nei due angoli.
            "bottom_center" -> [T, 2] piedi del pedone, punto di contatto col
                               piano stradale
    Returns:
        [T, D] con D = 4, 2, 4 o 2 secondo il mode
    """
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    if mode == "corners":
        return bboxes.astype(np.float32)
    if mode == "center":
        return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0], axis=-1).astype(np.float32)
    if mode == "center_size":
        return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0,
                         x2 - x1, y2 - y1], axis=-1).astype(np.float32)
    if mode == "bottom_center":
        return np.stack([(x1 + x2) / 2.0, y2], axis=-1).astype(np.float32)
    raise ValueError(f"mode sconosciuto: {mode}")


def track_feature_dim(mode: str = "corners") -> int:
    """Dimensione dell'output di bbox_track_feature."""
    return {"corners": 4, "center": 2, "center_size": 4, "bottom_center": 2}[mode]
