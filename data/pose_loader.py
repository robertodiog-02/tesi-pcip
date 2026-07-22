"""
Caricamento e allineamento delle pose HRNet per PIE
====================================================

I pickle (uno per set: pie_hrnet_poses_setXX.pkl) hanno struttura:

    { 'set05': { 'video_0001': { '<ped_id>': {
          'frames':    array [N]          numeri di frame del video,
          'keypoints': array [N, 17, 3]   (x, y, confidence) PIXEL GREZZI,
          'bbox':      array [N, 4],
    }}}}

ALLINEAMENTO: la finestra da 16 frame del dataset va mappata sugli indici
giusti della traccia pose tramite i NUMERI DI FRAME (estratti dai path
immagine di pie_data), non per posizione — le due tracce possono avere
lunghezze/offset diversi.

PREPROCESSING (GTransPDM, Skeleton Pose Encoder):
- Normalizzazione col top-left della bbox:  (x', y') = (x_k - x_btl, y_k - y_btl)
  La confidence s_t resta invariata.
- Il paper usa N=20 keypoint: i 17 COCO di AlphaPose + neck, hip e body
  center "calculated based on the average of adjacent points". Qui le pose
  vengono da HigherHRNet (17 COCO): i 3 punti extra sono ricostruiti allo
  stesso modo (media dei punti adiacenti), attivabili con un flag.
"""

import pickle
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# indici COCO-17
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

_FRAME_RE = re.compile(r"(\d+)\.\w+$")


def frame_number_from_path(path: str) -> int:
    """Estrae il numero di frame dal path immagine (es. '.../00532.png' -> 532)."""
    m = _FRAME_RE.search(str(path))
    if m is None:
        raise ValueError(f"Impossibile estrarre il frame number da: {path}")
    return int(m.group(1))


def load_pose_index(pose_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Carica tutti i pickle pie_hrnet_poses_set*.pkl e costruisce un indice
    globale ped_id -> {'frames': [N], 'keypoints': [N, 17, 3]}.

    I ped_id PIE ('5_1_1731') sono univoci tra set/video, quindi l'indice
    piatto e' sicuro.
    """
    pose_dir = Path(pose_dir)
    files = sorted(pose_dir.glob("pie_hrnet_poses_set*.pkl"))
    if not files:
        raise FileNotFoundError(
            f"Nessun pickle di pose trovato in {pose_dir} "
            f"(atteso pie_hrnet_poses_setXX.pkl)"
        )

    index: Dict[str, Dict[str, np.ndarray]] = {}
    for f in files:
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        for set_name, videos in data.items():
            for video_name, peds in videos.items():
                for ped_id, tr in peds.items():
                    index[ped_id] = {
                        "frames": np.asarray(tr["frames"]),
                        "keypoints": np.asarray(tr["keypoints"], dtype=np.float32),
                    }
    print(f"  Pose index: {len(files)} pickle, {len(index)} pedoni")
    return index


def extract_window_poses(
    pose_index: Dict[str, Dict[str, np.ndarray]],
    ped_id: str,
    window_frames: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Estrae i keypoint della finestra allineando per NUMERO DI FRAME.

    Args:
        pose_index: indice da load_pose_index
        ped_id: id del pedone (es. '5_1_1731')
        window_frames: [T] numeri di frame della finestra del dataset
    Returns:
        (keypoints [T, 17, 3] con zero-fill sui frame mancanti,
         n_missing: quanti frame non avevano posa)
    """
    T = len(window_frames)
    out = np.zeros((T, 17, 3), dtype=np.float32)

    tr = pose_index.get(ped_id)
    if tr is None:
        return out, T  # pedone senza pose: tutto zero

    frames = tr["frames"]
    kps = tr["keypoints"]

    # mapping frame_number -> indice nella traccia pose
    # (searchsorted: frames e' ordinato crescente)
    pos = np.searchsorted(frames, window_frames)
    n_missing = 0
    for t in range(T):
        p = pos[t]
        if p < len(frames) and frames[p] == window_frames[t]:
            out[t] = kps[p]
        else:
            n_missing += 1
    return out, n_missing


def normalize_keypoints(
    keypoints: np.ndarray,
    bboxes: np.ndarray,
    mode: str = "bbox_topleft",
) -> np.ndarray:
    """
    Normalizza le coordinate dei keypoint rispetto alla bbox del frame.

    Args:
        keypoints: [T, N, 3] (x, y, conf) PIXEL GREZZI
        bboxes:    [T, 4] [x1, y1, x2, y2] PIXEL GREZZI (annotazioni dataset)
        mode:
            "bbox_topleft"        : (x', y') = (x - x1, y - y1). E' la
                                    normalizzazione di GTransPDM ("normalized
                                    with its corresponding top-left bounding
                                    box coordinates"). La scala resta in pixel.
            "bbox_topleft_scaled" : come sopra, ma diviso anche per (w, h)
                                    della box -> coordinate in ~[0, 1],
                                    invarianti alla distanza del pedone.
                                    Deviazione dal paper, ablabile.
            "none"                : keypoint grezzi.
    Returns:
        [T, N, 3] normalizzati; la confidence non viene toccata.
        I keypoint a zero (frame mancanti) restano a zero.
    """
    out = keypoints.copy()
    if mode == "none":
        return out

    # maschera dei frame validi: un frame di pose tutto-zero resta zero
    valid = (keypoints[:, :, :2].sum(axis=(1, 2)) != 0)   # [T]

    x1 = bboxes[:, 0][:, None]
    y1 = bboxes[:, 1][:, None]
    out[:, :, 0] = np.where(valid[:, None], out[:, :, 0] - x1, 0.0)
    out[:, :, 1] = np.where(valid[:, None], out[:, :, 1] - y1, 0.0)

    if mode == "bbox_topleft_scaled":
        w = np.clip(bboxes[:, 2] - bboxes[:, 0], 1e-3, None)[:, None]
        h = np.clip(bboxes[:, 3] - bboxes[:, 1], 1e-3, None)[:, None]
        out[:, :, 0] = out[:, :, 0] / w
        out[:, :, 1] = out[:, :, 1] / h
    elif mode != "bbox_topleft":
        raise ValueError(f"skeleton_norm sconosciuto: {mode}")

    return out


def add_extra_joints(keypoints: np.ndarray, mode: str = "gtranspdm") -> np.ndarray:
    """
    Aggiunge giunti derivati come media dei punti adiacenti.

    mode="gtranspdm" — 3 punti extra (GTransPDM), [T,17,3] -> [T,20,3]:
        17: neck        = media delle spalle (5, 6)
        18: hip         = media delle anche (11, 12)
        19: body center = media di neck e hip

    mode="openpose18" — solo il collo (Dual-STGAT), [T,17,3] -> [T,18,3]:
        17: neck = media delle spalle (5, 6)

        Dual-STGAT usa 18 keypoint OpenPose, che includono il collo come
        giunto RILEVATO dalla rete. HigherHRNet produce COCO-17, che il collo
        non ce l'ha. Lo deriviamo come media delle spalle — stesso metodo che
        GTransPDM usa per i suoi punti extra. La posizione anatomica e gli
        archi coincidono con quelli di OpenPose; la differenza e' che qui il
        collo e' stimato, non osservato.

    mode="none" — nessuna aggiunta, [T,17,3] invariato.

    La confidence dei punti derivati e' la media di quella dei genitori.
    """
    if mode == "none":
        return keypoints

    neck = (keypoints[:, L_SHOULDER] + keypoints[:, R_SHOULDER]) / 2.0

    if mode == "openpose18":
        return np.concatenate([keypoints, neck[:, None, :]], axis=1)   # [T,18,3]

    if mode == "gtranspdm":
        hip = (keypoints[:, L_HIP] + keypoints[:, R_HIP]) / 2.0
        center = (neck + hip) / 2.0
        extra = np.stack([neck, hip, center], axis=1)                  # [T,3,3]
        return np.concatenate([keypoints, extra], axis=1)              # [T,20,3]

    raise ValueError(f"mode sconosciuto: {mode}")


def skeleton_n_joints(mode="gtranspdm") -> int:
    """
    Numero di giunti secondo la modalita'.
    Accetta anche il vecchio booleano per retrocompatibilita'.
    """
    if isinstance(mode, bool):
        mode = "gtranspdm" if mode else "none"
    return {"none": 17, "openpose18": 18, "gtranspdm": 20}[mode]

# ─── Preprocessing delle occlusioni (Dual-STGAT, sez. III-D.1) ───────────────

def interpolate_occlusions(
    keypoints: np.ndarray,
    max_gap: int = 2,
    interp_confidence: float = 0.5,
    conf_threshold: float = 0.0,
) -> Tuple[np.ndarray, dict]:
    """
    Gestione delle occlusioni come in Dual-STGAT:
      - occlusioni BREVI (buco <= max_gap frame): interpolazione lineare tra
        i frame validi prima e dopo
      - occlusioni PROLUNGATE (buco > max_gap): coordinate azzerate a (0, 0)

    L'interpolazione e' PER GIUNTO, non per frame intero: il detector puo'
    rilevare 15 giunti su 17 in un frame, quindi il buco riguarda il singolo
    keypoint. Questo copre anche il caso del frame completamente mancante.

    Un buco di 1-2 frame azzerato produce un salto a zero e ritorno, che e'
    un segnale spurio ben peggiore di una posizione stimata: per questo
    interpolare i buchi brevi aiuta.

    Args:
        keypoints: [T, N, 3] (x, y, confidence)
        max_gap: lunghezza massima del buco da interpolare
        interp_confidence: confidence assegnata ai punti interpolati.
            Il paper non lo specifica; un valore basso segnala al modello che
            il giunto e' stimato e non osservato.
        conf_threshold: un giunto e' considerato mancante se la confidence e'
            <= a questa soglia (oltre al caso coordinate esattamente (0,0))

    Returns:
        (keypoints processati [T, N, 3], statistiche)
    """
    kp = keypoints.copy()
    T, N, _ = kp.shape

    # Criterio di "mancante": la CONFIDENCE e' il segnale primario.
    # Le coordinate (0,0) contano solo se anche la confidence e' nulla —
    # un keypoint legittimamente sul bordo dell'immagine puo' avere x=0 con
    # confidence alta, e non va scambiato per mancante.
    zero_xy = (kp[:, :, 0] == 0) & (kp[:, :, 1] == 0)
    missing = (kp[:, :, 2] <= conf_threshold) | (zero_xy & (kp[:, :, 2] == 0))

    n_interp = n_zeroed = 0

    for j in range(N):
        miss_j = missing[:, j]
        if not miss_j.any():
            continue

        t = 0
        while t < T:
            if not miss_j[t]:
                t += 1
                continue

            # estensione del buco corrente
            start = t
            while t < T and miss_j[t]:
                t += 1
            end = t - 1                      # ultimo frame mancante
            gap_len = end - start + 1

            has_before = start > 0
            has_after = end < T - 1

            if gap_len <= max_gap and has_before and has_after:
                # ── occlusione BREVE: interpolazione lineare ──
                p0 = kp[start - 1, j, :2]
                p1 = kp[end + 1, j, :2]
                for k in range(gap_len):
                    a = (k + 1) / (gap_len + 1)
                    kp[start + k, j, :2] = (1 - a) * p0 + a * p1
                    kp[start + k, j, 2] = interp_confidence
                n_interp += gap_len
            else:
                # ── occlusione PROLUNGATA (o ai bordi): azzeramento ──
                kp[start:end + 1, j, :2] = 0.0
                kp[start:end + 1, j, 2] = 0.0
                n_zeroed += gap_len

    stats = {
        "n_missing": int(missing.sum()),
        "n_interpolated": n_interp,
        "n_zeroed": n_zeroed,
        "total": T * N,
    }
    return kp, stats


