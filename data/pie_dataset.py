"""
PIE Dataset — wrapper sull'interfaccia ufficiale
=================================================
Versione "geometry + ego (opzionale)" con MODALITA' BENCHMARK opzionale.

MODALITA' PREPROCESSING (flag benchmark_preproc):
  - False (default, la TUA pipeline):
        bbox       = coordinate normalizzate in [0,1]  (x/W, y/H)
        bbox_delta = differenza frame-by-frame delle bbox normalizzate
        ego_speed  = (obd+gps)/2 / 120
  - True (replica Kotseruba/PedestrianActionBenchmark, get_data_sequence):
        bbox       = DELTA-DAL-PRIMO-FRAME  (box[t] - box[0]), coordinate PIXEL grezze
        bbox_delta = frame-by-frame dei valori sopra (per compatibilita' interfaccia)
        ego_speed  = obd GREZZA (nessuna divisione), NON media con gps
        + la sequenza perde 1 frame (il frame 0 e' il riferimento) -> lunghezza obs_len-1

Nota: nel benchmark il "normalize" azzera il primo frame e restituisce
gli scarti da esso, quindi la finestra osservata scende da obs_len a obs_len-1.
Per mantenere l'allineamento, in modalita' benchmark TUTTE le feature
(bbox, bbox_delta, ego_speed) hanno lunghezza obs_len-1.

Protocollo temporale (identico nelle due modalita'):
  - min_track_size = obs_len + 60 = 76
  - TTE [30, 60] frame prima del crossing_point
  - overlap = 0.6 -> step = 6 frame
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

OBS_LEN = 16
TTE_MIN = 30   # frame (~1s)
TTE_MAX = 60   # frame (~2s)
IMG_W   = 1920.0
IMG_H   = 1080.0
SPEED_NORM = 120.0   # km/h, fattore di normalizzazione ego-speed (solo modalita' tua)


def get_pie_interface(pie_root: str):
    """Inizializza pie_data.py cercandolo in piu posizioni."""
    for p in [
        Path(pie_root) / "utilities",
        Path(__file__).parent,
    ]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    try:
        from pie_data import PIE
    except ImportError:
        raise ImportError(
            "pie_data.py non trovato.\n"
            "Copialo con: cp /path/to/PIE/utilities/pie_data.py data/"
        )

    pie_data_root = Path(pie_root) / "annotations"
    if not (pie_data_root / "annotations").exists():
        pie_data_root = Path(pie_root)

    print(f"PIE data root: {pie_data_root}")
    return PIE(data_path=str(pie_data_root))


def build_samples(
    pie_sequences: Dict,
    obs_len:  int   = OBS_LEN,
    overlap:  float = 0.6,
    tte_min:  int   = TTE_MIN,
    tte_max:  int   = TTE_MAX,
    bbox_normalization:  str = "raw",       # "raw" or "image_based"
    speed_normalization: str = "raw",       # "raw" or "divide"
    drop_first_frame:    bool = False,      # If True, drops first frame (len = obs_len-1)
) -> List[Dict]:
    """
    Costruisce i sample seguendo il protocollo temporale Kotseruba WACV 2021.
    Le normalizzazioni e lo slicing sono configurabili tramite parametri.
    """
    bboxes_all = pie_sequences["bbox"]
    intbin_all = pie_sequences.get("activities") or pie_sequences.get("intention_binary")
    obd_all    = pie_sequences.get("obd_speed", None)
    gps_all    = pie_sequences.get("gps_speed", None)
    pids_all   = pie_sequences.get("pid") or pie_sequences.get("ped_id")

    all_labels    = [int(intbin_all[i][0][0]) for i in range(len(bboxes_all))]
    n_cross_raw   = sum(l == 1 for l in all_labels)
    n_nocross_raw = sum(l == 0 for l in all_labels)
    n_total_raw   = len(all_labels)
    print(f"\n[INFO] Distribuzione tracks nel dataset raw:")
    print(f"  Crossing:     {n_cross_raw:4d}  ({100*n_cross_raw/n_total_raw:.1f}%)")
    print(f"  Non-crossing: {n_nocross_raw:4d}  ({100*n_nocross_raw/n_total_raw:.1f}%)")
    print(f"  Totale:       {n_total_raw:4d}")
    print(f"  Ratio NC/C:   {n_nocross_raw/max(n_cross_raw,1):.2f}:1")
    print(f"  ego-speed disponibile: "
          f"{'SI' if (obd_all is not None and gps_all is not None) else 'NO'}")
    print(f"  >>> Bbox Norm: {bbox_normalization} | Speed Norm: {speed_normalization} | Drop First: {drop_first_frame}")

    step = max(1, round(obs_len * (1.0 - overlap)))
    eff_len = obs_len - 1 if drop_first_frame else obs_len

    samples = []

    for i in range(len(bboxes_all)):
        track_bboxes = bboxes_all[i]
        T      = len(track_bboxes)
        label  = int(intbin_all[i][0][0])
        ped_id = pids_all[i][0][0] if pids_all[i] else f"ped_{i}"

        if T < obs_len + tte_min:
            continue

        start_idx = max(0, T - obs_len - tte_max)
        end_idx   = T - obs_len - tte_min
        if end_idx < 0 or end_idx < start_idx:
            continue

        obd_track = obd_all[i] if obd_all is not None else None
        gps_track = gps_all[i] if gps_all is not None else None

        for w_start in range(start_idx, end_idx + 1, step):
            w_end = w_start + obs_len

            obs_bboxes = np.array(track_bboxes[w_start:w_end], dtype=np.float32)
            if len(obs_bboxes) < obs_len:
                continue

            # --- PREPROCESSO BBOX ---
            obs_bboxes_norm = obs_bboxes.copy()
            if bbox_normalization == "image_based":
                obs_bboxes_norm[:, [0, 2]] /= IMG_W
                obs_bboxes_norm[:, [1, 3]] /= IMG_H

            # 1. Absolute position (normalization configured via bbox_normalization)
            if drop_first_frame:
                bbox_feat = obs_bboxes_norm[1:]
            else:
                bbox_feat = obs_bboxes_norm

            # 2. Relative displacement from first frame of window (always raw pixels)
            bbox_disp_full = obs_bboxes - obs_bboxes[0:1]
            if drop_first_frame:
                bbox_disp = bbox_disp_full[1:]
            else:
                bbox_disp = bbox_disp_full

            # 3. Delta frame-by-frame (always raw pixels)
            bbox_delta_full = np.zeros_like(obs_bboxes)
            bbox_delta_full[1:] = obs_bboxes[1:] - obs_bboxes[:-1]
            if drop_first_frame:
                bbox_delta = bbox_delta_full[1:]
            else:
                bbox_delta = bbox_delta_full

            # --- PREPROCESSO SPEED ---
            ego_speed_seq = np.zeros((eff_len, 1), dtype=np.float32)
            if speed_normalization == "raw":
                if obd_track is not None:
                    obd_slice = obd_track[w_start:w_end]
                    start_j = 1 if drop_first_frame else 0
                    for j in range(start_j, min(obs_len, len(obd_slice))):
                        ov = obd_slice[j][0] if isinstance(obd_slice[j], list) else obd_slice[j]
                        idx_out = j - 1 if drop_first_frame else j
                        ego_speed_seq[idx_out, 0] = float(ov)
            elif speed_normalization == "divide":
                if obd_track is not None and gps_track is not None:
                    obd_slice = obd_track[w_start:w_end]
                    gps_slice = gps_track[w_start:w_end]
                    start_j = 1 if drop_first_frame else 0
                    for j in range(start_j, min(obs_len, len(obd_slice))):
                        ov = obd_slice[j][0] if isinstance(obd_slice[j], list) else obd_slice[j]
                        gv = gps_slice[j][0] if isinstance(gps_slice[j], list) else gps_slice[j]
                        idx_out = j - 1 if drop_first_frame else j
                        ego_speed_seq[idx_out, 0] = (float(ov) + float(gv)) / (2.0 * SPEED_NORM)

            samples.append({
                "ped_id":            ped_id,
                "w_start":           w_start,
                "tte":               T - w_end,
                "bbox":              bbox_feat.astype(np.float32),
                "bbox_displacement": bbox_disp.astype(np.float32),
                "bbox_delta":        bbox_delta.astype(np.float32),
                "ego_speed":         ego_speed_seq.astype(np.float32),
                "label":             np.int64(label),
            })

    n_pos = sum(s["label"] == 1 for s in samples)
    n_neg = sum(s["label"] == 0 for s in samples)
    print(f"\n  tracks: {len(bboxes_all)}  ->  samples: {len(samples)} "
          f"(step={step}, overlap={overlap}, eff_len={eff_len})")
    print(f"  crossing: {n_pos}, non-crossing: {n_neg} "
          f"(ratio {n_pos/max(n_neg,1):.2f}:1)")
    return samples


class PIEDataset(Dataset):
    """PyTorch Dataset per PCIP su PIE with configurable normalization."""

    def __init__(
        self,
        pie_root:       str,
        split:          str,
        obs_len:        int  = OBS_LEN,
        min_track_size: int  = None,
        bbox_normalization:  str = "raw",
        speed_normalization: str = "raw",
        drop_first_frame:    bool = False,
        sample_type:    str = "all",
    ):
        assert split in ("train", "val", "test")
        print(f"\n=== PIEDataset [{split}] ===")

        if min_track_size is None:
            min_track_size = obs_len + TTE_MAX
        print(f"min_track_size: {min_track_size}  "
              f"(= obs_len {obs_len} + tte_max {TTE_MAX})")

        pie = get_pie_interface(pie_root)

        print(f"Generando sequenze [{split}]  (sample_type={sample_type})...")
        sequences = pie.generate_data_trajectory_sequence(
            split,
            fstride=1,
            sample_type=sample_type,
            seq_type="crossing",
            min_track_size=min_track_size,
            height_rng=[0, float("inf")],
            squarify_ratio=0,
            data_split_type="default",
        )

        self.samples = build_samples(
            sequences, obs_len,
            bbox_normalization=bbox_normalization,
            speed_normalization=speed_normalization,
            drop_first_frame=drop_first_frame,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        return {
            "bbox":              torch.from_numpy(s["bbox"]),
            "bbox_displacement": torch.from_numpy(s["bbox_displacement"]),
            "bbox_delta":        torch.from_numpy(s["bbox_delta"]),
            "ego_speed":         torch.from_numpy(s["ego_speed"]),
            "label":             torch.tensor(s["label"], dtype=torch.long),
            "ped_id":            s["ped_id"],
        }

    def get_class_weights(self) -> torch.Tensor:
        labels  = np.array([s["label"] for s in self.samples])
        n_pos   = (labels == 1).sum()
        n_neg   = (labels == 0).sum()
        n_total = len(labels)
        w_pos = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
        w_neg = n_total / (2.0 * n_neg) if n_neg > 0 else 1.0
        print(f"  Class weights — crossing: {w_pos:.3f}, "
              f"non-crossing: {w_neg:.3f}")
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)
