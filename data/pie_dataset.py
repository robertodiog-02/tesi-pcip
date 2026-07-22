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

try:
    from .geometry import (compute_pdm, compute_tripolar, pdm_dim, tripolar_dim,
                           bbox_track_feature, track_feature_dim)
    from .windowing import iter_windows, window_step
    from .pose_loader import (load_pose_index, extract_window_poses,
                              normalize_keypoints, add_extra_joints,
                              frame_number_from_path, skeleton_n_joints,
                              interpolate_occlusions)
except ImportError:
    from geometry import (compute_pdm, compute_tripolar, pdm_dim, tripolar_dim,
                          bbox_track_feature, track_feature_dim)
    from windowing import iter_windows, window_step
    from pose_loader import (load_pose_index, extract_window_poses,
                             normalize_keypoints, add_extra_joints,
                             frame_number_from_path, skeleton_n_joints,
                             interpolate_occlusions)

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
    use_pdm:             bool = False,      # GTransPDM Position Decoupling Module
    use_polar:           bool = False,      # PCIP quasi-polare a tre poli
    pdm_use_area_ratio:  bool = True,
    pdm_keep_absolute:   bool = False,
    polar_include_sin:   bool = False,
    polar_add_deltas:    bool = False,
    geom_anchor:         str  = "center",
    displacement_type:   str  = "corners",  # corners|center|center_size|bottom_center
    delta_type:          str  = "corners",
    y_min:               float = None,      # vertice B delle linee PDM
    use_skeleton:        bool = False,
    pose_index:          Dict = None,       # da load_pose_index (richiesto se use_skeleton)
    skeleton_norm:       str  = "bbox_topleft",
    skeleton_joints_mode: str = "gtranspdm",   # none|openpose18|gtranspdm
    pose_interpolate:    bool = False,
    pose_max_gap:        int  = 2,
    pose_interp_confidence: float = 0.5,
    pose_conf_threshold: float = 0.0,
) -> List[Dict]:
    """
    Costruisce i sample seguendo il protocollo temporale Kotseruba WACV 2021.
    Le normalizzazioni e lo slicing sono configurabili tramite parametri.
    """
    bboxes_all = pie_sequences["bbox"]
    images_all = pie_sequences.get("image", None)
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
    if use_pdm or use_polar:
        print(f"  >>> Geometria: PDM={use_pdm} (area_ratio={pdm_use_area_ratio}, "
              f"keep_abs={pdm_keep_absolute}) | "
              f"Polar={use_polar} (sin={polar_include_sin}, deltas={polar_add_deltas}) | "
              f"anchor={geom_anchor}")
    print(f"  >>> Displacement: {displacement_type} ({track_feature_dim(displacement_type)}d) | "
          f"Delta: {delta_type} ({track_feature_dim(delta_type)}d)")
    if use_skeleton:
        assert pose_index is not None, "use_skeleton=True richiede pose_index"
        assert images_all is not None, (
            "use_skeleton richiede i path immagine nelle sequenze PIE "
            "(chiave 'image') per l'allineamento dei frame")
        _n_j = skeleton_n_joints(skeleton_joints_mode)
        _desc = {"none": "17 COCO puri",
                 "openpose18": "17 COCO + collo derivato (topologia OpenPose-18)",
                 "gtranspdm": "17 COCO + collo/anca/centro derivati (GTransPDM)"}
        print(f"  >>> Skeleton: norm={skeleton_norm}, joints={_n_j} "
              f"-> {_desc[skeleton_joints_mode]}")
        if pose_interpolate:
            print(f"  >>> Occlusioni: buchi <= {pose_max_gap} frame interpolati "
                  f"(conf={pose_interp_confidence}), oltre -> (0,0)")

    # Y_min per le linee di riferimento del PDM: minimo y su TUTTI i campioni
    # del dataset (GTransPDM: "Ymin is the minimum yt of all samples").
    if use_pdm and y_min is None:
        _ys = [np.asarray(tb, dtype=np.float32)[:, 1].min()
               for tb in bboxes_all if len(tb) > 0]
        y_min = float(np.min(_ys)) if _ys else 0.0
        print(f"  >>> PDM Y_min calcolato dal dataset: {y_min:.1f}")

    step = window_step(obs_len, overlap)
    eff_len = obs_len - 1 if drop_first_frame else obs_len

    samples = []
    _miss_total = [0, 0]   # [frame senza posa, frame totali]
    _occ_total = [0, 0, 0]  # [giunti interpolati, azzerati, totali]

    for i in range(len(bboxes_all)):
        track_bboxes = bboxes_all[i]
        T      = len(track_bboxes)
        label  = int(intbin_all[i][0][0])
        ped_id = pids_all[i][0][0] if pids_all[i] else f"ped_{i}"

        obd_track = obd_all[i] if obd_all is not None else None
        gps_track = gps_all[i] if gps_all is not None else None

        # numeri di frame della traccia (per l'allineamento delle pose)
        if use_skeleton:
            track_frames = np.array(
                [frame_number_from_path(im) for im in images_all[i]],
                dtype=np.int64)

        # Finestre dal protocollo condiviso (windowing.py): unica fonte di
        # verita', usata anche da viz_skeleton e dall'estrattore di feature.
        for w_start, w_end, _tte in iter_windows(
                T, obs_len=obs_len, overlap=overlap,
                tte_min=tte_min, tte_max=tte_max):

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

            # 2. Relative displacement from first frame of window (raw pixels).
            #    Il punto di riferimento e' configurabile: "corners" usa i due
            #    angoli (comportamento storico), "center" il centro della box
            #    (definizione di GTransPDM per D_i: la traiettoria e' il moto
            #    del centro), "center_size" separa traslazione e scala.
            disp_src = bbox_track_feature(obs_bboxes, displacement_type)
            bbox_disp_full = disp_src - disp_src[0:1]
            if drop_first_frame:
                bbox_disp = bbox_disp_full[1:]
            else:
                bbox_disp = bbox_disp_full

            # 3. Delta frame-by-frame (raw pixels), stesso discorso
            delta_src = bbox_track_feature(obs_bboxes, delta_type)
            bbox_delta_full = np.zeros_like(delta_src)
            bbox_delta_full[1:] = delta_src[1:] - delta_src[:-1]
            if drop_first_frame:
                bbox_delta = bbox_delta_full[1:]
            else:
                bbox_delta = bbox_delta_full

            # --- FEATURE GEOMETRICHE (su PIXEL GREZZI, prima di ogni norm) ---
            # PDM e polare sono definiti sul piano immagine (linee di
            # riferimento e poli in coordinate pixel), quindi vanno calcolati
            # da obs_bboxes non normalizzate. La normalizzazione avviene
            # dentro le funzioni stesse.
            if use_pdm:
                pdm_full = compute_pdm(obs_bboxes, y_min=y_min,
                                       anchor=geom_anchor,
                                       use_area_ratio=pdm_use_area_ratio,
                                       keep_absolute=pdm_keep_absolute)
                pdm_feat = pdm_full[1:] if drop_first_frame else pdm_full
            else:
                pdm_feat = np.zeros((eff_len, 1), dtype=np.float32)

            if use_skeleton:
                win_frames = track_frames[w_start:w_end]
                kp_raw, n_miss = extract_window_poses(
                    pose_index, ped_id, win_frames)
                # Preprocessing delle occlusioni PRIMA della normalizzazione:
                # l'interpolazione va fatta sulle coordinate pixel originali.
                if pose_interpolate:
                    kp_raw, _ost = interpolate_occlusions(
                        kp_raw, max_gap=pose_max_gap,
                        interp_confidence=pose_interp_confidence,
                        conf_threshold=pose_conf_threshold)
                    _occ_total[0] += _ost["n_interpolated"]
                    _occ_total[1] += _ost["n_zeroed"]
                    _occ_total[2] += _ost["total"]
                kp_norm = normalize_keypoints(kp_raw, obs_bboxes, skeleton_norm)
                kp_norm = add_extra_joints(kp_norm, skeleton_joints_mode)
                kp_feat = kp_norm[1:] if drop_first_frame else kp_norm
                _miss_total[0] += n_miss
                _miss_total[1] += len(win_frames)
            else:
                kp_feat = np.zeros((eff_len, 1, 3), dtype=np.float32)

            if use_polar:
                pol_full = compute_tripolar(obs_bboxes, anchor=geom_anchor,
                                            include_sin=polar_include_sin,
                                            add_deltas=polar_add_deltas)
                polar_feat = pol_full[1:] if drop_first_frame else pol_full
            else:
                polar_feat = np.zeros((eff_len, 1), dtype=np.float32)

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
                "pdm":               pdm_feat.astype(np.float32),
                "polar":             polar_feat.astype(np.float32),
                "keypoints":         kp_feat.astype(np.float32),
                "ego_speed":         ego_speed_seq.astype(np.float32),
                "label":             np.int64(label),
            })

    n_pos = sum(s["label"] == 1 for s in samples)
    n_neg = sum(s["label"] == 0 for s in samples)
    print(f"\n  tracks: {len(bboxes_all)}  ->  samples: {len(samples)} "
          f"(step={step}, overlap={overlap}, eff_len={eff_len})")
    print(f"  crossing: {n_pos}, non-crossing: {n_neg} "
          f"(ratio {n_pos/max(n_neg,1):.2f}:1)")
    if use_skeleton and _miss_total[1] > 0:
        pct = 100.0 * _miss_total[0] / _miss_total[1]
        print(f"  skeleton: {_miss_total[0]}/{_miss_total[1]} frame senza posa "
              f"({pct:.1f}%) -> zero-fill")
    if pose_interpolate and _occ_total[2] > 0:
        pi = 100.0 * _occ_total[0] / _occ_total[2]
        pz = 100.0 * _occ_total[1] / _occ_total[2]
        print(f"  occlusioni: {_occ_total[0]} giunti interpolati ({pi:.2f}%), "
              f"{_occ_total[1]} azzerati ({pz:.2f}%) su {_occ_total[2]} totali")
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
        use_pdm:             bool = False,
        use_polar:           bool = False,
        pdm_use_area_ratio:  bool = True,
        pdm_keep_absolute:   bool = False,
        polar_include_sin:   bool = False,
        polar_add_deltas:    bool = False,
        geom_anchor:         str  = "center",
        displacement_type:   str  = "corners",
        delta_type:          str  = "corners",
        use_skeleton:        bool = False,
        pose_dir:            str  = "data/poses",
        skeleton_norm:       str  = "bbox_topleft",
        skeleton_joints_mode: str = "gtranspdm",
        pose_interpolate:    bool = False,
        pose_max_gap:        int  = 2,
        pose_interp_confidence: float = 0.5,
        pose_conf_threshold: float = 0.0,
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

        pose_index = None
        if use_skeleton:
            print(f"Caricamento pose da {pose_dir}...")
            pose_index = load_pose_index(pose_dir)

        self.samples = build_samples(
            sequences, obs_len,
            bbox_normalization=bbox_normalization,
            speed_normalization=speed_normalization,
            drop_first_frame=drop_first_frame,
            use_pdm=use_pdm,
            use_polar=use_polar,
            pdm_use_area_ratio=pdm_use_area_ratio,
            pdm_keep_absolute=pdm_keep_absolute,
            polar_include_sin=polar_include_sin,
            polar_add_deltas=polar_add_deltas,
            geom_anchor=geom_anchor,
            displacement_type=displacement_type,
            delta_type=delta_type,
            use_skeleton=use_skeleton,
            pose_index=pose_index,
            skeleton_norm=skeleton_norm,
            skeleton_joints_mode=skeleton_joints_mode,
            pose_interpolate=pose_interpolate,
            pose_max_gap=pose_max_gap,
            pose_interp_confidence=pose_interp_confidence,
            pose_conf_threshold=pose_conf_threshold,
        )

        # dimensioni effettive delle feature (servono a costruire il modello)
        self.pdm_dim = (pdm_dim(pdm_use_area_ratio, pdm_keep_absolute)
                        if use_pdm else 0)
        self.polar_dim = (tripolar_dim(polar_include_sin, polar_add_deltas)
                          if use_polar else 0)
        self.displacement_dim = track_feature_dim(displacement_type)
        self.delta_dim = track_feature_dim(delta_type)
        self.skeleton_joints = (skeleton_n_joints(skeleton_joints_mode)
                                if use_skeleton else 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        return {
            "bbox":              torch.from_numpy(s["bbox"]),
            "bbox_displacement": torch.from_numpy(s["bbox_displacement"]),
            "bbox_delta":        torch.from_numpy(s["bbox_delta"]),
            "pdm":               torch.from_numpy(s["pdm"]),
            "polar":             torch.from_numpy(s["polar"]),
            "keypoints":         torch.from_numpy(s["keypoints"]),
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
