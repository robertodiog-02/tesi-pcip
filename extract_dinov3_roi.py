"""
Estrazione feature visive DINOv3 per PIE — CLS globale + ROI Align multi-scala
==============================================================================

Per ogni frame che compare in almeno una finestra del dataset, esegue UN SOLO
forward DINOv3 e da quello ricava:

  1. CLS token           -> contesto globale della scena  [hidden_dim]
  2. ROI Align 1.0x      -> crop stretto sul pedone       [S, S, hidden_dim]
  3. ROI Align 1.5x      -> local context (alla Kotseruba)
  4. ROI Align 2.0x      -> extended context

Il CLS e' condiviso da tutti i pedoni del frame; i ROI Align sono per pedone.
Se cinque pedoni compaiono nello stesso frame, DINOv3 gira UNA volta sola.

Questo abilita le quattro configurazioni di ablation:
    A: cinematica + CLS                     (nessun focus locale)
    B: cinematica + CLS + ROI 1.0x          (tight)
    C: cinematica + CLS + ROI 1.5x          (local context)
    D: cinematica + CLS + ROI 2.0x          (extended)

FRAME NECESSARI: solo quelli che finiscono davvero in una finestra del
dataset, calcolati con windowing.py — lo STESSO protocollo di build_samples
(TTE in [30,60], obs_len 16, overlap 0.6). Nessun margine.

SCALE DEI BBOX: l'espansione 1.5x/2.0x segue l'algoritmo del benchmark PIE
(jitter_bbox legato al lato corto + squarify), lo stesso usato da Kotseruba,
cosi' il crop e' quadrato e non deformato.

OUTPUT: un .h5 per set + un indice pickle per la navigazione. L'indice mappa
    (set, video, ped_id, frame) -> {'h5', 'row_roi'}
    (set, video, frame)         -> {'h5', 'row_cls'}

Uso:
    # Mac (mps)
    python extract_dinov3_roi.py --pie-root ~/Desktop/PIE \\
        --video-root ~/Desktop/PIE/annotations/PIE_clips \\
        --out-dir features/dinov3 --sets set01 set02 set03

    # RTX (cuda)
    python extract_dinov3_roi.py --pie-root /mnt/c/Users/gabri/Desktop/PIE \\
        --video-root /mnt/c/Users/gabri/Desktop/pie_video \\
        --out-dir features/dinov3 --sets set04 set05 set06
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Espansione dei bbox — algoritmo del benchmark PIE (Kotseruba/Rasouli)
# ═══════════════════════════════════════════════════════════════════════════

def jitter_bbox_enlarge(bbox, ratio, img_w, img_h):
    """
    Espande il bbox legando l'espansione al LATO CORTO, cosi' un pedone alto
    e stretto non viene allargato a dismisura in verticale.
    ratio=0.5 -> +50% totale (il benchmark usa 1.5 come "enlarge ratio",
    che nella loro implementazione corrisponde a questo jitter).
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    w_change, h_change = w * ratio, h * ratio
    if w_change < h_change:
        h_change = w_change
    else:
        w_change = h_change
    x1 -= w_change / 2.0
    y1 -= h_change / 2.0
    x2 += w_change / 2.0
    y2 += h_change / 2.0
    return [max(0.0, x1), max(0.0, y1),
            min(img_w - 1.0, x2), min(img_h - 1.0, y2)]


def squarify(bbox, img_w):
    """Rende il bbox quadrato allargando la LARGHEZZA (l'altezza resta)."""
    x1, y1, x2, y2 = bbox
    width, height = abs(x2 - x1), abs(y2 - y1)
    dw = height - width
    x1 -= dw / 2.0
    x2 += dw / 2.0
    if x1 < 0:
        x1 = 0
    if x2 > img_w:
        x1 = max(0.0, x1 - (x2 - img_w))
        x2 = img_w
    return [x1, y1, x2, y2]


def scale_bbox(bbox, scale, img_w, img_h, do_squarify=True):
    """
    bbox espanso alla scala richiesta.
      scale=1.0 -> bbox originale (solo squarify + clamp)
      scale=1.5 -> +50%, scale=2.0 -> +100%
    """
    out = list(bbox)
    if scale > 1.0:
        out = jitter_bbox_enlarge(out, scale - 1.0, img_w, img_h)
    if do_squarify:
        out = squarify(out, img_w)
    x1, y1, x2, y2 = out
    return [max(0.0, x1), max(0.0, y1), min(img_w, x2), min(img_h, y2)]


# ═══════════════════════════════════════════════════════════════════════════
# Raccolta dei frame necessari (protocollo esatto del dataset)
# ═══════════════════════════════════════════════════════════════════════════

def collect_needed(pie_root, splits, sets_filter, obs_len, min_track_size,
                   sample_type="all"):
    """
    Percorre le sequenze PIE e raccoglie, per ogni video, i frame che
    compaiono in almeno una finestra valida, con i bbox dei pedoni.

    Returns:
        needed: {(set, video): {frame_id: [(ped_id, bbox), ...]}}
    """
    for p in [Path(pie_root) / "utilities", Path("data"), Path(".")]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from pie_data import PIE
    from windowing import iter_windows

    root = Path(pie_root) / "annotations"
    if not (root / "annotations").exists():
        root = Path(pie_root)
    imdb = PIE(data_path=str(root))

    opts = dict(fstride=1, sample_type=sample_type, data_split_type="default",
                seq_type="crossing", min_track_size=min_track_size,
                height_rng=[0, float("inf")], squarify_ratio=0)

    needed = defaultdict(lambda: defaultdict(list))
    n_tracks = n_wins = 0

    for split in splits:
        print(f"\n[{split}] generazione sequenze...")
        seq = imdb.generate_data_trajectory_sequence(split, **opts)
        pids = seq.get("pid") or seq.get("ped_id")

        for i, boxes in enumerate(seq["bbox"]):
            paths = seq["image"][i]
            ped_id = pids[i][0][0]

            # set/video/frame dal path immagine
            parts = Path(str(paths[0])).parts
            try:
                sid = [x for x in parts if x.startswith("set")][-1]
                vid = [x for x in parts if x.startswith("video_")][-1]
            except IndexError:
                continue
            if sets_filter and sid not in sets_filter:
                continue

            wins = list(iter_windows(len(boxes), obs_len=obs_len))
            if not wins:
                continue
            n_tracks += 1
            n_wins += len(wins)

            # unione dei frame coperti dalle finestre di questa traccia
            covered = set()
            for w_start, w_end, _ in wins:
                covered.update(range(w_start, w_end))

            for t in sorted(covered):
                fid = int(Path(str(paths[t])).stem)
                needed[(sid, vid)][fid].append((ped_id, list(map(float, boxes[t]))))

    n_frames = sum(len(v) for v in needed.values())
    n_rois = sum(len(pl) for v in needed.values() for pl in v.values())
    print(f"\n{'='*62}")
    print(f"Tracce: {n_tracks} | finestre: {n_wins}")
    print(f"Frame UNICI da processare: {n_frames} in {len(needed)} video")
    print(f"ROI totali (pedone x frame): {n_rois}")
    if n_frames:
        print(f"Pedoni per frame (media): {n_rois / n_frames:.2f} "
              f"-> risparmio {100*(1 - n_frames/max(n_rois,1)):.0f}% di forward")
    print(f"{'='*62}")
    return needed


# ═══════════════════════════════════════════════════════════════════════════
# ROI Align sulla griglia di patch
# ═══════════════════════════════════════════════════════════════════════════

def roi_align_patches(patch_grid, boxes, img_w, img_h, output_size=3):
    """
    ROI Align sulla griglia di patch di DINOv3.

    Args:
        patch_grid: tensore [1, C, Gh, Gw] — le patch del frame
        boxes:      [K, 4] bbox in coordinate IMMAGINE ORIGINALE (px)
        img_w/h:    dimensioni dell'immagine originale
        output_size: lato della griglia di output (3 -> 3x3 celle per ROI)

    Returns:
        [K, output_size, output_size, C]

    La bbox viene riscalata dalle coordinate immagine a quelle della griglia
    di patch; spatial_scale fa esattamente questo. L'interpolazione bilineare
    di roi_align gestisce le regioni frazionarie senza quantizzazione.
    """
    import torch
    from torchvision.ops import roi_align

    _, C, Gh, Gw = patch_grid.shape
    # una patch della griglia copre img_w/Gw px in orizzontale
    # (usiamo la scala orizzontale; con aspect ratio preservato coincide)
    scale_x = Gw / float(img_w)
    scale_y = Gh / float(img_h)

    b = torch.as_tensor(boxes, dtype=torch.float32, device=patch_grid.device)
    # roi_align accetta un solo spatial_scale: riscaliamo noi i box
    b_scaled = torch.stack([b[:, 0] * scale_x, b[:, 1] * scale_y,
                            b[:, 2] * scale_x, b[:, 3] * scale_y], dim=1)
    idx = torch.zeros((b.shape[0], 1), dtype=torch.float32, device=b.device)
    rois = torch.cat([idx, b_scaled], dim=1)          # [K, 5]

    out = roi_align(patch_grid, rois,
                    output_size=(output_size, output_size),
                    spatial_scale=1.0, sampling_ratio=2, aligned=True)
    return out.permute(0, 2, 3, 1).contiguous()        # [K, S, S, C]


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="DINOv3: CLS globale + ROI Align multi-scala per PIE")
    ap.add_argument("--pie-root", required=True,
                    help="root PIE (contiene annotations/)")
    ap.add_argument("--video-root", required=True,
                    help="cartella con setXX/video_XXXX.mp4")
    ap.add_argument("--out-dir", default="features/dinov3")
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--sets", nargs="+", default=None,
                    help="set da processare (es. set01 set02). Default: tutti")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--h", type=int, default=544, help="altezza input (mult. di 16)")
    ap.add_argument("--w", type=int, default=960, help="larghezza input (mult. di 16)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--obs-len", type=int, default=16)
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.5, 2.0],
                    help="scale di espansione dei bbox per il ROI Align")
    ap.add_argument("--roi-output-sizes", nargs="+", type=int, default=[1, 3],
                    help="lati delle griglie ROI Align da salvare. "
                         "1 -> vettore medio (leggero), 3 -> 3x3 (struttura "
                         "grossolana), 7 -> 7x7 (pesante: 5.4x il 3x3, e su "
                         "pedoni piccoli sovracampiona). Il forward DINOv3 e' "
                         "condiviso, quindi salvarne piu' di uno costa poco tempo.")
    ap.add_argument("--no-squarify", action="store_true",
                    help="non rendere quadrati i bbox espansi")
    ap.add_argument("--fp16", action="store_true",
                    help="salva le feature in float16 (meta' spazio)")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--seek", action="store_true",
                    help="usa il seek diretto invece della lettura sequenziale")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import torch
    import cv2
    import h5py
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── device ──
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    sets_filter = set(args.sets) if args.sets else None
    if sets_filter:
        print(f"Set selezionati: {sorted(sets_filter)}")

    # ── 1) frame necessari ──
    needed = collect_needed(args.pie_root, args.splits, sets_filter,
                            args.obs_len, args.obs_len + 60)
    if not needed:
        print("Nessun frame da processare.")
        return

    # ── 2) modello ──
    print(f"\nCarico {args.model} ...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    patch = model.config.patch_size
    nreg = getattr(model.config, "num_register_tokens", 0)
    hdim = model.config.hidden_size
    Gh, Gw = args.h // patch, args.w // patch
    sizes = sorted(set(args.roi_output_sizes))
    print(f"patch={patch} reg={nreg} dim={hdim} | griglia {Gh}x{Gw}={Gh*Gw}")
    print(f"ROI Align: sizes={sizes} x scale={args.scales} "
          f"-> {len(sizes)*len(args.scales)} tensori per ROI")
    kb = sum(S * S * hdim * (2 if args.fp16 else 4) for S in sizes) \
        * len(args.scales) / 1024.0
    print(f"Spazio per ROI: {kb:.0f} KB "
          f"({'fp16' if args.fp16 else 'fp32'})")

    dt = "float16" if args.fp16 else "float32"
    idx_path = out_dir / "index.pkl"
    index = {}
    if idx_path.exists() and not args.overwrite:
        with open(idx_path, "rb") as f:
            index = pickle.load(f)
        print(f"Indice esistente: {len(index)} voci")

    # ── 3) per video ──
    videos = sorted(needed.keys())
    for vi, (sid, vid) in enumerate(videos, 1):
        frame_map = needed[(sid, vid)]
        fids = sorted(frame_map.keys())
        h5path = out_dir / f"{sid}_{vid}.h5"

        if h5path.exists() and not args.overwrite:
            try:
                with h5py.File(h5path, "r") as hf:
                    if hf["cls"].shape[0] == len(fids):
                        print(f"[{vi}/{len(videos)}] {sid}/{vid}: gia' fatto, skip")
                        continue
            except Exception:
                print(f"[{vi}/{len(videos)}] {sid}/{vid}: h5 corrotto, rifaccio")

        mp4 = Path(args.video_root) / sid / f"{vid}.mp4"
        if not mp4.exists():
            print(f"[{vi}/{len(videos)}] manca {mp4}, salto")
            continue

        n_roi = sum(len(frame_map[f]) for f in fids)
        print(f"[{vi}/{len(videos)}] {sid}/{vid}: {len(fids)} frame, {n_roi} ROI")

        # ── lettura frame ──
        cap = cv2.VideoCapture(str(mp4))
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wanted = set(fids)
        frames_pil, valid = [], []

        if args.seek:
            for fid in fids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ok, fr = cap.read()
                if not ok:
                    continue
                frames_pil.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                valid.append(fid)
        else:
            # lettura sequenziale: piu' veloce e affidabile del seek su mp4
            k = 0
            last = max(fids)
            while k <= last:
                ok, fr = cap.read()
                if not ok:
                    break
                if k in wanted:
                    frames_pil.append(
                        Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                    valid.append(k)
                k += 1
        cap.release()

        if not frames_pil:
            print("   nessun frame letto, salto")
            continue
        if len(valid) < len(fids):
            print(f"   letti {len(valid)}/{len(fids)} frame")

        # ── righe ROI: (frame_row, ped_id) ──
        roi_rows = []
        for r, fid in enumerate(valid):
            for ped_id, box in frame_map[fid]:
                roi_rows.append((r, fid, ped_id, box))

        n_f, n_r = len(valid), len(roi_rows)
        with h5py.File(h5path, "w") as hf:
            hf.attrs.update(dict(set=sid, video=vid, h=args.h, w=args.w,
                                 patch=patch, grid_h=Gh, grid_w=Gw,
                                 hidden_dim=hdim,
                                 roi_sizes=np.array(sizes, dtype=np.int32),
                                 scales=np.array(args.scales, dtype=np.float32),
                                 img_w=img_w, img_h=img_h,
                                 squarify=not args.no_squarify,
                                 obs_len=args.obs_len))
            cls_ds = hf.create_dataset("cls", (n_f, hdim), dtype=dt)
            hf.create_dataset("frame_ids", data=np.array(valid, dtype=np.int32))
            hf.create_dataset("roi_frame_row",
                              data=np.array([r[0] for r in roi_rows], dtype=np.int32))
            hf.create_dataset("roi_frame_id",
                              data=np.array([r[1] for r in roi_rows], dtype=np.int32))
            hf.create_dataset("roi_ped_id",
                              data=np.array([str(r[2]) for r in roi_rows],
                                            dtype=h5py.string_dtype()))
            hf.create_dataset("roi_bbox",
                              data=np.array([r[3] for r in roi_rows], dtype=np.float32))

            # un dataset per (scala, output_size): roi_1.5x_s3 ecc.
            roi_ds = {}
            for sc in args.scales:
                for S in sizes:
                    roi_ds[(sc, S)] = hf.create_dataset(
                        f"roi_{sc:g}x_s{S}", (n_r, S, S, hdim), dtype=dt,
                        chunks=(min(32, max(1, n_r)), S, S, hdim),
                        compression="gzip", compression_opts=1)

            # ── forward + ROI Align ──
            import torch as _t
            roi_ptr = 0
            with _t.inference_mode():
                for start in range(0, n_f, args.batch):
                    chunk = frames_pil[start:start + args.batch]
                    inp = processor(images=chunk, return_tensors="pt",
                                    size={"height": args.h, "width": args.w},
                                    do_center_crop=False).to(device)
                    out = model(**inp)

                    cls = out.pooler_output.float().cpu().numpy()
                    cls_ds[start:start + len(chunk)] = (
                        cls.astype(np.float16) if args.fp16 else cls)

                    # griglia di patch: [B, Gh, Gw, C] -> [B, C, Gh, Gw]
                    lhs = out.last_hidden_state[:, 1 + nreg:, :]
                    grid = lhs.unflatten(1, (Gh, Gw)).permute(0, 3, 1, 2).float()

                    for bi in range(len(chunk)):
                        frow = start + bi
                        peds = frame_map[valid[frow]]
                        if not peds:
                            continue
                        for sc in args.scales:
                            # i box espansi si calcolano una volta per scala,
                            # poi si riusano per tutte le output_size
                            boxes = [scale_bbox(b, sc, img_w, img_h,
                                                do_squarify=not args.no_squarify)
                                     for _, b in peds]
                            for S in sizes:
                                feats = roi_align_patches(
                                    grid[bi:bi + 1], boxes, img_w, img_h, S)
                                f = feats.cpu().numpy()
                                if args.fp16:
                                    f = f.astype(np.float16)
                                roi_ds[(sc, S)][roi_ptr:roi_ptr + len(peds)] = f
                        roi_ptr += len(peds)

        # ── indice ──
        for r, fid in enumerate(valid):
            index[(sid, vid, fid)] = {"h5": str(h5path), "row_cls": r}
        for j, (_, fid, ped_id, _) in enumerate(roi_rows):
            index[(sid, vid, ped_id, fid)] = {"h5": str(h5path), "row_roi": j}
        with open(idx_path, "wb") as f:
            pickle.dump(index, f)

    print(f"\nFatto. {len(index)} voci nell'indice.")
    print(f"Indice: {idx_path}")
    print(f"Feature: {out_dir}/")
    print("\nPer navigare i dati usa dinov3_reader.py")


if __name__ == "__main__":
    main()
