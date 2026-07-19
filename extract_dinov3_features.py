"""
Estrazione feature DINOv3 (CLS + patch complete) dai .mp4 di PIE.
================================================================
Pipeline:
  1. Genera le sequenze PIE (train+val+test) e raccoglie l'insieme UNICO
     di frame (set, video, frame) che compaiono in almeno una finestra.
  2. Per ogni video apre l'mp4 UNA volta, estrae solo i frame necessari,
     li passa a DINOv3 e salva CLS (pooler) + griglia di patch.
  3. Salva su disco in formato .h5 (uno per set) + un indice pickle.

Requisiti: torch, transformers>=4.56, opencv-python, h5py, accesso HF a DINOv3.

Uso:
  python extract_dinov3_features.py \
      --pie-root /Users/robertodioguardi/Desktop/PIE \
      --video-root /Users/robertodioguardi/Desktop/PIE/annotations/PIE_clips \
      --out-dir features/dinov3 \
      --h 544 --w 960 --batch 8 --model facebook/dinov3-vitb16-pretrain-lvd1689m

Note:
  - --video-root deve contenere set01/video_0001.mp4 ...
  - --h --w multipli di 16. 544x960 -> griglia 34x60 = 2040 patch.
  - le patch complete sono PESANTI: ~6 MB/frame in fp32 a 544x960.
    Usa --save-patches per salvarle; di default salva SOLO il CLS (leggero).
    Con --patches-fp16 le patch sono salvate in float16 (meta' spazio).
"""

import argparse
import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def collect_needed_frames(pie_root, splits):
    """Ritorna dict {(set,video): set(frame_ids)} dei frame che servono."""
    sys.path.insert(0, str(Path(pie_root) / "utilities"))
    sys.path.insert(0, "data")
    sys.path.insert(0, ".")
    from data.pie_data import PIE

    pie_data_root = Path(pie_root) / "annotations"
    if not (pie_data_root / "annotations").exists():
        pie_data_root = Path(pie_root)
    imdb = PIE(data_path=str(pie_data_root))

    data_opts = dict(fstride=1, sample_type="all", data_split_type="default",
                     seq_type="crossing", min_track_size=76,
                     height_rng=[0, float("inf")], squarify_ratio=0)

    needed = defaultdict(set)
    for split in splits:
        seq = imdb.generate_data_trajectory_sequence(split, **data_opts)
        # seq['image'] e' una lista (per pedone) di liste di path immagine
        for track in seq["image"]:
            for p in track:
                p = p[0] if isinstance(p, (list, tuple, np.ndarray)) else p
                p = str(p)
                # .../setXX/video_XXXX/NNNNN.png
                parts = Path(p).parts
                try:
                    sid = [x for x in parts if x.startswith("set")][-1]
                    vid = [x for x in parts if x.startswith("video_")][-1]
                    fid = int(Path(p).stem)
                except (IndexError, ValueError):
                    continue
                needed[(sid, vid)].add(fid)

    total = sum(len(v) for v in needed.values())
    print(f"\nFrame unici necessari: {total} in {len(needed)} video")
    return needed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pie-root", default="/Users/robertodioguardi/Desktop/PIE")
    ap.add_argument("--video-root", required=True,
                    help="cartella con setXX/video_XXXX.mp4")
    ap.add_argument("--out-dir", default="features/dinov3")
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--h", type=int, default=544)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--save-patches", action="store_true",
                    help="salva anche la griglia di patch (pesante)")
    ap.add_argument("--patches-fp16", action="store_true",
                    help="salva le patch in float16 (meta' spazio)")
    args = ap.parse_args()

    import torch
    import cv2
    import h5py
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image

    os.makedirs(args.out_dir, exist_ok=True)

    # device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # 1) frame necessari
    needed = collect_needed_frames(args.pie_root, args.splits)

    # 2) DINOv3
    print(f"Carico {args.model} ...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    patch = model.config.patch_size
    nreg = model.config.num_register_tokens
    hdim = model.config.hidden_size
    nph, npw = args.h // patch, args.w // patch
    print(f"patch={patch} reg={nreg} dim={hdim} | griglia {nph}x{npw}={nph*npw}")

    # ripartenza: carica indice esistente se presente
    idx_path = Path(args.out_dir) / "index.pkl"
    if idx_path.exists():
        with open(idx_path, "rb") as f:
            index = pickle.load(f)
        print(f"Indice esistente caricato: {len(index)} frame gia' fatti")
    else:
        index = {}  # (set,video,frame) -> {'h5': path, 'row': i}

    # processa per video
    videos = sorted(needed.keys())
    for vi, (sid, vid) in enumerate(videos):
        frame_ids = sorted(needed[(sid, vid)])
        mp4 = Path(args.video_root) / sid / f"{vid}.mp4"
        if not mp4.exists():
            print(f"⚠️  manca {mp4}, salto ({len(frame_ids)} frame persi)")
            continue

        h5path = Path(args.out_dir) / f"{sid}_{vid}.h5"
        # SKIP se gia' fatto: h5 esiste, apribile, e con il numero giusto di frame
        if h5path.exists():
            try:
                with h5py.File(h5path, "r") as _hf:
                    if _hf["cls"].shape[0] == len(frame_ids):
                        print(f"[{vi+1}/{len(videos)}] {sid}/{vid}: gia' fatto, skip")
                        continue
            except Exception:
                print(f"[{vi+1}/{len(videos)}] {sid}/{vid}: h5 corrotto, rifaccio")
        print(f"[{vi+1}/{len(videos)}] {sid}/{vid}: {len(frame_ids)} frame")
        cap = cv2.VideoCapture(str(mp4))

        # estrai i frame necessari (seek diretto)
        pil_frames, valid_fids = [], []
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, fr = cap.read()
            if not ok:
                continue
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(fr))
            valid_fids.append(fid)
        cap.release()

        if not pil_frames:
            continue

        # file h5 per questo video (h5path gia' definito sopra)
        with h5py.File(h5path, "w") as hf:
            n = len(pil_frames)
            cls_ds = hf.create_dataset("cls", shape=(n, hdim), dtype="float32")
            hf.create_dataset("frame_ids", data=np.array(valid_fids, dtype=np.int32))
            hf.attrs["set"] = sid; hf.attrs["video"] = vid
            hf.attrs["h"] = args.h; hf.attrs["w"] = args.w
            hf.attrs["patch"] = patch; hf.attrs["grid_h"] = nph; hf.attrs["grid_w"] = npw
            if args.save_patches:
                pdt = "float16" if args.patches_fp16 else "float32"
                patch_ds = hf.create_dataset(
                    "patches", shape=(n, nph, npw, hdim), dtype=pdt,
                    chunks=(1, nph, npw, hdim), compression="gzip", compression_opts=1)

            # forward in batch
            with torch.inference_mode():
                for start in range(0, n, args.batch):
                    chunk = pil_frames[start:start + args.batch]
                    inp = processor(images=chunk, return_tensors="pt",
                                    size={"height": args.h, "width": args.w},
                                    do_center_crop=False).to(device)
                    out = model(**inp)
                    cls = out.pooler_output.float().cpu().numpy()
                    cls_ds[start:start + len(chunk)] = cls
                    if args.save_patches:
                        lhs = out.last_hidden_state
                        pf = lhs[:, 1 + nreg:, :]
                        grid = pf.unflatten(1, (nph, npw)).float().cpu().numpy()
                        if args.patches_fp16:
                            grid = grid.astype(np.float16)
                        patch_ds[start:start + len(chunk)] = grid

        # aggiorna indice e SALVALO subito (ripartenza sicura)
        for i, fid in enumerate(valid_fids):
            index[(sid, vid, fid)] = {"h5": str(h5path), "row": i}
        with open(idx_path, "wb") as f:
            pickle.dump(index, f)

    # salva indice finale
    with open(idx_path, "wb") as f:
        pickle.dump(index, f)
    print(f"\nFatto. {len(index)} frame indicizzati.")
    print(f"Indice: {idx_path}")
    print(f"Feature h5 in: {args.out_dir}/")


if __name__ == "__main__":
    main()
