"""
DUMP finestre dalla TUA repo (pie_tesi).
Mettere nella root della tua repo e lanciarlo.
Produce: mine_samples_<split>.pkl con la STESSA struttura di dump_benchmark_samples.py

Uso:
    python dump_mine_samples.py --split test --benchmark-preproc

Con --benchmark-preproc usa il preprocessing del benchmark (delta-dal-frame0,
speed grezza). Senza, usa il tuo preprocessing ([0,1] + speed/120).
"""

import argparse
import pickle
import sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pie-root", default="/Users/robertodioguardi/Desktop/PIE")
    ap.add_argument("--obs-len", type=int, default=16)
    ap.add_argument("--benchmark-preproc", action="store_true")
    ap.add_argument("--sample-type", default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, "data")
    sys.path.insert(0, ".")
    from data.pie_dataset import PIEDataset

    ds = PIEDataset(
        args.pie_root, split=args.split, obs_len=args.obs_len,
        benchmark_preproc=args.benchmark_preproc,
        sample_type=args.sample_type,
    )

    samples = []
    for s in ds.samples:
        box = np.asarray(s["bbox"], dtype=np.float32)
        speed = np.asarray(s["ego_speed"], dtype=np.float32).reshape(-1, 1)
        samples.append({
            "ped_id": str(s["ped_id"]),
            "label": int(s["label"]),
            "box": box,
            "speed": speed,
            "tte": int(s.get("tte", -1)),
            "frames": s.get("frames", []),
        })

    tag = "bench" if args.benchmark_preproc else "mine"
    out = args.out or f"mine_samples_{args.split}_{tag}.pkl"
    with open(out, "wb") as f:
        pickle.dump(samples, f)
    print(f"Salvato {len(samples)} finestre in {out}")
    s = samples[0]
    print("\nEsempio finestra [0]:")
    print(f"  ped_id={s['ped_id']} label={s['label']} tte={s['tte']}")
    print(f"  box.shape={s['box'].shape} speed.shape={s['speed'].shape}")
    print(f"  box[0]={s['box'][0]}  box[-1]={s['box'][-1]}")
    print(f"  speed[:3]={s['speed'][:3].ravel()}")


if __name__ == "__main__":
    main()
