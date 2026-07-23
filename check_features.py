"""
Verifica delle feature DINOv3 estratte
=======================================

Controlla che l'estrazione sia completa e coerente con quello che il
training si aspetta:

  1. l'indice si apre e gli .h5 referenziati esistono
  2. i metadati (scale, roi_sizes, hidden_dim) sono coerenti tra i file
  3. i valori letti sono sensati (niente NaN, niente tutto-zero)
  4. COPERTURA: per ogni finestra reale del dataset, ci sono le feature?
     Questo e' il controllo importante: usa lo stesso protocollo di
     build_samples, quindi dice esattamente quanti dati mancheranno al
     training.

Uso:
    # controllo veloce (solo indice e h5, nessuna annotazione PIE)
    python check_features.py --feat-dir features/dinov3

    # controllo completo, con la copertura sulle finestre reali
    python check_features.py --feat-dir features/dinov3 \\
        --pie-root ~/Desktop/PIE/annotations
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def check_index_and_files(reader):
    """1-2: indice, file, coerenza dei metadati."""
    print("=" * 68)
    print("1. INDICE E FILE")
    print("=" * 68)
    reader.info()

    vids = reader.videos()
    if not vids:
        print("\n  ERRORE: nessun video nell'indice")
        return False

    # coerenza dei metadati tra tutti i video
    print(f"\n  Coerenza metadati su {len(vids)} video...")
    ref = None
    bad = []
    for sid, vid in vids:
        try:
            a = reader.attrs(sid, vid)
            key = (int(a["hidden_dim"]), int(a["grid_h"]), int(a["grid_w"]),
                   tuple(float(x) for x in a["scales"]),
                   tuple(reader.roi_sizes(sid, vid)))
        except Exception as e:
            bad.append((sid, vid, str(e)))
            continue
        if ref is None:
            ref = key
        elif key != ref:
            bad.append((sid, vid, f"metadati diversi: {key} vs {ref}"))

    if bad:
        print(f"  ATTENZIONE: {len(bad)} video con problemi:")
        for sid, vid, e in bad[:5]:
            print(f"    {sid}/{vid}: {e}")
        return False
    print(f"    tutti coerenti: hidden_dim={ref[0]}, griglia {ref[1]}x{ref[2]}, "
          f"scale={list(ref[3])}, roi_sizes={list(ref[4])}")
    return True


def check_values(reader, n_samples=5):
    """3: i valori letti sono sensati?"""
    print("\n" + "=" * 68)
    print("2. VALORI")
    print("=" * 68)

    vids = reader.videos()
    rng = np.random.default_rng(0)
    picks = [vids[i] for i in rng.choice(len(vids), min(n_samples, len(vids)),
                                         replace=False)]
    ok = True
    for sid, vid in picks:
        frames = reader.frames(sid, vid)
        peds = reader.peds(sid, vid)
        if not frames:
            print(f"  {sid}/{vid}: nessun frame!")
            ok = False
            continue

        f = frames[len(frames) // 2]
        cls = reader.get_cls(sid, vid, f)
        msg = [f"  {sid}/{vid} f{f}:"]
        msg.append(f"CLS {cls.shape} range[{cls.min():.2f},{cls.max():.2f}]")
        if np.isnan(cls).any():
            msg.append("*** NaN ***"); ok = False
        if np.allclose(cls, 0):
            msg.append("*** tutto zero ***"); ok = False

        if peds:
            p = peds[0]
            # cerca un frame in cui questo pedone c'e'
            pf = next((x for x in frames
                       if reader.index.get((sid, vid, str(p), int(x)))), None)
            if pf is not None:
                for sc in reader.scales(sid, vid):
                    for S in reader.roi_sizes(sid, vid):
                        r = reader.get_roi(sid, vid, p, pf, sc, S)
                        if np.isnan(r).any():
                            msg.append(f"ROI {sc}x s{S}: *** NaN ***"); ok = False
                        elif np.allclose(r, 0):
                            msg.append(f"ROI {sc}x s{S}: *** zero ***"); ok = False
                r = reader.get_roi(sid, vid, p, pf, reader.scales(sid, vid)[0],
                                   reader.roi_sizes(sid, vid)[-1])
                msg.append(f"ROI {r.shape} range[{r.min():.2f},{r.max():.2f}]")
        print(" ".join(msg))

    print(f"\n  {'valori OK' if ok else 'PROBLEMI RILEVATI'}")
    return ok


def check_coverage(reader, pie_root, splits, obs_len):
    """4: copertura sulle finestre REALI del dataset."""
    print("\n" + "=" * 68)
    print("3. COPERTURA SULLE FINESTRE DEL DATASET")
    print("=" * 68)

    for p in [Path(pie_root) / "utilities", Path("data"), Path(".")]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from data.pie_data import PIE
    from data.windowing import iter_windows

    root = Path(pie_root) / "annotations"
    if not (root / "annotations").exists():
        root = Path(pie_root)
    imdb = PIE(data_path=str(root))
    opts = dict(fstride=1, sample_type="all", data_split_type="default",
                seq_type="crossing", min_track_size=obs_len + 60,
                height_rng=[0, float("inf")], squarify_ratio=0)

    import io, contextlib
    totals = defaultdict(lambda: {"win": 0, "cls_miss": 0, "roi_miss": 0,
                                  "frames": 0, "win_full": 0, "sets": set()})
    missing_sets = defaultdict(int)

    for split in splits:
        with contextlib.redirect_stdout(io.StringIO()):
            seq = imdb.generate_data_trajectory_sequence(split, **opts)
        pids = seq.get("pid") or seq.get("ped_id")

        for i, boxes in enumerate(seq["bbox"]):
            paths = seq["image"][i]
            ped_id = str(pids[i][0][0])
            parts = Path(str(paths[0])).parts
            try:
                sid = [x for x in parts if x.startswith("set")][-1]
                vid = [x for x in parts if x.startswith("video_")][-1]
            except IndexError:
                continue

            for w_start, w_end, _ in iter_windows(len(boxes), obs_len=obs_len):
                frames = [int(Path(str(paths[t])).stem)
                          for t in range(w_start, w_end)]
                rep = reader.missing_report(sid, vid, ped_id, frames)
                d = totals[split]
                d["win"] += 1
                d["frames"] += rep["n_frames"]
                d["cls_miss"] += rep["cls_missing"]
                d["roi_miss"] += rep["roi_missing"]
                d["sets"].add(sid)
                if rep["cls_missing"] == 0 and rep["roi_missing"] == 0:
                    d["win_full"] += 1
                elif rep["cls_missing"] == rep["n_frames"]:
                    missing_sets[sid] += 1

    print(f"\n{'split':<8} {'finestre':>9} {'complete':>9} {'%':>7} "
          f"{'CLS mancanti':>14} {'ROI mancanti':>14}")
    print("-" * 68)
    all_ok = True
    for split in splits:
        d = totals[split]
        if d["win"] == 0:
            print(f"{split:<8} nessuna finestra")
            continue
        pc = 100.0 * d["win_full"] / d["win"]
        pcls = 100.0 * d["cls_miss"] / max(d["frames"], 1)
        proi = 100.0 * d["roi_miss"] / max(d["frames"], 1)
        print(f"{split:<8} {d['win']:>9,} {d['win_full']:>9,} {pc:>6.1f}% "
              f"{d['cls_miss']:>7,} ({pcls:>4.1f}%) {d['roi_miss']:>7,} ({proi:>4.1f}%)")
        if pc < 99.0:
            all_ok = False

    if missing_sets:
        print(f"\n  Finestre SENZA alcuna feature, per set:")
        for sid, n in sorted(missing_sets.items(), key=lambda x: -x[1]):
            print(f"    {sid}: {n:,} finestre -> set probabilmente NON estratto")
        all_ok = False

    sets_found = sorted({s for d in totals.values() for s in d["sets"]})
    sets_in_feat = reader.sets()
    print(f"\n  Set richiesti dal dataset: {sets_found}")
    print(f"  Set presenti nelle feature: {sets_in_feat}")
    missing = set(sets_found) - set(sets_in_feat)
    if missing:
        print(f"  MANCANTI: {sorted(missing)}")
        all_ok = False

    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="features/dinov3")
    ap.add_argument("--pie-root", default=None,
                    help="se fornito, verifica la copertura sulle finestre reali")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--obs-len", type=int, default=16)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from dinov3_reader import DinoV3Reader

    with DinoV3Reader(args.feat_dir) as reader:
        ok1 = check_index_and_files(reader)
        ok2 = check_values(reader)
        ok3 = True
        if args.pie_root:
            ok3 = check_coverage(reader, args.pie_root, args.splits, args.obs_len)
        else:
            print("\n  (copertura non verificata: passa --pie-root per il "
                  "controllo completo)")

    print("\n" + "=" * 68)
    if ok1 and ok2 and ok3:
        print("TUTTO OK — le feature sono pronte per il training")
    else:
        print("CI SONO PROBLEMI — vedi sopra")
    print("=" * 68)


if __name__ == "__main__":
    main()
