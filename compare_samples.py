"""
CONFRONTA i due pickle di finestre (benchmark vs tua repo).
Allinea le finestre per (ped_id, tte) e verifica che box/speed/label coincidano.

Uso:
    python compare_samples.py benchmark_samples_test.pkl mine_samples_test_bench.pkl
"""

import argparse
import pickle
import numpy as np


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def key(s):
    return (str(s["ped_id"]), int(s["tte"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_pkl")
    ap.add_argument("mine_pkl")
    ap.add_argument("--atol", type=float, default=1e-3, help="tolleranza assoluta match numerico")
    args = ap.parse_args()

    A = load(args.bench_pkl)
    B = load(args.mine_pkl)
    print(f"benchmark: {len(A)} finestre")
    print(f"tua repo : {len(B)} finestre")

    # indicizza per (ped_id, tte)
    Ai = {}
    for s in A:
        Ai.setdefault(key(s), []).append(s)
    Bi = {}
    for s in B:
        Bi.setdefault(key(s), []).append(s)

    keysA = set(Ai); keysB = set(Bi)
    common = keysA & keysB
    only_a = keysA - keysB
    only_b = keysB - keysA
    print(f"\nchiavi (ped_id,tte) comuni : {len(common)}")
    print(f"solo nel benchmark          : {len(only_a)}")
    print(f"solo nella tua repo         : {len(only_b)}")
    if only_a:
        print("  es. solo-benchmark:", list(only_a)[:5])
    if only_b:
        print("  es. solo-tua:", list(only_b)[:5])

    # confronto numerico sulle chiavi comuni
    label_mismatch = 0
    box_shape_mismatch = 0
    box_val_mismatch = 0
    speed_val_mismatch = 0
    n_checked = 0
    max_box_diff = 0.0
    max_speed_diff = 0.0

    for k in common:
        sa = Ai[k][0]; sb = Bi[k][0]
        n_checked += 1
        if sa["label"] != sb["label"]:
            label_mismatch += 1
        ba, bb = np.asarray(sa["box"]), np.asarray(sb["box"])
        if ba.shape != bb.shape:
            box_shape_mismatch += 1
            continue
        dbox = np.abs(ba - bb).max()
        max_box_diff = max(max_box_diff, dbox)
        if dbox > args.atol:
            box_val_mismatch += 1
        pa, pb = np.asarray(sa["speed"]).ravel(), np.asarray(sb["speed"]).ravel()
        if pa.shape == pb.shape:
            dsp = np.abs(pa - pb).max()
            max_speed_diff = max(max_speed_diff, dsp)
            if dsp > args.atol:
                speed_val_mismatch += 1

    print("\n" + "=" * 60)
    print("CONFRONTO NUMERICO (sulle finestre comuni)")
    print("=" * 60)
    print(f"  finestre confrontate     : {n_checked}")
    print(f"  label diverse            : {label_mismatch}")
    print(f"  box shape diverse        : {box_shape_mismatch}")
    print(f"  box valori diversi (>atol): {box_val_mismatch}   (max diff={max_box_diff:.5f})")
    print(f"  speed valori diversi     : {speed_val_mismatch}   (max diff={max_speed_diff:.5f})")
    print("=" * 60)

    if (len(only_a)==0 and len(only_b)==0 and label_mismatch==0
            and box_val_mismatch==0 and speed_val_mismatch==0 and box_shape_mismatch==0):
        print("✅ IDENTICI: stesse finestre, stessi dati. Ogni differenza di risultato")
        print("   e' dovuta a modello/framework, NON ai dati.")
    else:
        print("⚠️  CI SONO DIFFERENZE nei dati. Dettaglio sopra: guarda se e'")
        print("   una questione di preprocessing (box/speed diversi) o di")
        print("   selezione finestre (chiavi solo-A / solo-B).")
        # mostra un esempio di finestra divergente
        for k in list(common)[:2000]:
            sa, sb = Ai[k][0], Bi[k][0]
            ba, bb = np.asarray(sa["box"]), np.asarray(sb["box"])
            if ba.shape == bb.shape and np.abs(ba-bb).max() > args.atol:
                print(f"\n  Esempio divergenza ped={k[0]} tte={k[1]}:")
                print(f"    bench box[0]={ba[0]}")
                print(f"    tua   box[0]={bb[0]}")
                print(f"    bench box[-1]={ba[-1]}")
                print(f"    tua   box[-1]={bb[-1]}")
                break


if __name__ == "__main__":
    main()
