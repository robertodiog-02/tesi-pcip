"""
Unione degli indici DINOv3 estratti su macchine diverse
=======================================================

Ogni video produce un .h5 indipendente e le chiavi dell'indice contengono
(set, video, ...), quindi due macchine che lavorano su set DISGIUNTI generano
chiavi che non collidono mai. L'unione e' quindi sicura.

Cosa fa lo script:
  1. legge gli index.pkl indicati
  2. segnala eventuali chiavi duplicate (indice di un problema: stesso set
     processato due volte)
  3. RICOSTRUISCE i path degli .h5 in base a dove si trovano davvero ora,
     cosi' l'indice resta valido anche dopo aver copiato i file da un'altra
     macchina (i path assoluti del Mac non esistono sulla RTX e viceversa)
  4. verifica che ogni .h5 referenziato esista e sia apribile
  5. scrive l'index.pkl unificato

Uso tipico — dopo aver copiato tutti gli .h5 in una cartella sola:

    python merge_index.py \\
        --inputs mac/index.pkl rtx/index.pkl \\
        --feat-dir features/dinov3 \\
        --out features/dinov3/index.pkl

Oppure, se hai perso gli index.pkl, li ricostruisce dagli .h5:

    python merge_index.py --rebuild --feat-dir features/dinov3
"""

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path


def _basename_any_os(path: str) -> str:
    """
    Nome del file da un path generato su QUALSIASI sistema operativo.

    Serve perche' Path("a\\b\\c.h5").name su Linux/macOS restituisce
    l'intera stringa: il backslash non e' un separatore su POSIX. Un index
    creato su Windows e riletto sul Mac (o viceversa) va normalizzato.
    """
    return str(path).replace("\\", "/").rstrip("/").split("/")[-1]


def rebuild_from_h5(feat_dir: Path) -> dict:
    """Ricostruisce l'indice leggendo direttamente gli .h5 (nessun pickle)."""
    import h5py
    index = {}
    files = sorted(feat_dir.glob("*.h5"))
    print(f"Ricostruzione da {len(files)} file .h5 in {feat_dir}")
    for i, f in enumerate(files, 1):
        try:
            with h5py.File(f, "r") as hf:
                sid = hf.attrs["set"]
                vid = hf.attrs["video"]
                sid = sid.decode() if isinstance(sid, bytes) else str(sid)
                vid = vid.decode() if isinstance(vid, bytes) else str(vid)

                for r, fid in enumerate(hf["frame_ids"][:]):
                    index[(sid, vid, int(fid))] = {"h5": str(f), "row_cls": r}

                if "roi_ped_id" in hf:
                    peds = hf["roi_ped_id"][:]
                    fids = hf["roi_frame_id"][:]
                    for j, (p, fd) in enumerate(zip(peds, fids)):
                        p = p.decode() if isinstance(p, bytes) else str(p)
                        index[(sid, vid, p, int(fd))] = {"h5": str(f), "row_roi": j}
            print(f"  [{i}/{len(files)}] {f.name}: OK")
        except Exception as e:
            print(f"  [{i}/{len(files)}] {f.name}: ERRORE ({e}) — salto")
    return index


def main():
    ap = argparse.ArgumentParser(
        description="Unisce gli index.pkl di estrazioni su macchine diverse")
    ap.add_argument("--inputs", nargs="+", default=None,
                    help="index.pkl da unire")
    ap.add_argument("--feat-dir", required=True,
                    help="cartella dove stanno ORA tutti gli .h5")
    ap.add_argument("--out", default=None,
                    help="output (default: <feat-dir>/index.pkl)")
    ap.add_argument("--rebuild", action="store_true",
                    help="ignora gli index.pkl e ricostruisci dagli .h5")
    ap.add_argument("--no-verify", action="store_true",
                    help="salta la verifica di apertura degli .h5")
    args = ap.parse_args()

    feat_dir = Path(args.feat_dir)
    out_path = Path(args.out) if args.out else feat_dir / "index.pkl"

    if args.rebuild:
        merged = rebuild_from_h5(feat_dir)
    else:
        if not args.inputs:
            ap.error("servono --inputs oppure --rebuild")

        merged = {}
        dup = Counter()
        for src in args.inputs:
            with open(src, "rb") as f:
                idx = pickle.load(f)
            n_new = 0
            for k, v in idx.items():
                if k in merged:
                    dup[k[:2]] += 1
                else:
                    n_new += 1
                merged[k] = v
            print(f"  {src}: {len(idx)} voci ({n_new} nuove)")

        if dup:
            print(f"\n  ATTENZIONE: {sum(dup.values())} chiavi duplicate.")
            print("  Significa che lo stesso video e' stato processato su piu'")
            print("  macchine. Non e' un errore fatale (vince l'ultimo), ma "
                  "controlla la divisione dei set:")
            for (sid, vid), n in dup.most_common(10):
                print(f"    {sid}/{vid}: {n}")

        # ── ripara i path ──
        # Gli assoluti di un'altra macchina non valgono qui. Attenzione ai
        # separatori: un index generato su Windows contiene backslash, che su
        # macOS/Linux NON sono separatori — Path() vedrebbe tutta la stringa
        # come un unico nome di file. Normalizziamo prima di estrarre il nome.
        fixed = missing = 0
        for k, v in merged.items():
            name = _basename_any_os(v["h5"])
            local = feat_dir / name
            if local.exists():
                if v["h5"] != str(local):
                    v["h5"] = str(local)
                    fixed += 1
            else:
                missing += 1
        if fixed:
            print(f"\n  Path riscritti su {feat_dir}: {fixed} voci")
        if missing:
            print(f"  ATTENZIONE: {missing} voci puntano a .h5 non presenti "
                  f"in {feat_dir} — copia i file mancanti prima di usare l'indice")

    # ── verifica ──
    if not args.no_verify:
        import h5py
        files = sorted({v["h5"] for v in merged.values()})
        print(f"\nVerifica di {len(files)} file .h5...")
        bad = []
        for f in files:
            try:
                with h5py.File(f, "r") as hf:
                    _ = hf["cls"].shape
            except Exception as e:
                bad.append((f, str(e)))
        if bad:
            print(f"  {len(bad)} file NON leggibili:")
            for f, e in bad[:10]:
                print(f"    {Path(f).name}: {e}")
        else:
            print("  tutti leggibili")

    # ── riepilogo ──
    per_set = defaultdict(lambda: {"videos": set(), "cls": 0, "roi": 0})
    for k in merged:
        sid, vid = k[0], k[1]
        per_set[sid]["videos"].add(vid)
        per_set[sid]["cls" if len(k) == 3 else "roi"] += 1

    print(f"\n{'set':<10} {'video':>6} {'frame':>9} {'ROI':>9}")
    print("-" * 38)
    for sid in sorted(per_set):
        d = per_set[sid]
        print(f"{sid:<10} {len(d['videos']):>6} {d['cls']:>9,} {d['roi']:>9,}")
    tot_c = sum(d["cls"] for d in per_set.values())
    tot_r = sum(d["roi"] for d in per_set.values())
    print("-" * 38)
    print(f"{'TOTALE':<10} {sum(len(d['videos']) for d in per_set.values()):>6} "
          f"{tot_c:>9,} {tot_r:>9,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(merged, f)
    print(f"\nIndice unificato: {out_path}  ({len(merged)} voci)")


if __name__ == "__main__":
    main()
