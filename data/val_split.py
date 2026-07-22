"""
Split stratificato train/val per pedone — persistente e riproducibile
======================================================================

Campiona N pedoni dal training set per usarli come validation interno.

DUE SCELTE IMPORTANTI:

1. SPLIT PER PEDONE, non per finestra.
   Ogni pedone genera piu' finestre sovrapposte (overlap 0.6). Se lo split
   fosse per finestra, finestre quasi identiche dello stesso pedone
   finirebbero sia in train sia in val, e il val darebbe risultati
   ottimisticamente gonfiati. Qui tutte le finestre di un pedone vanno
   insieme, da una parte sola.

2. CAMPIONAMENTO STRATIFICATO, non uniforme.
   PIE train e' sbilanciato (~25% crossing, ~75% non-crossing). Campionando
   80 pedoni uniformemente ci si aspettano ~20 crossing, ma con deviazione
   standard ~4: si potrebbe finire con 12 o con 28. Su un val piccolo questo
   sposta molto l'F1 e si finirebbe a selezionare il modello sul caso.
   Stratificando, la proporzione e' garantita.

Gli ID vengono salvati su file e RIUSATI nei run successivi, cosi' tutti gli
esperimenti confrontano i modelli sullo stesso identico validation set.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


def _labels_by_ped(samples: Sequence[Dict]) -> Dict[str, int]:
    """ped_id -> label (le finestre di un pedone hanno tutte la stessa)."""
    out = {}
    for s in samples:
        out[str(s["ped_id"])] = int(s["label"])
    return out


def sample_val_pedestrians(
    samples: Sequence[Dict],
    n_val: int = 80,
    seed: int = 42,
    stratified: bool = True,
) -> Tuple[List[str], Dict]:
    """
    Sceglie n_val pedoni da usare come validation.

    Returns:
        (lista di ped_id, dict con statistiche della composizione)
    """
    ped_label = _labels_by_ped(samples)
    peds = sorted(ped_label)                     # ordinato -> deterministico
    rng = np.random.default_rng(seed)

    if not stratified:
        chosen = list(rng.choice(peds, size=min(n_val, len(peds)), replace=False))
    else:
        pos = sorted([p for p in peds if ped_label[p] == 1])
        neg = sorted([p for p in peds if ped_label[p] == 0])
        frac_pos = len(pos) / max(len(peds), 1)

        n_pos = int(round(n_val * frac_pos))
        n_pos = max(1, min(n_pos, len(pos)))
        n_neg = min(n_val - n_pos, len(neg))

        chosen = (list(rng.choice(pos, size=n_pos, replace=False))
                  + list(rng.choice(neg, size=n_neg, replace=False)))

    chosen = sorted(str(c) for c in chosen)
    n_c = sum(ped_label[p] == 1 for p in chosen)
    stats = {
        "n_val_peds": len(chosen),
        "n_val_crossing": n_c,
        "n_val_non_crossing": len(chosen) - n_c,
        "val_crossing_frac": round(n_c / max(len(chosen), 1), 4),
        "n_train_peds_total": len(peds),
        "train_crossing_frac": round(
            sum(v == 1 for v in ped_label.values()) / max(len(peds), 1), 4),
        "stratified": stratified,
        "seed": seed,
    }
    return chosen, stats


def load_or_create_split(
    samples: Sequence[Dict],
    path: str,
    n_val: int = 80,
    seed: int = 42,
    stratified: bool = True,
    verbose: bool = True,
) -> Set[str]:
    """
    Carica lo split da file se esiste, altrimenti lo crea e lo salva.

    Riusare sempre lo stesso file e' importante: garantisce che tutti gli
    esperimenti selezionino il modello sullo stesso validation set, quindi
    i confronti tra configurazioni sono onesti.
    """
    p = Path(path)

    if p.exists():
        with open(p) as f:
            data = json.load(f)
        ids = set(data["val_ped_ids"])
        if verbose:
            st = data.get("stats", {})
            print(f"  Split caricato da {p}")
            print(f"    {len(ids)} pedoni in val "
                  f"({st.get('n_val_crossing','?')} crossing / "
                  f"{st.get('n_val_non_crossing','?')} non-crossing), "
                  f"seed={st.get('seed','?')}")

        # avviso se il file non copre i pedoni attuali (dataset cambiato)
        present = {str(s["ped_id"]) for s in samples}
        missing = ids - present
        if missing and verbose:
            print(f"    ATTENZIONE: {len(missing)} pedoni del file non sono "
                  f"nel dataset attuale (sample_type o filtri diversi?)")
        return ids

    ids_list, stats = sample_val_pedestrians(samples, n_val, seed, stratified)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"val_ped_ids": ids_list, "stats": stats}, f, indent=2)

    if verbose:
        print(f"  Split creato e salvato in {p}")
        print(f"    {stats['n_val_peds']} pedoni in val: "
              f"{stats['n_val_crossing']} crossing / "
              f"{stats['n_val_non_crossing']} non-crossing "
              f"({100*stats['val_crossing_frac']:.1f}% crossing)")
        print(f"    train completo: {stats['n_train_peds_total']} pedoni, "
              f"{100*stats['train_crossing_frac']:.1f}% crossing")
        if stats["stratified"]:
            print("    campionamento STRATIFICATO (proporzione preservata)")

    return set(ids_list)


def split_samples(
    samples: Sequence[Dict],
    val_ped_ids: Set[str],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Divide i sample in (train, val) secondo i ped_id scelti.
    Nessuna finestra viene spezzata: split per pedone.
    """
    tr = [s for s in samples if str(s["ped_id"]) not in val_ped_ids]
    va = [s for s in samples if str(s["ped_id"]) in val_ped_ids]
    return tr, va


def describe_split(train_s: Sequence[Dict], val_s: Sequence[Dict]) -> None:
    """Stampa la composizione dei due insiemi dopo lo split."""
    def stat(ss):
        n = len(ss)
        pos = sum(int(s["label"]) == 1 for s in ss)
        peds = len({str(s["ped_id"]) for s in ss})
        return n, pos, n - pos, peds

    ntr, ptr, gtr, dtr = stat(train_s)
    nva, pva, gva, dva = stat(val_s)
    print(f"    TRAIN: {ntr:5d} finestre  ({dtr:3d} pedoni)  "
          f"crossing {ptr:4d} / non-crossing {gtr:4d}  "
          f"({100*ptr/max(ntr,1):.1f}% pos)")
    print(f"    VAL:   {nva:5d} finestre  ({dva:3d} pedoni)  "
          f"crossing {pva:4d} / non-crossing {gva:4d}  "
          f"({100*pva/max(nva,1):.1f}% pos)")

    # controllo anti-leakage
    tr_p = {str(s["ped_id"]) for s in train_s}
    va_p = {str(s["ped_id"]) for s in val_s}
    overlap = tr_p & va_p
    if overlap:
        raise RuntimeError(
            f"LEAKAGE: {len(overlap)} pedoni presenti in entrambi gli insiemi! "
            f"Esempi: {sorted(overlap)[:5]}")
