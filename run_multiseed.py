"""
Esegue lo stesso training con piu' seed e riporta media +/- deviazione standard
delle metriche di test. Serve a capire se il tuo F1 e' un valore stabile o
soggetto a forte varianza di inizializzazione (cruciale per confrontarsi col
singolo numero del benchmark).

Uso:
  python run_multiseed.py --config configs/benchmark_singlernn.yaml
  python run_multiseed.py --config configs/benchmark_singlernn.yaml --seeds 42 43 44 45 46
  python run_multiseed.py --config configs/benchmark_singlernn.yaml --python python

Ogni run salva in checkpoints/<exp_name>/seed_<seed>/test_results.json.
Alla fine stampa una tabella riassuntiva e salva multiseed_summary.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import statistics
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--python", default=sys.executable,
                    help="interprete python da usare (default: quello corrente)")
    ap.add_argument("--train-script", default="train.py")
    args = ap.parse_args()

    # leggi nome esperimento e checkpoint_dir dal config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    exp_name = cfg["experiment"]["name"]
    ckpt_dir = cfg.get("output", {}).get("checkpoint_dir", "checkpoints")

    print("=" * 70)
    print(f"MULTI-SEED: {exp_name}")
    print(f"Seeds: {args.seeds}")
    print("=" * 70)

    results = {}   # seed -> dict metriche test
    for seed in args.seeds:
        print(f"\n{'#'*70}\n# RUN seed={seed}\n{'#'*70}")
        cmd = [args.python, args.train_script, "--config", args.config, "--seed", str(seed)]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"⚠️  run con seed={seed} terminato con errore (returncode={ret.returncode})")
            continue

        # leggi il test_results.json prodotto da quel run
        res_path = Path(ckpt_dir) / exp_name / f"seed_{seed}" / "test_results.json"
        if not res_path.exists():
            print(f"⚠️  {res_path} non trovato, salto")
            continue
        with open(res_path) as f:
            metrics = json.load(f)
        results[seed] = metrics.get("test", {})
        t = results[seed]
        print(f"  -> seed {seed}: f1={t.get('f1'):.4f} acc={t.get('acc'):.4f} "
              f"P={t.get('precision'):.4f} R={t.get('recall'):.4f} auc={t.get('auc'):.4f}")

    if len(results) == 0:
        print("\nNessun risultato raccolto.")
        return

    # aggrega
    print("\n" + "=" * 70)
    print("RIEPILOGO SUL TEST SET (media +/- dev.std su", len(results), "run)")
    print("=" * 70)
    metric_names = ["f1", "acc", "auc", "precision", "recall"]
    summary = {}
    header = f"{'seed':>6} | " + " | ".join(f"{m:>9}" for m in metric_names)
    print(header)
    print("-" * len(header))
    for seed in sorted(results.keys()):
        row = results[seed]
        line = f"{seed:>6} | " + " | ".join(f"{row.get(m, float('nan')):9.4f}" for m in metric_names)
        print(line)
    print("-" * len(header))
    for m in metric_names:
        vals = [results[s][m] for s in results if m in results[s]]
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[m] = {"mean": mean, "std": std, "values": vals}
    mean_line = f"{'MEAN':>6} | " + " | ".join(f"{summary[m]['mean']:9.4f}" for m in metric_names)
    std_line  = f"{'STD':>6} | " + " | ".join(f"{summary[m]['std']:9.4f}" for m in metric_names)
    print(mean_line)
    print(std_line)
    print("=" * 70)
    print(f"F1 test: {summary['f1']['mean']:.4f} +/- {summary['f1']['std']:.4f}")
    print(f"  range osservato: [{min(summary['f1']['values']):.4f}, {max(summary['f1']['values']):.4f}]")
    print("=" * 70)

    out = Path(ckpt_dir) / exp_name / "multiseed_summary.json"
    with open(out, "w") as f:
        json.dump({"seeds": args.seeds, "per_seed": results, "summary": summary}, f, indent=2)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
