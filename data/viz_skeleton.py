"""
Verifica visiva delle coordinate della posa
============================================

Per un pedone di riferimento, genera le finestre da 16 frame e plotta lo
scheletro in coordinate RAW (pixel immagine) accanto allo scheletro
NORMALIZZATO, per controllare che la normalizzazione sia corretta.

Cosa guardare:
- RAW: lo scheletro si muove dentro il frame 1920x1080 seguendo il pedone,
  e cambia scala man mano che si avvicina/allontana.
- NORMALIZZATO (bbox_topleft): lo scheletro resta ancorato in alto a
  sinistra, le coordinate partono da ~0 e arrivano a ~(w, h) della box.
  Il movimento globale del pedone e' stato rimosso, resta la POSA.
- NORMALIZZATO (bbox_topleft_scaled): come sopra ma dentro [0,1] su
  entrambi gli assi, quindi invariante anche alla distanza.

Uso:
    python viz_skeleton.py --pose-dir poses --ped-id 5_2_1752 --n-windows 4
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pose_loader import normalize_keypoints, add_extra_joints

IMG_W, IMG_H = 1920.0, 1080.0

# scheletro COCO-17
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
EXTRA_EDGES = [(17, 0), (17, 5), (17, 6), (18, 11), (18, 12), (19, 17), (19, 18)]


def draw_skeleton(ax, kps, conf_thr=0.2, color="tab:blue", alpha=1.0, lw=1.5):
    """Disegna uno scheletro. kps: [N, 3]"""
    n = len(kps)
    edges = COCO_EDGES + (EXTRA_EDGES if n == 20 else [])
    for i, j in edges:
        if kps[i, 2] > conf_thr and kps[j, 2] > conf_thr:
            ax.plot([kps[i, 0], kps[j, 0]], [kps[i, 1], kps[j, 1]],
                    color=color, alpha=alpha, lw=lw, zorder=2)
    vis = kps[:, 2] > conf_thr
    ax.scatter(kps[vis, 0], kps[vis, 1], s=12, color=color,
               alpha=alpha, zorder=3, edgecolors="none")
    # punti extra evidenziati
    if n == 20:
        ex = kps[17:]
        exv = ex[:, 2] > conf_thr
        ax.scatter(ex[exv, 0], ex[exv, 1], s=45, marker="s",
                   facecolors="none", edgecolors="tab:red", lw=1.2, zorder=4)


def visualize_window(kps_raw, bboxes, frames, out_path, add_extra=True):
    """
    Tre pannelli: raw, bbox_topleft, bbox_topleft_scaled.
    kps_raw: [T, 17, 3], bboxes: [T, 4], frames: [T]
    """
    T = len(kps_raw)
    n_norm = normalize_keypoints(kps_raw, bboxes, "bbox_topleft")
    n_scal = normalize_keypoints(kps_raw, bboxes, "bbox_topleft_scaled")
    if add_extra:
        kps_raw = add_extra_joints(kps_raw)
        n_norm = add_extra_joints(n_norm)
        n_scal = add_extra_joints(n_scal)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(T - 1, 1)) for i in range(T)]

    # ── RAW ──
    ax = axes[0]
    for t in range(T):
        draw_skeleton(ax, kps_raw[t], color=colors[t], alpha=0.75)
    for t in (0, T - 1):
        x1, y1, x2, y2 = bboxes[t]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor="tab:orange", lw=1.4, ls="--"))
    ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0)
    ax.set_title(f"RAW — pixel immagine\nframe {int(frames[0])}–{int(frames[-1])}")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.grid(alpha=0.25)

    # ── bbox_topleft ──
    ax = axes[1]
    for t in range(T):
        draw_skeleton(ax, n_norm[t], color=colors[t], alpha=0.75)
    for t in (0, T - 1):
        w = bboxes[t, 2] - bboxes[t, 0]; h = bboxes[t, 3] - bboxes[t, 1]
        ax.add_patch(plt.Rectangle((0, 0), w, h, fill=False,
                                   edgecolor="tab:orange", lw=1.4, ls="--"))
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.invert_yaxis()
    ax.set_title("NORM bbox_topleft\n(x-x1, y-y1) — scala in pixel")
    ax.set_xlabel("x - x1 (px)"); ax.set_ylabel("y - y1 (px)")
    ax.grid(alpha=0.25)

    # ── bbox_topleft_scaled ──
    ax = axes[2]
    for t in range(T):
        draw_skeleton(ax, n_scal[t], color=colors[t], alpha=0.75)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False,
                               edgecolor="tab:orange", lw=1.4, ls="--"))
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-0.25, 1.25); ax.set_ylim(1.25, -0.25)
    ax.set_title("NORM bbox_topleft_scaled\n/(w,h) — invariante alla distanza")
    ax.set_xlabel("(x-x1)/w"); ax.set_ylabel("(y-y1)/h")
    ax.grid(alpha=0.25)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, T - 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("frame nella finestra (viola = t0, giallo = t15)")

    fig.suptitle("Quadrati rossi = punti extra (neck, hip, body center) | "
                 "tratteggio arancione = bounding box", y=0.02, fontsize=9)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def visualize_window_grid(kps_raw, bboxes, frames, out_path,
                          norm_mode="bbox_topleft", add_extra=True):
    """
    Griglia 2x16: UN PANNELLO PER FRAME.
    Riga superiore = coordinate RAW, riga inferiore = normalizzate.
    Ogni colonna e' lo stesso frame, cosi' si confrontano direttamente.
    """
    T = len(kps_raw)
    kn = normalize_keypoints(kps_raw, bboxes, norm_mode)
    kr = kps_raw.copy()
    if add_extra:
        kr = add_extra_joints(kr)
        kn = add_extra_joints(kn)

    fig, axes = plt.subplots(2, T, figsize=(1.55 * T, 7.0))
    if T == 1:
        axes = axes.reshape(2, 1)

    for t in range(T):
        # ── RAW: zoom sulla bbox del frame, con margine ──
        ax = axes[0, t]
        x1, y1, x2, y2 = bboxes[t]
        mx = max((x2 - x1) * 0.35, 12.0)
        my = max((y2 - y1) * 0.18, 12.0)
        draw_skeleton(ax, kr[t], color="tab:blue", lw=1.3)
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor="tab:orange", lw=1.1, ls="--"))
        ax.set_xlim(x1 - mx, x2 + mx)
        ax.set_ylim(y2 + my, y1 - my)          # y invertita
        ax.set_title(f"f{int(frames[t])}", fontsize=8, pad=3)
        ax.tick_params(labelsize=5.5)
        if t == 0:
            ax.set_ylabel("RAW\n(pixel immagine)", fontsize=9)

        # ── NORMALIZZATO ──
        ax = axes[1, t]
        draw_skeleton(ax, kn[t], color="tab:green", lw=1.3)
        if norm_mode == "bbox_topleft_scaled":
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False,
                                       edgecolor="tab:orange", lw=1.1, ls="--"))
            ax.set_xlim(-0.3, 1.3); ax.set_ylim(1.3, -0.3)
        else:
            w = x2 - x1; h = y2 - y1
            ax.add_patch(plt.Rectangle((0, 0), w, h, fill=False,
                                       edgecolor="tab:orange", lw=1.1, ls="--"))
            ax.set_xlim(-w * 0.35, w * 1.35)
            ax.set_ylim(h * 1.2, -h * 0.2)
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        ax.tick_params(labelsize=5.5)
        if t == 0:
            ax.set_ylabel(f"NORM\n{norm_mode}", fontsize=9)

    fig.suptitle(
        f"Un pannello per frame — finestra {int(frames[0])}–{int(frames[-1])}   |   "
        f"quadrati rossi = punti extra (neck, hip, center), "
        f"tratteggio = bounding box",
        fontsize=10, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir", default="poses")
    ap.add_argument("--ped-id", default=None, help="se omesso, sceglie il migliore")
    ap.add_argument("--n-windows", type=int, default=4)
    ap.add_argument("--obs-len", type=int, default=16)
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--out-dir", default="viz_out")
    ap.add_argument("--no-extra-joints", action="store_true")
    ap.add_argument("--norm-mode", default="bbox_topleft",
                    choices=["bbox_topleft", "bbox_topleft_scaled", "none"])
    ap.add_argument("--overlay", action="store_true",
                    help="genera anche la figura con i 16 frame sovrapposti")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True, parents=True)

    tracks = {}
    for f in sorted(Path(args.pose_dir).glob("pie_hrnet_poses_set*.pkl")):
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        for _, videos in data.items():
            for _, peds in videos.items():
                tracks.update(peds)

    if args.ped_id and args.ped_id in tracks:
        pid = args.ped_id
    else:
        pid = max(tracks, key=lambda k: (len(tracks[k]["frames"]) >= 100,
                                         tracks[k]["keypoints"][:, :, 2].mean()))
        print(f"ped-id non specificato/trovato -> uso {pid}")

    tr = tracks[pid]
    kps = np.asarray(tr["keypoints"], dtype=np.float32)
    bbs = np.asarray(tr["bbox"], dtype=np.float32)
    frs = np.asarray(tr["frames"])
    print(f"Pedone {pid}: {len(frs)} frame ({frs[0]}–{frs[-1]}), "
          f"conf media {kps[:, :, 2].mean():.3f}")

    starts = list(range(0, len(frs) - args.obs_len + 1, args.step))
    picks = np.linspace(0, len(starts) - 1, args.n_windows).astype(int)

    for k, si in enumerate(picks):
        s = starts[si]
        e = s + args.obs_len
        out = out_dir / f"skeleton_{pid}_win{k:02d}_f{int(frs[s])}_grid.png"
        visualize_window_grid(kps[s:e], bbs[s:e], frs[s:e], out,
                              norm_mode=args.norm_mode,
                              add_extra=not args.no_extra_joints)
        print(f"  finestra {k}: frame {int(frs[s])}–{int(frs[e-1])} -> {out}")
        if args.overlay:
            out2 = out_dir / f"skeleton_{pid}_win{k:02d}_f{int(frs[s])}_overlay.png"
            visualize_window(kps[s:e], bbs[s:e], frs[s:e], out2,
                             add_extra=not args.no_extra_joints)
            print(f"              overlay -> {out2}")

    print(f"\n{args.n_windows} figure in {out_dir}/")


if __name__ == "__main__":
    main()
