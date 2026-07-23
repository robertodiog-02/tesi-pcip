"""
Lettore delle feature DINOv3 — navigazione stile pickle sugli .h5
==================================================================

Gli .h5 sono efficienti ma scomodi da navigare a mano. Questo lettore
espone un'interfaccia a dizionario:

    r = DinoV3Reader("features/dinov3")

    r.get_cls("set05", "video_0001", 1234)              -> [768]
    r.get_roi("set05", "video_0001", "5_1_1731", 1234, scale=1.5)  -> [3,3,768]
    r.get_window_cls("set05", "video_0001", frames)     -> [T, 768]
    r.get_window_roi("set05", "video_0001", pid, frames, 1.5) -> [T,3,3,768]

    r.peds("set05", "video_0001")     -> lista dei pedoni disponibili
    r.frames("set05", "video_0001")   -> frame disponibili
    r.info()                          -> riepilogo di cosa c'e' dentro

I frame mancanti nelle finestre vengono riempiti con zeri (come per le pose).
"""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


class DinoV3Reader:
    """Accesso comodo alle feature DINOv3 salvate in .h5."""

    def __init__(self, feat_dir: str, cache_files: bool = True):
        import h5py
        self._h5py = h5py
        self.dir = Path(feat_dir)
        idx = self.dir / "index.pkl"
        if not idx.exists():
            raise FileNotFoundError(f"index.pkl non trovato in {self.dir}")
        with open(idx, "rb") as f:
            self.index: Dict = pickle.load(f)

        self._cache_files = cache_files
        self._open: Dict[str, "h5py.File"] = {}
        self._resolved: Dict[str, str] = {}

        # indici derivati per la navigazione
        self._by_video = defaultdict(lambda: {"frames": set(), "peds": set()})
        for k in self.index:
            if len(k) == 3:
                sid, vid, fid = k
                self._by_video[(sid, vid)]["frames"].add(fid)
            elif len(k) == 4:
                sid, vid, ped, fid = k
                self._by_video[(sid, vid)]["peds"].add(ped)

    # ── gestione file ────────────────────────────────────────────────────
    def _resolve(self, path: str) -> str:
        """
        Path utilizzabile su QUESTA macchina.

        Un indice creato su Windows contiene backslash, che su macOS/Linux non
        sono separatori: Path() vedrebbe l'intera stringa come nome di file.
        Se il path cosi' com'e' non esiste, si ripiega sul nome del file dentro
        la cartella delle feature.
        """
        if path in self._resolved:
            return self._resolved[path]
        p = path
        if not Path(p).exists():
            name = str(path).replace("\\", "/").rstrip("/").split("/")[-1]
            cand = self.dir / name
            if cand.exists():
                p = str(cand)
        self._resolved[path] = p
        return p

    def _file(self, path: str):
        path = self._resolve(path)
        if not self._cache_files:
            return self._h5py.File(path, "r")
        if path not in self._open:
            self._open[path] = self._h5py.File(path, "r")
        return self._open[path]

    def close(self):
        for f in self._open.values():
            f.close()
        self._open.clear()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ── metadati ─────────────────────────────────────────────────────────
    def videos(self) -> List:
        return sorted(self._by_video.keys())

    def sets(self) -> List[str]:
        return sorted({s for s, _ in self._by_video})

    def frames(self, set_id: str, video_id: str) -> List[int]:
        return sorted(self._by_video[(set_id, video_id)]["frames"])

    def peds(self, set_id: str, video_id: str) -> List[str]:
        return sorted(self._by_video[(set_id, video_id)]["peds"])

    def attrs(self, set_id: str, video_id: str) -> Dict:
        k = next(k for k in self.index
                 if len(k) == 3 and k[0] == set_id and k[1] == video_id)
        hf = self._file(self.index[k]["h5"])
        return dict(hf.attrs)

    def scales(self, set_id: str = None, video_id: str = None) -> List[float]:
        if set_id is None:
            set_id, video_id = self.videos()[0]
        return [float(s) for s in self.attrs(set_id, video_id)["scales"]]

    def info(self):
        """Riepilogo leggibile di cosa contiene la cartella."""
        vids = self.videos()
        n_cls = sum(1 for k in self.index if len(k) == 3)
        n_roi = sum(1 for k in self.index if len(k) == 4)
        print(f"Feature DINOv3 in {self.dir}")
        print(f"  set:    {', '.join(self.sets())}")
        print(f"  video:  {len(vids)}")
        print(f"  frame (CLS):  {n_cls}")
        print(f"  ROI (ped x frame): {n_roi}")
        if vids:
            a = self.attrs(*vids[0])
            print(f"  input:  {a['h']}x{a['w']}  griglia {a['grid_h']}x{a['grid_w']}")
            sz = self.roi_sizes(*vids[0])
            print(f"  hidden_dim: {a['hidden_dim']}")
            print(f"  scale:      {[float(s) for s in a['scales']]}")
            print(f"  roi sizes:  {sz}  -> dataset: "
                  + ", ".join(f"roi_{s:g}x_s{S}"
                              for s in [float(x) for x in a['scales']] for S in sz))
            print(f"  squarify: {bool(a['squarify'])}")

    # ── accesso ai dati ──────────────────────────────────────────────────
    def get_cls(self, set_id: str, video_id: str, frame: int) -> Optional[np.ndarray]:
        """CLS token del frame: [hidden_dim], None se assente."""
        e = self.index.get((set_id, video_id, int(frame)))
        if e is None:
            return None
        return np.asarray(self._file(e["h5"])["cls"][e["row_cls"]])

    def get_roi(self, set_id: str, video_id: str, ped_id: str, frame: int,
                scale: float = 1.5, size: int = None,
                flat: bool = False) -> Optional[np.ndarray]:
        """
        ROI Align del pedone a quel frame.

        Args:
            scale: 1.0 | 1.5 | 2.0 — quanto e' larga la regione ritagliata
            size:  1 | 3 | 7 — con quante celle e' descritta (None = la piu'
                   piccola disponibile)
            flat:  se True appiattisce [S,S,C] -> [S*S*C]
        Returns:
            [S, S, hidden_dim] oppure [S*S*hidden_dim] se flat
        """
        e = self.index.get((set_id, video_id, str(ped_id), int(frame)))
        if e is None:
            return None
        hf = self._file(e["h5"])
        if size is None:
            size = self.roi_sizes(set_id, video_id)[0]
        key = f"roi_{scale:g}x_s{size}"
        if key not in hf:
            avail = [k for k in hf.keys() if k.startswith("roi_")]
            raise KeyError(f"{key} non presente. Disponibili: {avail}")
        v = np.asarray(hf[key][e["row_roi"]])
        return v.reshape(-1) if flat else v

    def roi_sizes(self, set_id: str = None, video_id: str = None) -> List[int]:
        """Output size disponibili (es. [1, 3])."""
        if set_id is None:
            set_id, video_id = self.videos()[0]
        a = self.attrs(set_id, video_id)
        if "roi_sizes" in a:
            return [int(x) for x in a["roi_sizes"]]
        return [int(a["roi_size"])]        # retrocompat con h5 vecchi

    def get_window_cls(self, set_id: str, video_id: str,
                       frames: Sequence[int]) -> np.ndarray:
        """CLS per una finestra: [T, hidden_dim], zero-fill sui mancanti."""
        out, dim = [], None
        for f in frames:
            v = self.get_cls(set_id, video_id, f)
            if v is not None:
                dim = v.shape[-1]
            out.append(v)
        if dim is None:
            dim = int(self.attrs(set_id, video_id)["hidden_dim"])
        return np.stack([o if o is not None else np.zeros(dim, np.float32)
                         for o in out]).astype(np.float32)

    def get_window_roi(self, set_id: str, video_id: str, ped_id: str,
                       frames: Sequence[int], scale: float = 1.5,
                       size: int = None, flat: bool = False) -> np.ndarray:
        """ROI per una finestra: [T, S, S, C] (o [T, S*S*C] se flat)."""
        if size is None:
            size = self.roi_sizes(set_id, video_id)[0]
        out, shape = [], None
        for f in frames:
            v = self.get_roi(set_id, video_id, ped_id, f, scale, size, flat)
            if v is not None:
                shape = v.shape
            out.append(v)
        if shape is None:
            C = int(self.attrs(set_id, video_id)["hidden_dim"])
            shape = (size * size * C,) if flat else (size, size, C)
        return np.stack([o if o is not None else np.zeros(shape, np.float32)
                         for o in out]).astype(np.float32)

    def missing_report(self, set_id: str, video_id: str, ped_id: str,
                       frames: Sequence[int]) -> Dict:
        """Quanti frame della finestra hanno CLS / ROI disponibili."""
        n = len(frames)
        c = sum(self.index.get((set_id, video_id, int(f))) is not None
                for f in frames)
        r = sum(self.index.get((set_id, video_id, str(ped_id), int(f))) is not None
                for f in frames)
        return {"n_frames": n, "cls_ok": c, "roi_ok": r,
                "cls_missing": n - c, "roi_missing": n - r}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ispeziona le feature DINOv3")
    ap.add_argument("feat_dir")
    ap.add_argument("--set", default=None)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    with DinoV3Reader(args.feat_dir) as r:
        r.info()
        if args.set and args.video:
            fr = r.frames(args.set, args.video)
            pd = r.peds(args.set, args.video)
            print(f"\n{args.set}/{args.video}")
            print(f"  frame: {len(fr)} ({fr[0]}..{fr[-1]})")
            print(f"  pedoni: {pd}")
            if fr and pd:
                c = r.get_cls(args.set, args.video, fr[0])
                print(f"  CLS[{fr[0]}]: {c.shape}")
                for sc in r.scales(args.set, args.video):
                    for S in r.roi_sizes(args.set, args.video):
                        v = r.get_roi(args.set, args.video, pd[0], fr[0], sc, S)
                        if v is not None:
                            print(f"  ROI {sc}x s{S} [{pd[0]}, f{fr[0]}]: {v.shape}")
