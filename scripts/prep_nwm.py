"""NWM official-data memmap shards (ep%05d.{frames,actions}.npy, frames (T,V,R,R,3)) -> LPWM OGBench folder layout.

<out>/scene/{train,valid}/ep%04d/{%04d.png, metadata%04d.pt}; one folder per (episode, view), id = view * 1000 + ep.
valid = GWM's held-out train split (NWMEpisodeDataset seed=42, val_ratio=0.05); official val eps stay referee-only.
  python scripts/prep_nwm.py --src <ogb_scene_off_ep128_v2v4> --out <dir> [--workers 16]
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

RES = 128


def convert(job):
    src, dst, view, idx = job
    d = dst / f"ep{idx:04d}"
    done = d / ".done"
    if done.exists():
        return idx, "skip"
    d.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".npz":   # official 256 export: same kernel as gaussianwm prep_nwm_npy (bilinear, antialias, round)
        z = np.load(src)
        x = torch.from_numpy(z["frames"][:, view]).permute(0, 3, 1, 2).float()
        fr = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False, antialias=True) \
            .round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
        act = z["actions"]
    else:
        fr = np.load(src, mmap_mode="r")[:, view]
        act = np.load(src.with_name(src.name.replace(".frames.", ".actions.")))
    T = fr.shape[0]
    for t in range(T):
        Image.fromarray(np.ascontiguousarray(fr[t])).save(d / f"{t:04d}.png", compress_level=1)
        torch.save({"actions": torch.from_numpy(act[t].copy()), "terminals": t == T - 1}, d / f"metadata{t:04d}.pt")
    done.touch()
    return idx, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="0,1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, help="first N episodes only (smoke data)")
    a = ap.parse_args()
    shards = (sorted(Path(a.src).glob("ep*.frames.npy")) or sorted(Path(a.src).glob("ep*.npz")))[:a.limit]
    assert shards, f"no ep*.frames.npy / ep*.npz under {a.src}"
    perm = np.random.default_rng(42).permutation(len(shards))
    held = set(perm[:max(1, int(len(shards) * 0.05))].tolist())
    out = Path(a.out)
    for split in ["train", "valid"]:
        (out / "cube" / split).mkdir(parents=True, exist_ok=True)   # loader lists cube/ too
    jobs = [(s, out / "scene" / ("valid" if e in held else "train"), v, v * 1000 + e)
            for v in [int(x) for x in a.views.split(",")] for e, s in enumerate(shards)]
    with Pool(a.workers) as p:
        for i, (idx, info) in enumerate(p.imap_unordered(convert, jobs)):
            if i % 100 == 0:
                print(f"[{i + 1}/{len(jobs)}] ep{idx:04d} {info}", flush=True)
    print(f"done: {len(jobs)} folders ({len(held)} held-out eps x views in valid) -> {out}", flush=True)


if __name__ == "__main__":
    main()
