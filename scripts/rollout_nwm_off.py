"""Official-data LPWM rollouts for the shared pixel referee (paper Table 3), one view at a time.

Windows = nwm score_baselines.windows_ep (--per-ep 4 -> 400 on the 100 official val eps); context = frames [t0-1, t0]
(cond_steps 2), then 32 frames from the logged actions. use_all_ctx=False: their eval default infers latent context
from future GT frames; open-loop rollouts must not. Preds stay at the model's res (the scorer resizes to 224).
  rollout: python scripts/rollout_nwm_off.py --ckpt <pth> --data <ogb_scene_off_ep256_val> --view 0 --out <prefix>
  select:  python scripts/rollout_nwm_off.py --select <snaps dir> --data <ogb_scene_off_ep128_v2v4>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_lpwm_video_prediction import load_dlp_from_config  # noqa: E402

CTX, HMAX, COND = 1, 32, 2
DEV = os.environ.get("LPWM_DEVICE", "cuda")


def windows_ep(n_eps, T, per_ep):
    span = T - (CTX + HMAX)
    return [(i, (i * 37 + k * (span // per_ep)) % span) for i in range(n_eps) for k in range(per_ep)]


def episode(path):  # -> frames (T,V,R,R,3) u8, actions (T,A)
    if path.suffix == ".npz":
        d = np.load(path)
        return d["frames"], d["actions"]
    return np.load(path, mmap_mode="r"), np.load(str(path).replace(".frames.", ".actions."))


def to_model(x, res):  # (N,R,R,3) u8 -> (N,3,res,res) 0..1: prep_nwm_npy 256->128 kernel, then their PIL Resize
    x = np.ascontiguousarray(x)
    if x.shape[1] != 128:
        t = F.interpolate(torch.from_numpy(x).permute(0, 3, 1, 2).float(), size=(128, 128), mode="bilinear",
                          align_corners=False, antialias=True)
        x = t.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    return torch.stack([torch.from_numpy(np.asarray(Image.fromarray(f).resize((res, res), Image.BILINEAR)))
                        .permute(2, 0, 1).float() / 255 for f in x])


@torch.no_grad()
def rollout(model, res, shards, wins, view, batch):
    preds, gts, cache = [], [], {}
    for b0 in range(0, len(wins), batch):
        xs, acts = [], []
        for ep, t0 in wins[b0:b0 + batch]:
            if ep not in cache:
                cache = {ep: episode(shards[ep])}
            fr, ac = cache[ep]
            idx = [max(t0 - 1, 0), t0]
            xs.append(to_model(fr[idx, view], res))
            acts.append(torch.from_numpy(np.asarray(ac[[*idx, *range(t0 + 1, t0 + 1 + HMAX)]], np.float32)))
            gts.append(np.asarray(fr[t0 + CTX:t0 + CTX + HMAX, view]))
        x, a = torch.stack(xs).to(DEV), torch.stack(acts).to(DEV)
        out = model.sample_from_x(x, cond_steps=COND, num_steps=HMAX, deterministic=True, use_all_ctx=False,
                                  actions=a, n_pred_eq_gt=False).clamp(0, 1)[:, COND:]
        preds.append((out.permute(0, 1, 3, 4, 2) * 255).round().byte().cpu().numpy())
    return np.concatenate(preds), np.stack(gts)


def psnr_curve(pred, gt):  # mean over windows of per-frame PSNR at 224, (32,)
    def g(x):
        t = torch.from_numpy(np.ascontiguousarray(x)).flatten(0, 1).permute(0, 3, 1, 2).float()
        return F.interpolate(t, size=(224, 224), mode="bilinear", antialias=True, align_corners=False) / 255
    p, q = g(pred), g(gt)
    mse = ((p - q) ** 2).mean((1, 2, 3)).view(pred.shape[0], pred.shape[1])
    return (-10 * torch.log10(mse.clamp(min=1e-10))).mean(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "ogbench_nwm.json"))
    ap.add_argument("--ckpt")
    ap.add_argument("--select", help="snapshot dir: rank epoch_*.pth / best_lpips_*.pth on held-out train episodes")
    ap.add_argument("--view", type=int, default=0)   # export view index: 0 = v2, 1 = v4
    ap.add_argument("--per-ep", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    shards = sorted(Path(a.data).glob("ep*.npz")) or sorted(Path(a.data).glob("ep*.frames.npy"))
    assert shards, f"no episodes under {a.data}"

    res = json.load(open(a.config))["image_size"]

    def load(ck):
        m = load_dlp_from_config(a.config, None)
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False), strict=True)   # theirs swallows errors
        return m.to(DEV).eval(), res

    if a.select:   # same held-out split as prep_nwm.py valid / GWM NWMEpisodeDataset(seed=42, val_ratio=0.05)
        perm = np.random.default_rng(42).permutation(len(shards))
        held = [shards[k] for k in sorted(perm[:max(1, int(len(shards) * 0.05))])]
        T = episode(held[0])[0].shape[0]
        wins = windows_ep(len(held), T, 2)[:a.limit]
        cks = sorted(Path(a.select).glob("epoch_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
        cks += sorted(Path(a.select).glob("best_lpips_*.pth"))
        for ck in cks:
            model, res = load(str(ck))
            px, gt = rollout(model, res, held, wins, a.view, a.batch)
            c = psnr_curve(px, gt)
            print(f"SELECT {ck.name} psnr@16 {c[:16].mean():.3f} psnr@32 {c.mean():.3f} ({len(wins)} held-out train windows)",
                  flush=True)
        return

    T = episode(shards[0])[0].shape[0]
    wins = windows_ep(len(shards), T, a.per_ep)[:a.limit]
    model, res = load(a.ckpt)
    px, gt = rollout(model, res, shards, wins, a.view, a.batch)
    c = psnr_curve(px, gt)
    print(f"ROLLOUT view {a.view} psnr@16 {c[:16].mean():.3f} psnr@32 {c.mean():.3f} ({len(wins)} windows)", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{out}.npz", pred=px, model="lpwm", ckpt=a.ckpt, view=a.view, per_ep=a.per_ep, data=a.data)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    torch.manual_seed(0)
    main()
