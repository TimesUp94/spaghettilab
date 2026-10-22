"""Frozen ImageNet backbone features for every crop / full-frame tile (cached).

    .venv/Scripts/python.exe tools/ggst_logger/model/features.py [--backbone resnet50] [--view crops|full]

-> <data>/ggst_dataset/model/feat_<backbone>_<view>.npz  {keys, feats [N, 13, D] float16}
Small batches, below-normal priority, so the GPU / CPUs stay usable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OFFS, OUT, key, load_rows, low_priority  # noqa: E402


def backbone(name):
    import torch
    import torchvision
    if name == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    elif name == "resnet18":
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    elif name == "convnext_tiny":
        m = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = torch.nn.Identity()
        return m
    else:
        raise ValueError(name)
    m.fc = torch.nn.Identity()
    return m


def tiles(path: Path, view: str):
    import cv2
    img = cv2.imread(str(path))[:, :, ::-1]
    w = img.shape[1] // len(OFFS)
    return [img[:, i * w:(i + 1) * w] for i in range(len(OFFS))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--view", default="crops", choices=["crops", "full"])
    ap.add_argument("--batch", type=int, default=8, help="openers per batch (x13 tiles)")
    a = ap.parse_args()
    low_priority()
    import torch
    torch.set_num_threads(4)
    rows = load_rows()
    keys = [key(r) for r in rows]
    m = backbone(a.backbone).cuda().eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1).half()
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1).half()
    feats = []
    for b in range(0, len(rows), a.batch):
        ims = []
        for r in rows[b:b + a.batch]:
            p = DATA / a.view / r["replay"] / f"{r['log_row']}_{r['attacker']}.jpg"
            ims += tiles(p, a.view)
        x = torch.from_numpy(np.stack(ims)).cuda().permute(0, 3, 1, 2).half().div(255)
        if a.view == "full":     # mirror like the crops: attacker always faces right
            for j, r in enumerate(rows[b:b + a.batch]):
                if not r["attacker_facing_right"]:
                    x[j * len(OFFS):(j + 1) * len(OFFS)] = x[j * len(OFFS):(j + 1) * len(OFFS)].flip(-1)
        with torch.no_grad():
            f = m((x - mean) / std).float().cpu().numpy()
        feats.append(f.reshape(-1, len(OFFS), f.shape[-1]).astype(np.float16))
        if b % 400 == 0:
            print(b, len(rows), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"feat_{a.backbone}_{a.view}.npz", keys=np.array(keys), feats=np.concatenate(feats))
    print("done", len(keys))


if __name__ == "__main__":
    main()
