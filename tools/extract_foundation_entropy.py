#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取 FoundationStereo 在指定 manifest 上的 **初始概率分布 (init prob)** 的熵图，
用于诊断 BANet2D / LightStereo 与 teacher 的差距是否与 teacher 自身不确定性相关。

- 只跑到 init prob 阶段，跳过 GRU 迭代，单卡数百张 5-15 分钟
- 输出 (H/4, W/4) float16 .npy，与 BANet2D 的 prob 同分辨率
- 输入图像约定与原 foundation_out 生成脚本 (process_orbbec_all_in_one.py) 一致：
  raw uint8 -> float32 RGB，不做 mean/std normalize

用法:
  python tools/extract_foundation_entropy.py \\
      --manifest /data2/.../orbbec_merged_real_5ds/splits/openstereo/manifest_test.txt \\
      --data_root /data2/.../orbbec_merged_real_5ds/foundation_out \\
      --teacher_ckpt /data2/shendu/code/ruoyu/FoundationStereo/<model_dir>/model_best_bp2.pth \\
      --teacher_cfg /data2/shendu/code/ruoyu/FoundationStereo/<model_dir>/cfg.yaml \\
      --out_dir /data2/.../orbbec_merged_real_5ds/foundation_out/test/training/entropy
"""



# 读图（保持 uint8）→ pad 32 倍数 → 共享 backbone 算特征 → 构 GWC + concat cost volume → 3D 聚合 →
#▎ classifier 出 logits → softmax 得 prob → 沿 disparity 维算熵 → 还原到原图 1/4 分辨率 → 存 fp16 .npy
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from stereo.modeling.models.foundationstereo.core.foundation_stereo import FoundationStereo  # noqa: E402
from stereo.modeling.models.foundationstereo.core.submodule import (  # noqa: E402
    build_concat_volume,
    build_gwc_volume,
)
from stereo.modeling.models.foundationstereo.core.utils.utils import InputPadder  # noqa: E402


def _load_model_cfg(cfg_path: Path) -> EasyDict:
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if "MODEL" in raw:
        return EasyDict(raw["MODEL"])
    return EasyDict(raw)


def _maybe_strip_module(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


@torch.no_grad()
def _forward_prob(model: FoundationStereo, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """
    复制 foundation_stereo.py:198-228 的前半段，跑到 init prob，返回 prob (B, max_disp/4, H/4, W/4).
    熵与截断/重归一化都在外面基于这份 prob 算（单一数据源）。
    """
    B = image1.shape[0]
    autocast = torch.cuda.amp.autocast
    with autocast(enabled=bool(model.args.get("mixed_precision", True)), dtype=model.precision_dtype):
        out, _ = model.feature(torch.cat([image1, image2], dim=0))
        features_left = [o[:B] for o in out]
        features_right = [o[B:] for o in out]
        gwc_volume = build_gwc_volume(
            features_left[0], features_right[0], model.args.max_disp // 4, model.cv_group
        )
        left_tmp = model.proj_cmb(features_left[0])
        right_tmp = model.proj_cmb(features_right[0])
        concat_volume = build_concat_volume(left_tmp, right_tmp, maxdisp=model.args.max_disp // 4)
        comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
        comb_volume = model.corr_stem(comb_volume)
        comb_volume = model.corr_feature_att(comb_volume, features_left[0])
        comb_volume = model.cost_agg(comb_volume, features_left)
        logits = model.classifier(comb_volume).squeeze(1)  # (B, max_disp/4, H/4, W/4)
        prob = F.softmax(logits, dim=1)
    return prob.float()


def _read_image(path: Path) -> np.ndarray:
    """读 RGB float-friendly uint8。与 process_orbbec_all_in_one.py 的 imageio.imread 等价（cv2 BGR→RGB）。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _parse_manifest(manifest: Path, data_root: Path):
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        left_rel, right_rel = parts[0], parts[1]
        left_abs = data_root / left_rel
        right_abs = data_root / right_rel
        stem = Path(left_rel).stem
        rows.append((left_abs, right_abs, stem))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="提取 FoundationStereo init prob 熵图")
    ap.add_argument("--manifest", type=Path, required=True, help="测试集 manifest（一行: left right disp）")
    ap.add_argument("--data_root", type=Path, required=True, help="foundation_out 根目录")
    ap.add_argument("--teacher_ckpt", type=Path, required=True, help="FoundationStereo 权重 .pth")
    ap.add_argument(
        "--teacher_cfg",
        type=Path,
        required=True,
        help="FoundationStereo cfg.yaml（权重同目录的 cfg.yaml，或 OpenStereo 的 fstereo_*.yaml）",
    )
    ap.add_argument("--out_dir", type=Path, required=True, help="熵图输出目录")
    ap.add_argument(
        "--save_prob_dir",
        type=Path,
        default=None,
        help="若指定，则额外保存截断+重归一化后的 teacher prob 分布到此目录（蒸馏用）",
    )
    ap.add_argument(
        "--student_bins",
        type=int,
        default=48,
        help="保存 teacher prob 时截取前 N 个 disparity bin（与 student max_disp//4 对齐）",
    )
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 张做 smoke test")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="默认跳过已存在的 .npy；加此 flag 强制覆盖",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_prob_dir is not None:
        args.save_prob_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    model_cfg = _load_model_cfg(args.teacher_cfg)
    # 一些必要字段的兜底。注意：EasyDict 的属性访问靠 __setattr__ 注册，dict.setdefault
    # 会绕过它（key 进底层 dict 但 .vit_size 属性读不到）。下面这几个字段代码里用 .get()
    # 读，setdefault 没问题；但 vit_size 在 extractor 里是属性访问，必须用属性赋值。
    model_cfg.setdefault("low_memory", False)
    model_cfg.setdefault("mixed_precision", True)
    model_cfg.setdefault("precision_dtype", "fp16")
    # 23-51-11 的 cfg.yaml 不含 vit_size；与 FoundationStereo 官方脚本一致默认 vitl
    # （run_demo / batch_generate_pseudo_gt 等均如此，伪 GT 即由 vitl 生成）。
    if "vit_size" not in model_cfg:
        model_cfg.vit_size = "vitl"

    print(f"[cfg] vit_size={model_cfg.get('vit_size')} max_disp={model_cfg.get('max_disp')}")
    print(f"[device] {device}")

    model = FoundationStereo(model_cfg).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    sd = _maybe_strip_module(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"       e.g. missing: {missing[:3]}")
    if unexpected:
        print(f"       e.g. unexpected: {unexpected[:3]}")
    model.eval()

    rows = _parse_manifest(args.manifest, args.data_root)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[manifest] {len(rows)} pairs from {args.manifest}")

    for i, (lp, rp, stem) in enumerate(rows):
        out_path = args.out_dir / f"{stem}.npy"
        prob_path = args.save_prob_dir / f"{stem}.npy" if args.save_prob_dir is not None else None
        need_entropy = args.overwrite or not out_path.is_file()
        need_prob = prob_path is not None and (args.overwrite or not prob_path.is_file())
        if not need_entropy and not need_prob:
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(rows)}] skip existing")
            continue
        if not lp.is_file() or not rp.is_file():
            print(f"[skip] missing image: {lp}")
            continue

        left_img = _read_image(lp)
        right_img = _read_image(rp)
        H_orig, W_orig = left_img.shape[:2]
        H_q = max(1, H_orig // 4)
        W_q = max(1, W_orig // 4)

        left_t = torch.as_tensor(left_img, device=device).float()[None].permute(0, 3, 1, 2)
        right_t = torch.as_tensor(right_img, device=device).float()[None].permute(0, 3, 1, 2)

        padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
        left_t, right_t = padder.pad(left_t, right_t)

        prob_pad_q = _forward_prob(model, left_t, right_t)  # (1, max_disp/4, Hp/4, Wp/4)

        def _to_orig_quarter(x_pad_q: torch.Tensor) -> torch.Tensor:
            """(1, C, Hp/4, Wp/4) -> (1, C, H_orig/4, W_orig/4)，经 pad 全分辨率 unpad 去掉 padder 偏移。"""
            x_pad = F.interpolate(
                x_pad_q, size=(left_t.shape[2], left_t.shape[3]), mode="bilinear", align_corners=False
            )
            x_full = padder.unpad(x_pad)  # (1, C, H_orig, W_orig)
            return F.interpolate(x_full, size=(H_q, W_q), mode="bilinear", align_corners=False)

        if need_entropy:
            entropy_pad_q = -(prob_pad_q * (prob_pad_q + 1e-8).log()).sum(dim=1, keepdim=True)
            entropy_q = _to_orig_quarter(entropy_pad_q).squeeze().cpu().numpy().astype(np.float16)
            np.save(out_path, entropy_q)

        if need_prob:
            # 截取前 student_bins 个 bin 并沿 bin 维重归一化（与 BANet 48 bin 对齐）
            sub = prob_pad_q[:, : args.student_bins]
            sub = sub / sub.sum(dim=1, keepdim=True).clamp_min(1e-8)
            prob_q = _to_orig_quarter(sub)  # (1, student_bins, H_q, W_q)
            # 双线性重采后再归一化一次，确保是合法分布
            prob_q = prob_q / prob_q.sum(dim=1, keepdim=True).clamp_min(1e-8)
            prob_q = prob_q.squeeze(0).cpu().numpy().astype(np.float16)  # (student_bins, H_q, W_q)
            np.save(prob_path, prob_q)

        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            msg = f"[{i+1}/{len(rows)}] {stem} q=({H_q},{W_q})"
            if need_prob:
                msg += f" prob_bins={args.student_bins}"
            print(msg)

    print(f"[done] entropy -> {args.out_dir}")
    if args.save_prob_dir is not None:
        print(f"[done] prob -> {args.save_prob_dir}")


if __name__ == "__main__":
    main()
