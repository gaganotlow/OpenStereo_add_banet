#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从实验目录 train_*.log 中按验证集 EPE 选最佳 checkpoint，对 test manifest 一次完成：
  - 视差图（KITTI uint16 PNG）
  - 与 manifest 中视差 GT 对比：EPE；KITTI d1_all；以及与 BANet 官方 test_and_save.py 一致的 D1(>1px)、D3(>3px)（per-frame CSV + summary_test.txt）
  - 左目系 PLY，默认去掉 Z>2.5m（可改 --max-z）

用法示例（LightStereo 实验）:
  cd OpenStereo
  python tools/infer_exp_best_test_full.py \\
    --exp_dir output/lightstereo_m_orbbec_capture_20260509_640x480/orbbec_capture_202605011_m
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from easydict import EasyDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cfgs.data_basic import DATA_PATH_DICT  # noqa: E402
from stereo.evaluation.metric_per_image import (  # noqa: E402
    d1_metric,
    epe_metric,
    threshold_metric,
)
from stereo.modeling import build_trainer  # noqa: E402
from stereo.utils import common_utils  # noqa: E402
from stereo.utils.disp_color import disp_to_color  # noqa: E402


def _pick_train_log(exp_dir: Path) -> Path:
    logs = sorted(exp_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        raise FileNotFoundError("未找到 %s/train_*.log" % exp_dir)
    return logs[0]


def _parse_best_val_epe_epoch(log_path: Path) -> tuple[int, float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    pairs: list[tuple[int, float]] = []
    for m in re.finditer(
        r"Epoch (\d+) metrics:.*?'epe': tensor\(([0-9.]+)\)", text
    ):
        pairs.append((int(m.group(1)), float(m.group(2))))
    if not pairs:
        raise RuntimeError("日志中未解析到验证 EPE: %s" % log_path)
    ep, epe = min(pairs, key=lambda x: x[1])
    return ep, epe


def _pick_ckpt_path(exp_dir: Path, epoch: int) -> Path:
    p = exp_dir / "ckpt" / ("checkpoint_epoch_%d.pth" % epoch)
    if p.is_file():
        return p
    cks = glob.glob(str(exp_dir / "ckpt" / "checkpoint_epoch_*.pth"))
    if not cks:
        raise FileNotFoundError("无 checkpoint: %s/ckpt/" % exp_dir)
    best_alternate = None
    best_num = -1
    for c in cks:
        m = re.search(r"checkpoint_epoch_(\d+)\.pth$", c)
        if m:
            n = int(m.group(1))
            if n <= epoch and n > best_num:
                best_num = n
                best_alternate = Path(c)
    if best_alternate is not None:
        return best_alternate
    best_num = -1
    best_p = None
    for c in cks:
        m = re.search(r"checkpoint_epoch_(\d+)\.pth$", str(c))
        if not m:
            continue
        n = int(m.group(1))
        if n > best_num:
            best_num = n
            best_p = Path(c)
    if best_p is not None:
        return best_p
    raise FileNotFoundError("无法解析 ckpt: %s" % exp_dir)


def _default_cfg_in_exp(exp_dir: Path) -> Path:
    yamls = list(exp_dir.glob("*.yaml"))
    if not yamls:
        raise FileNotFoundError("实验目录下无 .yaml: %s" % exp_dir)
    return yamls[0]


def _expand_data_paths(cfgs) -> None:
    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        yaml_dp = getattr(each, "DATA_PATH", None)
        if dataset_name == "KittiDataset":
            sr = ""
            for k in ("TRAINING", "EVALUATING", "TESTING"):
                try:
                    v = each.DATA_SPLIT[k]
                except (KeyError, TypeError):
                    v = getattr(each.DATA_SPLIT, k, "")
                sr += str(v or "")
            path_ref = sr + (str(yaml_dp) if yaml_dp else "")
            use_k15 = ("kitti15" in path_ref) or ("foundation_out" in path_ref)
            dataset_name = "KittiDataset15" if use_k15 else "KittiDataset12"
        if yaml_dp:
            each.DATA_PATH = os.path.expanduser(os.path.expandvars(str(yaml_dp)))
        else:
            each.DATA_PATH = DATA_PATH_DICT[dataset_name]


def _load_intrinsics(path: Path, baseline_unit: str) -> tuple[float, float, float, float, float]:
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    k = j.get("depth_intrinsic") or j.get("rgb_intrinsic")
    fx = float(k["fx"])
    fy = float(k["fy"])
    cx = float(k["cx"])
    cy = float(k["cy"])
    sb = j.get("stereo_baseline") or {}
    b_raw = float(sb["baseline"])
    json_unit = (sb.get("unit") or "").strip().lower()
    if json_unit in ("mm", "millimeter", "millimeters"):
        b = b_raw / 1000.0
    elif json_unit in ("m", "meter", "meters"):
        b = b_raw
    elif baseline_unit == "mm":
        b = b_raw / 1000.0
    elif baseline_unit == "m":
        b = b_raw
    else:
        b = b_raw / 1000.0 if b_raw > 10.0 else b_raw
    return fx, fy, cx, cy, b


def _save_ply_ascii(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = xyz.shape[0]
    head = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            "element vertex %d" % n,
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n")
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def _export_ply(
    foundation_out: Path,
    manifest_test: Path,
    pred_disp_dir: Path,
    out_ply_dir: Path,
    intr_path: Path,
    pixel_subsample: int,
    min_disp: float,
    max_z: float,
    baseline_unit: str,
) -> None:
    fx, fy, cx, cy, baseline = _load_intrinsics(intr_path, baseline_unit)
    lines = manifest_test.read_text(encoding="utf-8").strip().splitlines()
    out_ply_dir.mkdir(parents=True, exist_ok=True)
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        left_rel = parts[0]
        left_abs = foundation_out / left_rel
        stem = Path(left_rel).stem
        pred = pred_disp_dir / ("%s.png" % stem)
        if not pred.is_file():
            continue
        d16 = cv2.imread(str(pred), cv2.IMREAD_UNCHANGED)
        if d16 is None:
            continue
        if d16.ndim != 2:
            d16 = d16[:, :, 0]
        disp = d16.astype(np.float32) / 256.0
        left_bgr = cv2.imread(str(left_abs))
        if left_bgr is None:
            continue
        if left_bgr.ndim == 2:
            left_bgr = cv2.cvtColor(left_bgr, cv2.COLOR_GRAY2BGR)
        h, w = disp.shape
        valid = (disp > 0) & np.isfinite(disp) & (disp >= min_disp)
        pts, cols = [], []
        for v in range(0, h, pixel_subsample):
            for u in range(0, w, pixel_subsample):
                d = float(disp[v, u])
                if not valid[v, u]:
                    continue
                z = (fx * baseline) / d
                if max_z > 0 and z > max_z:
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                pts.append((x, y, z))
                b, g, r = left_bgr[v, u]
                cols.append((r, g, b))
        if not pts:
            continue
        _save_ply_ascii(
            out_ply_dir / ("%s.ply" % stem),
            np.asarray(pts, dtype=np.float32),
            np.asarray(cols, dtype=np.uint8),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="最佳 ckpt + test 推理 + EPE + PLY(Z≤max-z)")
    parser.add_argument(
        "--exp_dir",
        type=Path,
        required=True,
        help="实验目录（含 ckpt/、train_*.log、备份 yaml）",
    )
    parser.add_argument("--cfg_file", type=Path, default=None, help="默认取实验目录下首个 .yaml")
    parser.add_argument("--out_dir", type=Path, default=None, help="默认 <exp_dir>/infer_test_best")
    parser.add_argument("--force_epoch", type=int, default=-1, help="指定 epoch，跳过按日志选最佳")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save_color", action="store_true")
    parser.add_argument("--pixel_subsample", type=int, default=2)
    parser.add_argument("--min-disp", type=float, default=0.0)
    parser.add_argument(
        "--max-z",
        type=float,
        default=2.5,
        help="去掉左相机 Z>该值(米)的点；0 表示不裁深度",
    )
    parser.add_argument("--baseline-unit", choices=("mm", "m", "auto"), default="mm")
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--test_manifest", type=Path, default=None, help="指定测试集manifest路径，覆盖默认的<session_root>/splits/openstereo/manifest_test.txt")
    args = parser.parse_args()

    exp_dir = args.exp_dir.resolve()
    log_path = _pick_train_log(exp_dir)
    if args.force_epoch >= 0:
        best_ep = args.force_epoch
        best_epe = float("nan")
    else:
        best_ep, best_epe = _parse_best_val_epe_epoch(log_path)
    ckpt_path = _pick_ckpt_path(exp_dir, best_ep)

    cfg_path = args.cfg_file.resolve() if args.cfg_file else _default_cfg_in_exp(exp_dir)
    out_root = args.out_dir.resolve() if args.out_dir else (exp_dir / "infer_test_best")
    out_root.mkdir(parents=True, exist_ok=True)

    summary_pick = out_root / "best_ckpt_info.txt"
    summary_pick.write_text(
        "train_log: %s\nbest_val_epe_epoch: %s\nbest_val_epe: %s\n"
        "checkpoint: %s\n"
        % (log_path, best_ep, best_epe, ckpt_path),
        encoding="utf-8",
    )

    yaml_config = common_utils.config_loader(str(cfg_path))
    cfgs = EasyDict(yaml_config)
    cfgs.MODEL.PRETRAINED_MODEL = str(ckpt_path)
    cfgs.MODEL.CKPT = -1

    fo = Path(cfgs.DATA_CONFIG.DATA_INFOS[0].DATA_PATH).resolve()
    session_root = fo.parent

    if args.test_manifest:
        manifest_test = args.test_manifest.resolve()
        if not manifest_test.is_file():
            raise FileNotFoundError(f"指定的测试集不存在: {manifest_test}")
        # 从manifest路径推导foundation_out路径
        # manifest路径格式: <session_root>/splits/openstereo/manifest_test.txt
        # foundation_out路径: <session_root>/foundation_out
        test_session_root = manifest_test.parent.parent.parent
        test_fo = test_session_root / "foundation_out"
        if test_fo.is_dir():
            cfgs.DATA_CONFIG.DATA_INFOS[0].DATA_PATH = str(test_fo)
            session_root = test_session_root
            fo = test_fo
    else:
        manifest_test = session_root / "splits" / "openstereo" / "manifest_test.txt"
        if not manifest_test.is_file():
            raise FileNotFoundError(manifest_test)

    cfgs.DATA_CONFIG.DATA_INFOS[0].DATA_SPLIT["EVALUATING"] = str(manifest_test)
    _expand_data_paths(cfgs)

    run_args = argparse.Namespace(
        dist_mode=False,
        run_mode="eval",
        workers=args.workers,
        pin_memory=False,
        save_root_dir="./output",
    )

    disp_dir = out_root / "disp_png"
    color_dir = out_root / "disp_color"
    ply_dir = out_root / "ply"
    disp_dir.mkdir(parents=True, exist_ok=True)
    if args.save_color:
        color_dir.mkdir(parents=True, exist_ok=True)

    logger = common_utils.create_logger(None, rank=0)
    logger.info("使用 checkpoint: %s (epoch=%s, log 中 val epe=%s)", ckpt_path, best_ep, best_epe)
    logger.info("test manifest: %s", manifest_test)
    if args.test_manifest:
        logger.info("使用自定义测试集，DATA_PATH: %s", cfgs.DATA_CONFIG.DATA_INFOS[0].DATA_PATH)

    trainer = build_trainer(run_args, cfgs, 0, 0, logger, None)
    model = trainer.model
    model.eval()
    loader = trainer.eval_loader
    device = 0
    rows = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            for k, v in list(data.items()):
                if torch.is_tensor(v):
                    data[k] = v.to(device)
            with torch.cuda.amp.autocast(enabled=cfgs.OPTIMIZATION.AMP):
                pred = model(data)
            disp_pred = pred["disp_pred"].squeeze(1)
            disp_gt = data["disp"]
            mask = (disp_gt < cfgs.EVALUATOR.MAX_DISP) & (disp_gt > 0)
            epe_b = epe_metric(disp_pred, disp_gt, mask)
            d1_kitti_b = d1_metric(disp_pred, disp_gt, mask)
            d1_gt1_b = threshold_metric(disp_pred, disp_gt, mask, 1.0)
            d3_gt3_b = threshold_metric(disp_pred, disp_gt, mask, 3.0)
            names = data["name"]
            if not isinstance(names, (list, tuple)):
                names = [names]
            bsz = disp_pred.shape[0]
            for bi in range(bsz):
                stem = Path(str(names[bi])).stem
                dp = disp_pred[bi].detach().float().cpu().numpy()
                u16 = np.round(np.clip(dp, 0.0, None) * 256.0).astype(np.uint16)
                cv2.imwrite(str(disp_dir / ("%s.png" % stem)), u16)
                rows.append(
                    {
                        "name": stem,
                        "epe": float(epe_b[bi].item()),
                        "d1_kitti_all_pct": float(d1_kitti_b[bi].item()),
                        "d1_abs_gt1px_pct": float(d1_gt1_b[bi].item()),
                        "d3_abs_gt3px_pct": float(d3_gt3_b[bi].item()),
                    }
                )
                if args.save_color:
                    col = disp_to_color(dp, max_disp=int(cfgs.EVALUATOR.MAX_DISP))
                    col_bgr = cv2.cvtColor(col.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(color_dir / ("%s_color.png" % stem)), col_bgr)
            if (i + 1) % 50 == 0:
                logger.info("inference batch %d / %d", i + 1, len(loader))

    csv_path = out_root / "metrics_per_frame.csv"
    fields = ["name", "epe", "d1_kitti_all_pct", "d1_abs_gt1px_pct", "d3_abs_gt3px_pct"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    mean_epe = float(np.mean([r["epe"] for r in rows])) if rows else 0.0
    mean_d1_kitti = float(np.mean([r["d1_kitti_all_pct"] for r in rows])) if rows else 0.0
    mean_d1_gt1 = float(np.mean([r["d1_abs_gt1px_pct"] for r in rows])) if rows else 0.0
    mean_d3_gt3 = float(np.mean([r["d3_abs_gt3px_pct"] for r in rows])) if rows else 0.0
    summary_lines = [
        "frames: %d" % len(rows),
        "mean_epe: %.6f" % mean_epe,
        "mean_d1_kitti_all_pct: %.6f" % mean_d1_kitti,
        "mean_d1_abs_gt1px_pct: %.6f" % mean_d1_gt1,
        "mean_d3_abs_gt3px_pct: %.6f" % mean_d3_gt3,
        "mean_epe_on_pseudo_gt: %.6f" % mean_epe,
        "mean_d1_pct: %.6f" % mean_d1_kitti,
    ]
    (out_root / "summary_test.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    intr = args.intrinsics or (session_root / "raw" / "camera_intrinsics.json")
    intr = intr.resolve()
    _export_ply(
        foundation_out=fo,
        manifest_test=manifest_test,
        pred_disp_dir=disp_dir,
        out_ply_dir=ply_dir,
        intr_path=intr,
        pixel_subsample=args.pixel_subsample,
        min_disp=args.min_disp,
        max_z=args.max_z,
        baseline_unit=args.baseline_unit,
    )

    logger.info("完成。视差: %s\n指标: %s\nPLY(Z<=%s m): %s", disp_dir, csv_path, args.max_z, ply_dir)


if __name__ == "__main__":
    main()
