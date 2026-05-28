#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断 BANet2D 预测 vs FoundationStereo 伪 GT 的差异。

回答三个问题：
- Q1 总 gap：mean EPE / D1@1px / D1@3px（相对伪 GT）
- Q2 gap 的空间分布：按 7 类桶分别统计（edge/flat/near/far/highlight/lowtex/fine）
- Q3 gap 与 teacher 不确定性的相关性：把像素按熵分箱画 mean EPE 折线

依赖产物：
- BANet2D 推理输出：disp_png/<stem>.png (KITTI uint16, disp*256)
- 伪 GT：foundation_out/<gt_rel>.png （manifest 第 3 列）
- FoundationStereo 熵图：tools/extract_foundation_entropy.py 输出，<entropy_dir>/<stem>.npy
- 左图：foundation_out/<left_rel>.png （manifest 第 1 列）

输出：
- bucket_table.csv         每个桶 (frames, pixels, pct, EPE, D1@1px, D1@3px)
- entropy_error_curve.png  熵分箱 vs mean EPE 折线（含 Spearman / Pearson）
- vis_topk/<stem>.png      mean EPE 最大的 K 张，左图 + error heatmap 拼图
- report.md                文字总结，自动判定决策树分支
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# 桶定义参数（一处改全局生效）
EDGE_GRAD_THR = 1.0          # |∇disp_gt| > 此值 视为边界
EDGE_DILATE = 2              # 边界膨胀像素数
NEAR_DISP = 64.0             # disp >= 此值 视为近景
FAR_DISP = 32.0              # disp < 此值（且 > 0）视为远景
HIGHLIGHT_GRAY = 230         # 灰度 > 此值
HIGHLIGHT_VAR = 50           # 且 局部方差 < 此值 视为高反/过曝
LOWTEX_VAR = 30              # 局部方差 < 此值 视为弱纹理
LOCAL_WIN = 11               # 局部统计窗口
FINE_EDGE_DENSITY = 0.3      # 边界密度 > 此值的非 edge 区视为细小结构

D1_THR = 1.0                 # D1@1px 阈值
D3_THR = 3.0                 # D1@3px 阈值

BUCKETS = ["all", "edge", "flat", "near", "far", "highlight", "lowtex", "fine"]


def _read_disp_u16(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32) / 256.0


def _read_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _compute_buckets(left_rgb: np.ndarray, disp_gt: np.ndarray) -> dict:
    """返回 {bucket_name: bool mask (H, W)}。"""
    H, W = disp_gt.shape
    valid = disp_gt > 0

    # 1. edge / flat (基于 disp_gt 梯度)
    gx = np.abs(np.diff(disp_gt, axis=1, append=disp_gt[:, -1:]))
    gy = np.abs(np.diff(disp_gt, axis=0, append=disp_gt[-1:, :]))
    grad_mag = gx + gy
    edge_raw = (grad_mag > EDGE_GRAD_THR) & valid
    if EDGE_DILATE > 0:
        k = np.ones((2 * EDGE_DILATE + 1, 2 * EDGE_DILATE + 1), np.uint8)
        edge = cv2.dilate(edge_raw.astype(np.uint8), k).astype(bool)
    else:
        edge = edge_raw.copy()
    flat = (~edge) & valid

    # 2. near / far
    near = (disp_gt >= NEAR_DISP) & valid
    far = (disp_gt > 0) & (disp_gt < FAR_DISP)

    # 3. highlight / lowtex
    gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.boxFilter(gray, ddepth=-1, ksize=(LOCAL_WIN, LOCAL_WIN))
    mu2 = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(LOCAL_WIN, LOCAL_WIN))
    var_local = np.maximum(mu2 - mu * mu, 0.0)
    highlight = (gray > HIGHLIGHT_GRAY) & (var_local < HIGHLIGHT_VAR) & valid
    lowtex = (var_local < LOWTEX_VAR) & (~highlight) & valid

    # 4. fine: 边界密度高的非 edge 区域 → 细丝/小物体内部
    edge_density = cv2.boxFilter(
        edge_raw.astype(np.float32), ddepth=-1, ksize=(LOCAL_WIN, LOCAL_WIN)
    )
    fine = (edge_density > FINE_EDGE_DENSITY) & flat

    return {
        "all": valid,
        "edge": edge,
        "flat": flat,
        "near": near,
        "far": far,
        "highlight": highlight,
        "lowtex": lowtex,
        "fine": fine,
    }


def _accumulate(stats: dict, frame_buckets: dict, err: np.ndarray) -> None:
    """把单帧的桶统计累加到全局 stats。"""
    d1 = (err > D1_THR).astype(np.float32)
    d3 = (err > D3_THR).astype(np.float32)
    for name, mask in frame_buckets.items():
        n = int(mask.sum())
        if n == 0:
            continue
        bucket = stats[name]
        bucket["frames"] += 1
        bucket["pixels"] += n
        bucket["sum_epe"] += float(err[mask].sum())
        bucket["sum_d1"] += float(d1[mask].sum())
        bucket["sum_d3"] += float(d3[mask].sum())


def _entropy_to_full(entropy_q: np.ndarray, target_hw: tuple) -> np.ndarray:
    """(H/4, W/4) -> (H, W) 双线性上采。"""
    return cv2.resize(entropy_q.astype(np.float32), (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)


def _error_heatmap_overlay(left_rgb: np.ndarray, err: np.ndarray, valid: np.ndarray, max_err: float = 5.0) -> np.ndarray:
    """error heatmap 叠加到左图，输出 RGB uint8。"""
    e = np.clip(err / max_err, 0.0, 1.0)
    e_u8 = (e * 255).astype(np.uint8)
    heat = cv2.applyColorMap(e_u8, cv2.COLORMAP_JET)  # BGR
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * left_rgb.astype(np.float32) + 0.5 * heat.astype(np.float32))
    overlay[~valid] = left_rgb[~valid]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _save_topk_vis(out_dir: Path, frame_records: list, topk: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(frame_records, key=lambda r: r["epe"], reverse=True)[:topk]
    for rec in ranked:
        left = rec["_left_rgb"]
        err = rec["_err"]
        valid = rec["_valid"]
        overlay = _error_heatmap_overlay(left, err, valid)
        h, w = left.shape[:2]
        pad = np.full((h, 8, 3), 255, np.uint8)
        canvas = np.concatenate([left, pad, overlay], axis=1)
        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{rec['stem']}.png"), bgr)


def _bin_entropy_error(entropy_all: np.ndarray, err_all: np.ndarray, n_bins: int = 10) -> dict:
    """按熵分位数分箱，统计每箱 mean EPE / count。"""
    if entropy_all.size == 0:
        return {"bins": [], "epe": [], "count": [], "pearson": float("nan"), "spearman": float("nan")}
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(entropy_all, qs)
    centers = 0.5 * (edges[:-1] + edges[1:])
    epe_means = []
    counts = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            m = (entropy_all >= lo) & (entropy_all <= hi)
        else:
            m = (entropy_all >= lo) & (entropy_all < hi)
        c = int(m.sum())
        counts.append(c)
        epe_means.append(float(err_all[m].mean()) if c > 0 else float("nan"))

    # 相关系数（用子采样避免内存爆炸）
    n = entropy_all.size
    if n > 5_000_000:
        idx = np.random.default_rng(0).choice(n, 5_000_000, replace=False)
        e = entropy_all[idx]
        r = err_all[idx]
    else:
        e, r = entropy_all, err_all
    pearson = float(np.corrcoef(e, r)[0, 1])
    # Spearman: 对秩做 Pearson
    er = np.argsort(np.argsort(e))
    rr = np.argsort(np.argsort(r))
    spearman = float(np.corrcoef(er, rr)[0, 1])

    return {
        "bins": centers.tolist(),
        "epe": epe_means,
        "count": counts,
        "pearson": pearson,
        "spearman": spearman,
    }


def _plot_entropy_curve(curve: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(curve["bins"], curve["epe"], "o-", color="C0", label="mean EPE")
    ax1.set_xlabel("FoundationStereo init prob entropy (分位数中心)")
    ax1.set_ylabel("mean EPE (px)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.bar(curve["bins"], curve["count"], width=(curve["bins"][1] - curve["bins"][0]) * 0.6 if len(curve["bins"]) > 1 else 0.1,
            alpha=0.2, color="gray", label="pixel count")
    ax2.set_ylabel("pixel count", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    title = f"entropy vs EPE  |  pearson={curve['pearson']:.3f}  spearman={curve['spearman']:.3f}"
    plt.title(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_bucket_csv(stats: dict, csv_path: Path) -> list:
    rows = []
    total_pix = max(stats["all"]["pixels"], 1)
    for name in BUCKETS:
        b = stats[name]
        pix = b["pixels"]
        epe = b["sum_epe"] / pix if pix else 0.0
        d1 = 100.0 * b["sum_d1"] / pix if pix else 0.0
        d3 = 100.0 * b["sum_d3"] / pix if pix else 0.0
        rows.append({
            "bucket": name,
            "frames": b["frames"],
            "pixels": pix,
            "pct": 100.0 * pix / total_pix,
            "epe": epe,
            "d1_pct": d1,
            "d3_pct": d3,
        })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["bucket", "frames", "pixels", "pct", "epe", "d1_pct", "d3_pct"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "pct": f"{r['pct']:.3f}", "epe": f"{r['epe']:.4f}",
                        "d1_pct": f"{r['d1_pct']:.3f}", "d3_pct": f"{r['d3_pct']:.3f}"})
    return rows


def _decide_branch(rows: list, curve: dict) -> tuple:
    """根据 bucket 表 + 熵曲线判定决策树分支。返回 (branch_key, reason)。"""
    by = {r["bucket"]: r for r in rows}
    overall_epe = by["all"]["epe"]
    if overall_epe < 0.3:
        return ("A_gap_small", f"总 EPE={overall_epe:.3f}px < 0.3, 不值得复杂蒸馏")

    # 计算各桶相对 all 的 EPE 倍数
    def ratio(name):
        if by[name]["pixels"] == 0:
            return 0.0
        return by[name]["epe"] / max(overall_epe, 1e-6)

    edge_r = ratio("edge")
    high_r = ratio("highlight")
    lowtex_r = ratio("lowtex")
    fine_r = ratio("fine")
    far_r = ratio("far")

    spearman = curve.get("spearman", float("nan"))

    # 判定优先级
    if high_r > 1.5 and spearman > 0.3:
        return ("C_teacher_bottleneck",
                f"highlight 桶 EPE ratio={high_r:.2f} 且 entropy-error spearman={spearman:.2f}>0.3，"
                "teacher 自身是瓶颈，蒸馏帮不上，考虑半监督一致性 / 多 teacher 集成 / 主动光辅助")
    if edge_r > 1.5 and (high_r <= 1.5 or spearman <= 0.3):
        return ("B_edge_dominant",
                f"edge 桶 EPE ratio={edge_r:.2f}, 主要 gap 在视差边界 → 方案 1（概率分布 KL 蒸馏）对症")
    if fine_r > 1.5 or far_r > 1.5:
        return ("B_fine_or_far",
                f"fine={fine_r:.2f} far={far_r:.2f} 占主导 → 方案 1 + 方案 3（特征蒸馏）")
    if all(r < 1.3 for r in (edge_r, high_r, lowtex_r, fine_r, far_r)):
        if spearman < 0.2:
            return ("D_student_capacity",
                    f"各桶 EPE ratio 都 < 1.3，spearman={spearman:.2f} 弱相关 → 学生容量瓶颈，"
                    "方案 1 + 方案 3 组合")
    return ("E_mixed", "无明显主导桶或与决策树不匹配，需人工查看 bucket_table 与 entropy_curve")


def _write_report(report_path: Path, rows: list, curve: dict, branch: tuple, args, topk_files: list) -> None:
    by = {r["bucket"]: r for r in rows}
    lines = []
    lines.append("# BANet2D vs FoundationStereo 伪 GT 诊断报告")
    lines.append("")
    lines.append(f"- pred_dir: `{args.pred_dir}`")
    lines.append(f"- data_root: `{args.data_root}`")
    lines.append(f"- entropy_dir: `{args.entropy_dir}`")
    lines.append(f"- manifest: `{args.manifest}`")
    lines.append(f"- 总帧数: {by['all']['frames']}")
    lines.append("")
    lines.append("## Q1 — 全局 gap")
    lines.append("")
    lines.append(f"- mean EPE: **{by['all']['epe']:.4f} px**")
    lines.append(f"- D1 (>1px): **{by['all']['d1_pct']:.2f}%**")
    lines.append(f"- D1 (>3px): **{by['all']['d3_pct']:.2f}%**")
    lines.append("")
    lines.append("## Q2 — 分桶 EPE")
    lines.append("")
    lines.append("| 桶 | 占比 | EPE | EPE/all | D1@1px | D1@3px |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    overall = max(by["all"]["epe"], 1e-6)
    for name in BUCKETS:
        r = by[name]
        ratio = r["epe"] / overall
        lines.append(
            f"| {name} | {r['pct']:.2f}% | {r['epe']:.4f} | ×{ratio:.2f} | "
            f"{r['d1_pct']:.2f}% | {r['d3_pct']:.2f}% |"
        )
    lines.append("")
    lines.append("## Q3 — 熵-error 相关性")
    lines.append("")
    lines.append(f"- Pearson  r = **{curve['pearson']:.3f}**")
    lines.append(f"- Spearman r = **{curve['spearman']:.3f}**")
    lines.append(f"- 曲线见 `entropy_error_curve.png`")
    lines.append("")
    lines.append("## 决策树判定")
    lines.append("")
    lines.append(f"- 分支：**{branch[0]}**")
    lines.append(f"- 理由：{branch[1]}")
    lines.append("")
    if topk_files:
        lines.append("## EPE 最大的 N 张（vis_topk/）")
        lines.append("")
        for p in topk_files:
            lines.append(f"- `{p}`")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("注：")
    lines.append("- 本诊断的 GT 是 FoundationStereo 的伪 GT，EPE 反映的是 BANet2D 相对 teacher 的拟合误差。")
    lines.append("- 熵来自 FS 的 **init prob**，是 teacher 在 GRU 迭代前的不确定性代理。")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_manifest(manifest: Path, data_root: Path):
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        left_rel, _, gt_rel = parts[:3]
        rows.append((data_root / left_rel, data_root / gt_rel, Path(left_rel).stem))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="BANet2D vs FoundationStereo 伪 GT 诊断")
    ap.add_argument("--pred_dir", type=Path, required=True, help="BANet2D 推理输出 disp_png/")
    ap.add_argument("--data_root", type=Path, required=True, help="foundation_out 根目录（用于解析 manifest 相对路径）")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--entropy_dir", type=Path, required=True, help="P1 输出目录")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--topk_vis", type=int, default=5)
    ap.add_argument("--max_disp_valid", type=float, default=192.0, help="disp_gt < 此值才计入")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--entropy_sample_per_frame",
        type=int,
        default=20000,
        help="每帧随机采样多少像素累积到全局 entropy-error 数组（防止内存爆炸）",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = _parse_manifest(args.manifest, args.data_root)
    if args.limit > 0:
        manifest_rows = manifest_rows[: args.limit]
    print(f"[manifest] {len(manifest_rows)} frames")

    stats = {b: {"frames": 0, "pixels": 0, "sum_epe": 0.0, "sum_d1": 0.0, "sum_d3": 0.0} for b in BUCKETS}
    frame_records = []

    entropy_buf, err_buf = [], []
    rng = np.random.default_rng(0)
    n_proc = 0
    n_skip = 0

    for i, (left_path, gt_path, stem) in enumerate(manifest_rows):
        pred_path = args.pred_dir / f"{stem}.png"
        ent_path = args.entropy_dir / f"{stem}.npy"
        if not pred_path.is_file() or not gt_path.is_file() or not ent_path.is_file() or not left_path.is_file():
            n_skip += 1
            continue
        disp_pred = _read_disp_u16(pred_path)
        disp_gt = _read_disp_u16(gt_path)
        left_rgb = _read_image_rgb(left_path)
        entropy_q = np.load(ent_path).astype(np.float32)
        if disp_pred is None or disp_gt is None or left_rgb is None:
            n_skip += 1
            continue

        # 对齐尺寸（以 disp_gt 为准）
        H, W = disp_gt.shape
        if disp_pred.shape != (H, W):
            disp_pred = cv2.resize(disp_pred, (W, H), interpolation=cv2.INTER_NEAREST)
        if left_rgb.shape[:2] != (H, W):
            left_rgb = cv2.resize(left_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        valid = (disp_gt > 0) & (disp_gt < args.max_disp_valid)
        err = np.abs(disp_pred - disp_gt)

        buckets = _compute_buckets(left_rgb, disp_gt)
        # 与 valid 取交集
        buckets = {k: (v & valid) for k, v in buckets.items()}
        _accumulate(stats, buckets, err)

        # entropy-error 采样
        entropy_full = _entropy_to_full(entropy_q, (H, W))
        v_idx = np.flatnonzero(valid.reshape(-1))
        if v_idx.size > 0:
            k = min(args.entropy_sample_per_frame, v_idx.size)
            sel = rng.choice(v_idx, k, replace=False)
            entropy_buf.append(entropy_full.reshape(-1)[sel])
            err_buf.append(err.reshape(-1)[sel])

        # 帧级 EPE 用于 topk 可视化
        if valid.any():
            frame_records.append({
                "stem": stem,
                "epe": float(err[valid].mean()),
                "_left_rgb": left_rgb,
                "_err": err,
                "_valid": valid,
            })
        n_proc += 1
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(manifest_rows)}] processed={n_proc} skip={n_skip}")

    if n_proc == 0:
        print("[error] 没有有效帧，请检查 pred/gt/entropy 目录")
        sys.exit(1)

    print(f"[done] processed={n_proc} skip={n_skip}")

    # 1. 桶表
    csv_path = args.out_dir / "bucket_table.csv"
    rows = _write_bucket_csv(stats, csv_path)
    print(f"[out] {csv_path}")

    # 2. 熵曲线
    entropy_all = np.concatenate(entropy_buf) if entropy_buf else np.zeros(0, np.float32)
    err_all = np.concatenate(err_buf) if err_buf else np.zeros(0, np.float32)
    curve = _bin_entropy_error(entropy_all, err_all, n_bins=10)
    if entropy_all.size > 0:
        plot_path = args.out_dir / "entropy_error_curve.png"
        _plot_entropy_curve(curve, plot_path)
        print(f"[out] {plot_path}")

    # 3. topk 可视化
    vis_dir = args.out_dir / "vis_topk"
    _save_topk_vis(vis_dir, frame_records, args.topk_vis)
    topk_files = [
        f"vis_topk/{r['stem']}.png"
        for r in sorted(frame_records, key=lambda r: r["epe"], reverse=True)[: args.topk_vis]
    ]
    print(f"[out] {vis_dir}/")

    # 4. 决策树 + 报告
    branch = _decide_branch(rows, curve)
    report_path = args.out_dir / "report.md"
    _write_report(report_path, rows, curve, branch, args, topk_files)
    print(f"[out] {report_path}")
    print(f"[branch] {branch[0]} - {branch[1]}")


if __name__ == "__main__":
    main()
