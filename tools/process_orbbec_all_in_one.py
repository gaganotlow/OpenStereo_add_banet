#!/usr/bin/env python3
"""
Orbbec 数据处理一体化脚本
整合四个步骤：
1. 整理 IR 配对
2. 生成伪视差 GT
3. 生成训练数据布局
4. 导出可视化（视差图 + 点云）
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================================
# 步骤 1: 整理 IR 配对
# ============================================================================
def step1_prepare_ir_pairs(src_dir: Path, out_dir: Path, use_copy: bool = False) -> int:
    """整理 IR 左右图为配对格式"""
    print("\n" + "="*80)
    print("步骤 1: 整理 IR 图像配对")
    print("="*80)

    left_files = sorted(src_dir.glob("left_ir_*.png")) + sorted(src_dir.glob("ir_left_*.png"))
    right_files = sorted(src_dir.glob("right_ir_*.png")) + sorted(src_dir.glob("ir_right_*.png"))

    left_dict = {f.name.replace("ir_left_", "").replace("left_ir_", ""): f for f in left_files}
    right_dict = {f.name.replace("ir_right_", "").replace("right_ir_", ""): f for f in right_files}

    common_keys = sorted(set(left_dict.keys()) & set(right_dict.keys()))

    if not common_keys:
        raise SystemExit(f"未找到配对的 IR 图像: {src_dir}")

    left_out = out_dir / "left"
    right_out = out_dir / "right"
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)

    for idx, key in enumerate(tqdm(common_keys, desc="整理配对")):
        left_src = left_dict[key]
        right_src = right_dict[key]

        name = f"{idx:06d}.png"
        left_dst = left_out / name
        right_dst = right_out / name

        if left_dst.exists() or left_dst.is_symlink():
            left_dst.unlink()
        if right_dst.exists() or right_dst.is_symlink():
            right_dst.unlink()

        if use_copy:
            shutil.copy2(left_src, left_dst)
            shutil.copy2(right_src, right_dst)
        else:
            left_dst.symlink_to(left_src.resolve())
            right_dst.symlink_to(right_src.resolve())

    print(f"✓ 完成: {len(common_keys)} 对图像 -> {out_dir}")
    return len(common_keys)


# ============================================================================
# 步骤 2: 生成伪视差 GT
# ============================================================================
def step2_generate_pseudo_gt(
    input_dir: Path,
    output_dir: Path,
    foundation_stereo_root: Path,
    model_path: Path,
    scale: float = 1.0,
    valid_iters: int = 32,
    save_npy: bool = True,
    use_hiera: bool = False,
) -> dict:
    """使用 FoundationStereo 生成伪视差 GT"""
    print("\n" + "="*80)
    print("步骤 2: 生成伪视差 GT")
    print("="*80)

    # 动态导入 FoundationStereo
    sys.path.insert(0, str(foundation_stereo_root))
    try:
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        from omegaconf import OmegaConf
        import torch
        import imageio
    except ImportError as e:
        raise SystemExit(f"无法导入 FoundationStereo 或其依赖: {e}\n请确保已激活 foundation_stereo 环境")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载配置和模型
    cfg_path = model_path.parent / "cfg.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"未找到配置文件: {cfg_path}")
        
    cfg = OmegaConf.load(str(cfg_path))
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    cfg['scale'] = scale
    cfg['hiera'] = int(use_hiera)
    cfg['valid_iters'] = valid_iters
    
    model_args = OmegaConf.create(cfg)
    model = FoundationStereo(model_args)
    
    # 支持加载直接的 state_dict 或者包含 'model' key 的 checkpoint
    ckpt = torch.load(model_path, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
        
    model.cuda()
    model.eval()

    left_dir = input_dir / "left"
    right_dir = input_dir / "right"
    disp_out = output_dir / "disp_occ_0"
    disp_out.mkdir(parents=True, exist_ok=True)

    if save_npy:
        npy_out = output_dir / "disp_npy"
        npy_out.mkdir(parents=True, exist_ok=True)

    left_files = sorted(left_dir.glob("*.png"))
    disp_stats = []

    with torch.no_grad():
        for lp in tqdm(left_files, desc="生成伪GT"):
            rp = right_dir / lp.name
            if not rp.is_file():
                tqdm.write(f"跳过（缺右图）: {lp.name}")
                continue

            left_img = imageio.imread(str(lp))
            right_img = imageio.imread(str(rp))

            if left_img is None or right_img is None:
                tqdm.write(f"读取失败: {lp.name}")
                continue

            h, w = left_img.shape[:2]
            if scale != 1.0:
                left_img = cv2.resize(left_img, fx=scale, fy=scale, dsize=None)
                right_img = cv2.resize(right_img, fx=scale, fy=scale, dsize=None)

            # 转为 tensor (保持3通道)
            left_t = torch.as_tensor(left_img).cuda().float()[None].permute(0, 3, 1, 2)
            right_t = torch.as_tensor(right_img).cuda().float()[None].permute(0, 3, 1, 2)

            # Pad 到 32 的倍数
            padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
            left_t, right_t = padder.pad(left_t, right_t)

            # 推理（混合精度）
            with torch.cuda.amp.autocast(True):
                disp_pred = model.forward(left_t, right_t, iters=valid_iters, test_mode=True)

            # Unpad
            disp_pred = padder.unpad(disp_pred.float())
            disp_np = disp_pred.squeeze().cpu().numpy()

            if scale != 1.0:
                disp_np = cv2.resize(disp_np, (w, h)) / scale

            # 统计
            valid_mask = disp_np > 0
            if valid_mask.any():
                disp_stats.append({
                    "max": float(disp_np[valid_mask].max()),
                    "p99": float(np.percentile(disp_np[valid_mask], 99)),
                    "mean": float(disp_np[valid_mask].mean()),
                })

            # 保存 KITTI 格式 uint16
            disp_u16 = (disp_np * 256.0).astype(np.uint16)
            cv2.imwrite(str(disp_out / lp.name), disp_u16)

            if save_npy:
                np.save(npy_out / f"{lp.stem}.npy", disp_np)

    # 计算统计
    if disp_stats:
        max_disp = max(s["max"] for s in disp_stats)
        p99_disp = np.percentile([s["p99"] for s in disp_stats], 99)
        mean_disp = np.mean([s["mean"] for s in disp_stats])

        suggested_max_disp = int(np.ceil(p99_disp / 16) * 16)

        print(f"\n视差统计:")
        print(f"  最大值: {max_disp:.2f}")
        print(f"  99分位: {p99_disp:.2f}")
        print(f"  平均值: {mean_disp:.2f}")
        print(f"  建议 MAX_DISP: {suggested_max_disp}")

        stats = {
            "max": max_disp,
            "p99": p99_disp,
            "mean": mean_disp,
            "suggested_max_disp": suggested_max_disp,
        }
    else:
        stats = {}

    print(f"✓ 完成: {len(left_files)} 张视差图 -> {disp_out}")
    return stats


# ============================================================================
# 步骤 3: 生成训练数据布局
# ============================================================================
def step3_prepare_training_layout(
    session_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """生成训练数据布局"""
    print("\n" + "="*80)
    print("步骤 3: 生成训练数据布局")
    print("="*80)

    r = train_ratio + val_ratio + test_ratio
    if abs(r - 1.0) > 1e-6:
        raise SystemExit(f"train+val+test 须为 1，当前 {r}")

    left_dir = session_root / "foundation_stereo_ir_input" / "left"
    right_dir = session_root / "foundation_stereo_ir_input" / "right"
    disp_dir = session_root / "foundation_stereo_pseudo_gt" / "disp_occ_0"
    out_root = session_root / "foundation_out"
    list_dir = session_root / "splits" / "openstereo"

    # 清理旧布局
    if out_root.exists():
        for name in ("train", "val", "test", "training"):
            p = out_root / name
            if p.is_dir():
                shutil.rmtree(p)

    # 收集配对
    pairs = []
    for lp in sorted(left_dir.glob("*.png")):
        rp = right_dir / lp.name
        dp = disp_dir / lp.name
        if rp.is_file() and dp.is_file():
            pairs.append((lp, rp, dp))

    n = len(pairs)
    if n == 0:
        raise SystemExit(f"无完整配对: {left_dir}")

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    if n_val < 1 or n_test < 1:
        raise SystemExit("样本过少：请调低 val/test 比例")

    # 随机划分
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    split_bucket = []
    for j in range(n):
        if j < n_train:
            split_bucket.append("train")
        elif j < n_train + n_val:
            split_bucket.append("val")
        else:
            split_bucket.append("test")

    rows = {"train": [], "val": [], "test": []}

    for j, (lp, rp, dp) in enumerate(tqdm(pairs, desc="生成布局")):
        idx = indices[j]
        split = split_bucket[idx]
        stem = lp.stem
        name_10 = f"{stem}_10.png"

        t_train = out_root / split / "training" / "image_2"
        t_right = out_root / split / "training" / "image_3"
        t_disp = out_root / split / "training" / "disp_occ_0"

        for d in (t_train, t_right, t_disp):
            d.mkdir(parents=True, exist_ok=True)

        for dst, src in (
            (t_train / name_10, lp.resolve()),
            (t_right / name_10, rp.resolve()),
            (t_disp / name_10, dp.resolve()),
        ):
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)

        prefix = f"{split}/training"
        line = f"{prefix}/image_2/{name_10} {prefix}/image_3/{name_10} {prefix}/disp_occ_0/{name_10}\n"
        rows[split].append(line)

    # 写入 manifest
    list_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (list_dir / f"manifest_{split}.txt").write_text("".join(rows[split]))

    # 清理旧文件
    for legacy in ("train.txt", "val.txt", "test.txt"):
        leg = list_dir / legacy
        if leg.exists():
            leg.unlink()

    print(f"✓ 完成: {n} 对 -> train/val/test: {len(rows['train'])}/{len(rows['val'])}/{len(rows['test'])}")
    print(f"  输出: {out_root}")
    print(f"  列表: {list_dir}")

    return {
        "total": n,
        "train": len(rows["train"]),
        "val": len(rows["val"]),
        "test": len(rows["test"]),
    }


# ============================================================================
# 步骤 4: 导出可视化
# ============================================================================
def baseline_to_meters(raw: float, unit: str) -> float:
    """将 JSON 中的基线标量换算为米"""
    u = unit.lower()
    if u == "mm":
        return raw / 1000.0
    if u == "m":
        return raw
    if raw > 10.0:
        return raw / 1000.0
    return raw


def load_intrinsics(path: Path, baseline_unit: str) -> tuple[float, float, float, float, float]:
    """加载相机内参和基线"""
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    k = j.get("depth_intrinsic") or j.get("rgb_intrinsic")
    fx = float(k["fx"])
    fy = float(k["fy"])
    cx = float(k["cx"])
    cy = float(k["cy"])
    sb = j.get("stereo_baseline") or {}

    # 如果没有基线信息，使用默认值（Orbbec Gemini 2 典型值约 50mm）
    if "baseline" not in sb:
        print("警告: camera_intrinsics.json 中未找到 stereo_baseline.baseline")
        print("使用默认基线: 50mm (0.05m)")
        b = 0.05
    else:
        b_raw = float(sb["baseline"])
        json_unit = (sb.get("unit") or "").strip().lower()
        if json_unit in ("mm", "millimeter", "millimeters"):
            b = b_raw / 1000.0
        elif json_unit in ("m", "meter", "meters"):
            b = b_raw
        else:
            b = baseline_to_meters(b_raw, baseline_unit)
    return fx, fy, cx, cy, b


def save_ply_ascii(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """保存 ASCII 格式 PLY"""
    n = xyz.shape[0]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def disp_vis_bgr(disp: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """生成伪彩色视差图"""
    out = np.zeros((*disp.shape, 3), dtype=np.uint8)
    if not valid_mask.any():
        return out
    dv = disp[valid_mask]
    lo, hi = np.percentile(dv, 1), np.percentile(dv, 99)
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.zeros_like(disp, dtype=np.float32)
    norm[valid_mask] = np.clip((disp[valid_mask] - lo) / (hi - lo), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    out[valid_mask] = color[valid_mask]
    return out


def step4_export_visualization(
    session_root: Path,
    intrinsics_path: Path,
    max_frames: int = 0,
    ply_frame_stride: int = 5,
    pixel_subsample: int = 2,
    baseline_unit: str = "mm",
    min_disp: float = 0.0,
    max_z: float = 2.5,
) -> None:
    """导出视差可视化和点云"""
    print("\n" + "="*80)
    print("步骤 4: 导出可视化")
    print("="*80)

    left_dir = session_root / "foundation_stereo_ir_input" / "left"
    right_dir = session_root / "foundation_stereo_ir_input" / "right"
    disp_dir = session_root / "foundation_stereo_pseudo_gt" / "disp_occ_0"
    out_disp = session_root / "foundation_stereo_vis_disparity"
    out_ply = session_root / "foundation_stereo_vis_pointcloud"

    out_disp.mkdir(parents=True, exist_ok=True)
    out_ply.mkdir(parents=True, exist_ok=True)

    fx, fy, cx, cy, baseline = load_intrinsics(intrinsics_path, baseline_unit)
    print(f"相机参数: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} baseline={baseline:.4f}m")

    left_files = sorted(left_dir.glob("*.png"))
    if max_frames > 0:
        left_files = left_files[:max_frames]

    for fi, lp in enumerate(tqdm(left_files, desc="可视化")):
        rp = right_dir / lp.name
        dp = disp_dir / lp.name
        if not rp.is_file() or not dp.is_file():
            continue

        disp_u16 = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        if disp_u16 is None:
            continue
        if disp_u16.ndim != 2:
            disp_u16 = disp_u16[:, :, 0]
        disp = disp_u16.astype(np.float32) / 256.0

        valid = (disp > 0) & np.isfinite(disp)
        valid_ply = valid & (disp >= min_disp)

        # 保存视差伪彩色
        vis_bgr = disp_vis_bgr(disp, valid)
        cv2.imwrite(str(out_disp / lp.name), vis_bgr)

        # 每隔 N 帧生成点云
        if fi % ply_frame_stride != 0:
            continue

        left_bgr = cv2.imread(str(lp))
        if left_bgr is None:
            continue
        if left_bgr.ndim == 2:
            left_bgr = cv2.cvtColor(left_bgr, cv2.COLOR_GRAY2BGR)
        elif left_bgr.shape[2] == 4:
            left_bgr = cv2.cvtColor(left_bgr, cv2.COLOR_BGRA2BGR)

        h, w = disp.shape
        pts_list = []
        col_list = []
        for v in range(0, h, pixel_subsample):
            for u in range(0, w, pixel_subsample):
                d = float(disp[v, u])
                if not valid_ply[v, u]:
                    continue
                Z = (fx * baseline) / d
                if max_z > 0 and Z > max_z:
                    continue
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                pts_list.append((X, Y, Z))
                b, g, r = left_bgr[v, u]
                col_list.append((r, g, b))

        if pts_list:
            xyz = np.array(pts_list, dtype=np.float32)
            rgb = np.array(col_list, dtype=np.uint8)
            ply_path = out_ply / f"{lp.stem}.ply"
            save_ply_ascii(ply_path, xyz, rgb)

    print(f"✓ 完成:")
    print(f"  视差图: {out_disp}")
    print(f"  点云: {out_ply}")


# ============================================================================
# 主函数
# ============================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="Orbbec 数据处理一体化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 通用参数
    p.add_argument(
        "--session_root",
        type=Path,
        default=Path.cwd(),
        help="会话根目录（默认当前目录）",
    )
    p.add_argument(
        "--skip_steps",
        type=str,
        default="",
        help="跳过的步骤，逗号分隔，如 '1,4' 跳过步骤1和4",
    )

    # 步骤1参数
    p.add_argument("--raw_dir", type=Path, default=None, help="原始数据目录（默认 <session>/raw）")
    p.add_argument("--copy", action="store_true", help="复制文件而非符号链接")

    # 步骤2参数
    p.add_argument(
        "--foundation_stereo_root",
        type=Path,
        default=Path("/data2/shendu/code/ruoyu/FoundationStereo"),
        help="FoundationStereo 代码根目录",
    )
    p.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="模型路径（默认 <foundation_stereo_root>/pretrained_models/23-51-11/model_best_bp2.pth）",
    )
    p.add_argument("--scale", type=float, default=1.0, help="图像缩放比例")
    p.add_argument("--valid_iters", type=int, default=32, help="推理迭代次数")
    p.add_argument("--save_npy", action="store_true", help="保存 npy 格式视差")
    p.add_argument("--hiera", action="store_true", help="使用分层推理（高分辨率）")

    # 步骤3参数
    p.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    p.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    p.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    p.add_argument("--seed", type=int, default=42, help="随机种子")

    # 步骤4参数
    p.add_argument("--max_frames", type=int, default=0, help="可视化最大帧数（0=全部）")
    p.add_argument("--ply_frame_stride", type=int, default=5, help="点云帧间隔")
    p.add_argument("--pixel_subsample", type=int, default=2, help="点云像素采样步长")
    p.add_argument("--baseline_unit", choices=("mm", "m", "auto"), default="mm", help="基线单位")
    p.add_argument("--min_disp", type=float, default=0.0, help="点云最小视差")
    p.add_argument("--max_z", type=float, default=2.5, help="点云最大深度（米）")

    args = p.parse_args()

    session_root = args.session_root.resolve()
    skip_steps = set(int(x.strip()) for x in args.skip_steps.split(",") if x.strip())

    print("="*80)
    print("Orbbec 数据处理一体化脚本")
    print("="*80)
    print(f"会话根目录: {session_root}")
    if skip_steps:
        print(f"跳过步骤: {sorted(skip_steps)}")

    # 步骤1: 整理 IR 配对
    if 1 not in skip_steps:
        raw_dir = (args.raw_dir or (session_root / "raw")).resolve()
        ir_input_dir = session_root / "foundation_stereo_ir_input"
        step1_prepare_ir_pairs(raw_dir, ir_input_dir, args.copy)
    else:
        print("\n跳过步骤 1")

    # 步骤2: 生成伪视差 GT
    if 2 not in skip_steps:
        ir_input_dir = session_root / "foundation_stereo_ir_input"
        pseudo_gt_dir = session_root / "foundation_stereo_pseudo_gt"
        foundation_stereo_root = args.foundation_stereo_root.resolve()
        model_path = args.model_path or (
            foundation_stereo_root / "pretrained_models" / "23-51-11" / "model_best_bp2.pth"
        )

        if not model_path.is_file():
            raise SystemExit(f"模型文件不存在: {model_path}")

        stats = step2_generate_pseudo_gt(
            ir_input_dir,
            pseudo_gt_dir,
            foundation_stereo_root,
            model_path,
            args.scale,
            args.valid_iters,
            args.save_npy,
            args.hiera,
        )

        # 保存统计信息
        if stats:
            stats_file = session_root / "disparity_stats.json"
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"  统计信息已保存: {stats_file}")
    else:
        print("\n跳过步骤 2")

    # 步骤3: 生成训练数据布局
    if 3 not in skip_steps:
        layout_stats = step3_prepare_training_layout(
            session_root,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed,
        )
    else:
        print("\n跳过步骤 3")

    # 步骤4: 导出可视化
    if 4 not in skip_steps:
        intrinsics_path = session_root / "raw" / "camera_intrinsics.json"
        if not intrinsics_path.is_file():
            print(f"\n警告: 未找到相机内参文件 {intrinsics_path}，跳过可视化")
        else:
            step4_export_visualization(
                session_root,
                intrinsics_path,
                args.max_frames,
                args.ply_frame_stride,
                args.pixel_subsample,
                args.baseline_unit,
                args.min_disp,
                args.max_z,
            )
    else:
        print("\n跳过步骤 4")

    print("\n" + "="*80)
    print("全部完成！")
    print("="*80)
    print(f"训练数据: {session_root / 'foundation_out'}")
    print(f"数据列表: {session_root / 'splits' / 'openstereo'}")
    if 4 not in skip_steps:
        print(f"可视化: {session_root / 'foundation_stereo_vis_disparity'}")
        print(f"点云: {session_root / 'foundation_stereo_vis_pointcloud'}")


if __name__ == "__main__":
    main()
