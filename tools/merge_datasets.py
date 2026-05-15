#!/usr/bin/env python3
"""
真正合并多个 Orbbec 数据集用于 OpenStereo 训练

用法:
    python tools/merge_datasets.py \
        --datasets /path/to/dataset1 /path/to/dataset2 /path/to/dataset3 \
        --output /path/to/merged_dataset \
        --name merged_name

说明:
    - 每个数据集目录需包含 foundation_out/ 和 splits/openstereo/manifest_{train,val,test}.txt
    - 脚本会真正复制所有图像和视差数据到统一的 foundation_out
    - 生成统一的 manifest 文件（train/val/test 各一个）
    - 自动生成 LightStereo 和 BANet2D 的配置文件
"""

import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def merge_datasets(dataset_paths, output_path, name, symlink=False):
    """真正合并多个数据集的数据和 manifest"""

    output_path = Path(output_path)
    merged_foundation = output_path / "foundation_out"
    manifest_dir = output_path / "splits" / "openstereo"

    # 创建目录结构
    for subdir in ["training/image_2", "training/image_3", "training/disp_occ_0"]:
        (merged_foundation / "train" / subdir).mkdir(parents=True, exist_ok=True)
        (merged_foundation / "val" / subdir).mkdir(parents=True, exist_ok=True)
        (merged_foundation / "test" / subdir).mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = []
    merged_manifests = {"train": [], "val": [], "test": []}
    total_samples = {"train": 0, "val": 0, "test": 0}

    # 处理每个数据集
    for i, dataset_path in enumerate(dataset_paths):
        dataset_path = Path(dataset_path)
        dataset_name = dataset_path.name

        print(f"\n处理数据集 {i+1}/{len(dataset_paths)}: {dataset_name}")

        # 检查数据集结构
        foundation_out = dataset_path / "foundation_out"
        if not foundation_out.exists():
            print(f"  警告: 未找到 foundation_out 目录，跳过")
            continue

        source_manifest_dir = dataset_path / "splits" / "openstereo"
        if not source_manifest_dir.exists():
            print(f"  警告: 未找到 splits/openstereo 目录，跳过")
            continue

        samples = {}

        # 处理每个 split
        for split in ["train", "val", "test"]:
            source_manifest = source_manifest_dir / f"manifest_{split}.txt"
            if not source_manifest.exists():
                print(f"  警告: 未找到 manifest_{split}.txt，跳过")
                continue

            # 读取 manifest
            with open(source_manifest) as f:
                lines = [line.strip() for line in f if line.strip()]

            samples[split] = len(lines)
            total_samples[split] += len(lines)

            print(f"  {split}: 复制 {len(lines)} 个样本...")

            # 复制文件并更新 manifest
            for line in tqdm(lines, desc=f"    {split}", leave=False):
                parts = line.split()
                if len(parts) != 3:
                    continue

                left_rel, right_rel, disp_rel = parts

                # 源文件路径
                left_src = foundation_out / left_rel
                right_src = foundation_out / right_rel
                disp_src = foundation_out / disp_rel

                # 生成唯一的文件名（使用数据集名作为前缀）
                left_name = left_rel.replace("/", "_")
                right_name = right_rel.replace("/", "_")
                disp_name = disp_rel.replace("/", "_")

                # 添加数据集前缀避免冲突
                left_new = f"{dataset_name}_{left_name}"
                right_new = f"{dataset_name}_{right_name}"
                disp_new = f"{dataset_name}_{disp_name}"

                # 目标文件路径
                left_dst = merged_foundation / split / "training" / "image_2" / left_new
                right_dst = merged_foundation / split / "training" / "image_3" / right_new
                disp_dst = merged_foundation / split / "training" / "disp_occ_0" / disp_new

                # 复制或链接文件
                try:
                    if symlink:
                        if not left_dst.exists():
                            left_dst.symlink_to(left_src.absolute())
                        if not right_dst.exists():
                            right_dst.symlink_to(right_src.absolute())
                        if not disp_dst.exists():
                            disp_dst.symlink_to(disp_src.absolute())
                    else:
                        if not left_dst.exists():
                            shutil.copy2(left_src, left_dst)
                        if not right_dst.exists():
                            shutil.copy2(right_src, right_dst)
                        if not disp_dst.exists():
                            shutil.copy2(disp_src, disp_dst)

                    # 添加到合并的 manifest（使用新的相对路径）
                    new_line = f"{split}/training/image_2/{left_new} {split}/training/image_3/{right_new} {split}/training/disp_occ_0/{disp_new}"
                    merged_manifests[split].append(new_line)

                except Exception as e:
                    print(f"    错误: 复制文件失败 {left_src}: {e}")
                    continue

        dataset_info.append({
            "name": dataset_name,
            "path": str(dataset_path.absolute()),
            "samples": samples
        })

    if not dataset_info:
        print("\n错误: 没有有效的数据集")
        return

    # 写入合并的 manifest 文件
    print(f"\n生成合并的 manifest 文件...")
    for split in ["train", "val", "test"]:
        manifest_file = manifest_dir / f"manifest_{split}.txt"
        with open(manifest_file, "w") as f:
            f.write("\n".join(merged_manifests[split]) + "\n")
        print(f"  {split}: {len(merged_manifests[split])} 样本 -> {manifest_file}")

    print(f"\n合并完成:")
    print(f"  训练集: {total_samples['train']} 样本")
    print(f"  验证集: {total_samples['val']} 样本")
    print(f"  测试集: {total_samples['test']} 样本")
    print(f"  数据目录: {merged_foundation}")

    # 生成配置文件
    generate_configs(output_path, name, total_samples)

    # 生成说明文件
    generate_readme(dataset_info, output_path, name, total_samples)


def generate_configs(output_path, name, total_samples):
    """生成 LightStereo 和 BANet2D 的配置文件"""

    # 统一的 DATA_INFOS（只有一个数据集配置）
    foundation_out = str((output_path / "foundation_out").absolute())
    manifest_base = str((output_path / "splits" / "openstereo").absolute())

    data_infos_str = f"""        -   DATASET: OrbbecDataset
            DATA_PATH: {foundation_out}
            DATA_SPLIT: {{
                TRAINING: {manifest_base}/manifest_train.txt,
                EVALUATING: {manifest_base}/manifest_val.txt,
                TESTING: {manifest_base}/manifest_test.txt
            }}
            RETURN_RIGHT_DISP: false"""

    # LightStereo 配置
    lightstereo_config = f"""# LightStereo-M：合并数据集 {name}（共 {total_samples['train']} 训练样本）
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA_CONFIG:
    DATA_INFOS:
{data_infos_str}

    DATA_TRANSFORM:
        TRAINING:
            - {{ NAME: StereoColorJitter, BRIGHTNESS: [ 0.7, 1.3 ], CONTRAST: [ 0.7, 1.3 ], SATURATION: [ 0.7, 1.3 ], HUE: 0.3, ASYMMETRIC_PROB: 0 }}
            - {{ NAME: RandomErase, PROB: 0.5, MAX_TIME: 2, BOUNDS: [ 40, 80 ] }}
            - {{ NAME: RandomSparseScale, SIZE: [ 320, 608 ], MIN_SCALE: 0.2, MAX_SCALE: 0.5, SCALE_PROB: 1.0 }}
            - {{ NAME: RandomCrop, SIZE: [ 320, 608 ] }}
            - {{ NAME: TransposeImage }}
            - {{ NAME: ToTensor }}
            - {{ NAME: NormalizeImage, MEAN: [ 0.485, 0.456, 0.406 ], STD: [ 0.229, 0.224, 0.225 ] }}
        EVALUATING:
            - {{ NAME: RightTopPad, SIZE: [ 480, 640 ] }}
            - {{ NAME: TransposeImage }}
            - {{ NAME: ToTensor }}
            - {{ NAME: NormalizeImage, MEAN: [ 0.485, 0.456, 0.406 ], STD: [ 0.229, 0.224, 0.225 ] }}

MODEL:
    NAME: LightStereo
    MAX_DISP: 192
    EXPANSE_RATIO: 4
    AGGREGATION_BLOCKS: [ 4, 8, 16 ]
    LEFT_ATT: true
    FIND_UNUSED_PARAMETERS: false
    CKPT: -1
    PRETRAINED_MODEL: pretrained/lightstereo/LightStereo-M-SceneFlow-General.pth

OPTIMIZATION:
    FREEZE_BN: false
    SYNC_BN: true
    AMP: false
    BATCH_SIZE_PER_GPU: 4
    NUM_EPOCHS: 200

    OPTIMIZER:
        NAME: AdamW
        LR: &lr 0.0001
        WEIGHT_DECAY: 1.0e-05
        EPS: 1.0e-08

    SCHEDULER:
        NAME: OneCycleLR
        MAX_LR: *lr
        PCT_START: 0.05
        ON_EPOCH: False

    CLIP_GRAD:
        TYPE: value
        CLIP_VALUE: 0.1

EVALUATOR:
    BATCH_SIZE_PER_GPU: 1
    MAX_DISP: 192
    METRIC:
        - d1_all
        - epe
        - thres_1
        - thres_2
        - thres_3

TRAINER:
    EVAL_INTERVAL: 5
    CKPT_SAVE_INTERVAL: 5
    MAX_CKPT_SAVE_NUM: 15
    LOGGER_ITER_INTERVAL: 10
    TRAIN_VISUALIZATION: True
    EVAL_VISUALIZATION: True
"""

    # BANet2D aligned 配置（完全对齐官方）
    banet_aligned_config = f"""# BANet2D aligned：合并数据集 {name}（共 {total_samples['train']} 训练样本）
# 完全对齐官方 BANet/banet-2d 训练 pipeline
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA_CONFIG:
    DATA_INFOS:
{data_infos_str}

    DATA_TRANSFORM:
        TRAINING:
            - {{ NAME: BANetFlowAugmentor }}
            - {{ NAME: TransposeImage }}
            - {{ NAME: ToTensor }}
        EVALUATING:
            - {{ NAME: RightTopPad, SIZE: [ 480, 640 ] }}
            - {{ NAME: TransposeImage }}
            - {{ NAME: ToTensor }}

MODEL:
    NAME: BANet2D
    MAX_DISP: 192
    FIND_UNUSED_PARAMETERS: false
    CKPT: -1
    PRETRAINED_MODEL: pretrained/banet2d/sceneflow.pth

OPTIMIZATION:
    FREEZE_BN: false
    SYNC_BN: true
    AMP: false
    BATCH_SIZE_PER_GPU: 4
    NUM_EPOCHS: 200

    OPTIMIZER:
        NAME: AdamW
        LR: &lr 0.0008
        WEIGHT_DECAY: 1.0e-05
        EPS: 1.0e-08

    SCHEDULER:
        NAME: OneCycleLR
        MAX_LR: *lr
        PCT_START: 0.01
        CYCLE_MOMENTUM: False
        ANNEAL_STRATEGY: linear
        ON_EPOCH: False

    CLIP_GRAD:
        TYPE: norm
        MAX_NORM: 1.0
        NORM_TYPE: 2

EVALUATOR:
    BATCH_SIZE_PER_GPU: 1
    MAX_DISP: 192
    METRIC:
        - d1_all
        - epe
        - thres_1
        - thres_2
        - thres_3

TRAINER:
    EVAL_INTERVAL: 5
    CKPT_SAVE_INTERVAL: 5
    MAX_CKPT_SAVE_NUM: 15
    LOGGER_ITER_INTERVAL: 10
    TRAIN_VISUALIZATION: True
    EVAL_VISUALIZATION: True
"""

    # 保存配置文件
    config_dir = Path("cfgs")

    lightstereo_dir = config_dir / "lightstereo"
    lightstereo_dir.mkdir(parents=True, exist_ok=True)
    lightstereo_file = lightstereo_dir / f"lightstereo_m_{name}.yaml"
    with open(lightstereo_file, "w") as f:
        f.write(lightstereo_config)
    print(f"\n生成配置: {lightstereo_file}")

    banet_dir = config_dir / "banet2d"
    banet_dir.mkdir(parents=True, exist_ok=True)
    banet_file = banet_dir / f"banet2d_{name}.yaml"
    with open(banet_file, "w") as f:
        f.write(banet_config)
    print(f"生成配置: {banet_file}")

    banet_aligned_file = banet_dir / f"banet2d_{name}_aligned.yaml"
    with open(banet_aligned_file, "w") as f:
        f.write(banet_aligned_config)
    print(f"生成配置: {banet_aligned_file}")


def generate_readme(dataset_info, output_path, name, total_samples):
    """生成数据集说明文件"""

    readme_content = f"""# 合并数据集: {name}

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据集组成

"""

    for i, ds in enumerate(dataset_info, 1):
        readme_content += f"{i}. **{ds['name']}**\n"
        readme_content += f"   - 路径: {ds['path']}\n"
        readme_content += f"   - 训练: {ds['samples'].get('train', 0)} 样本\n"
        readme_content += f"   - 验证: {ds['samples'].get('val', 0)} 样本\n"
        readme_content += f"   - 测试: {ds['samples'].get('test', 0)} 样本\n\n"

    readme_content += f"""## 总计

- 训练集: {total_samples['train']} 样本
- 验证集: {total_samples['val']} 样本
- 测试集: {total_samples['test']} 样本

## 数据结构

所有数据已真正合并到统一的 foundation_out 目录：

```
{output_path}/
├── foundation_out/
│   ├── train/training/
│   │   ├── image_2/    # 左图
│   │   ├── image_3/    # 右图
│   │   └── disp_occ_0/ # 视差图
│   ├── val/training/
│   │   ├── image_2/
│   │   ├── image_3/
│   │   └── disp_occ_0/
│   └── test/training/
│       ├── image_2/
│       ├── image_3/
│       └── disp_occ_0/
└── splits/openstereo/
    ├── manifest_train.txt
    ├── manifest_val.txt
    └── manifest_test.txt
```

## 训练命令

### LightStereo-M

```bash
python tools/train.py \\
  --cfg_file cfgs/lightstereo/lightstereo_m_{name}.yaml \\
  --extra_tag my_run
```

### BANet2D aligned（完全对齐官方，推荐）

```bash
python tools/train.py \\
  --cfg_file cfgs/banet2d/banet2d_{name}_aligned.yaml \\
  --extra_tag my_run
```

## 测试命令

```bash
python tools/infer_exp_best_test_full.py \\
  --exp_dir output/OrbbecDataset/<model>/<config>/<extra_tag> \\
  --save_color
```
"""

    readme_file = output_path / "README.md"
    with open(readme_file, "w") as f:
        f.write(readme_content)
    print(f"生成说明: {readme_file}")


def main():
    parser = argparse.ArgumentParser(description="真正合并多个 Orbbec 数据集")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="数据集路径列表")
    parser.add_argument("--output", required=True,
                        help="合并后的输出目录")
    parser.add_argument("--name", required=True,
                        help="合并数据集的名称（用于配置文件命名）")
    parser.add_argument("--symlink", action="store_true",
                        help="使用符号链接而不是复制文件（节省空间）")

    args = parser.parse_args()

    print(f"合并 {len(args.datasets)} 个数据集到: {args.output}")
    if args.symlink:
        print("使用符号链接模式（节省空间）")
    merge_datasets(args.datasets, args.output, args.name, args.symlink)
    print("\n完成！")


if __name__ == "__main__":
    main()
