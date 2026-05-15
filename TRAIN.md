# OpenStereo 训练与测试

## 环境

```bash
source /data2/shendu/anaconda3/etc/profile.d/conda.sh
conda activate openstereo
cd /data2/shendu/code/ruoyu/openstereo_add_banet/OpenStereo
```

## 合并新数据集（推荐）

当有新数据集时，使用脚本**真正合并**数据并生成配置：

```bash
python tools/merge_datasets.py \
  --datasets /path/to/dataset1 /path/to/dataset2 /path/to/dataset3 \
  --output /path/to/merged_output \
  --name merged_name \
  --symlink  # 可选：使用符号链接节省空间
```

**示例：合并现有的两个数据集**

```bash
python tools/merge_datasets.py \
  --datasets \
    /data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_capture_20260509_140955 \
    /data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_capture_20260513_103730 \
  --output /data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_merged_real \
  --name orbbec_merged_real_640x480 \
  --symlink
```

脚本会自动：
- **真正合并数据**：所有图像和视差复制/链接到统一的 `foundation_out` 目录
- 生成统一的 manifest 文件（`manifest_train.txt`、`manifest_val.txt`、`manifest_test.txt`）
- 生成 3 个配置文件：`lightstereo_m_{name}.yaml`、`banet2d_{name}.yaml`、`banet2d_{name}_aligned.yaml`
- 在输出目录生成 `README.md` 说明文件
- 配置文件只有一个 DATA_INFOS，测试时可一次性在所有数据上运行

**合并后的数据结构：**
```
merged_output/
├── foundation_out/
│   ├── train/training/{image_2,image_3,disp_occ_0}/
│   ├── val/training/{image_2,image_3,disp_occ_0}/
│   └── test/training/{image_2,image_3,disp_occ_0}/
└── splits/openstereo/
    ├── manifest_train.txt
    ├── manifest_val.txt
    └── manifest_test.txt
```

## 训练

### 使用合并数据集

```bash
# LightStereo-M
python tools/train.py \
  --cfg_file cfgs/lightstereo/lightstereo_m_orbbec_merged_real_640x480.yaml \
  --extra_tag my_run

# BANet2D（训练 pipeline 完全对齐官方 BANet/banet-2d）
python tools/train.py \
  --cfg_file cfgs/banet2d/banet2d_orbbec_merged_real_640x480_aligned.yaml \
  --extra_tag aligned_run
```

> BANet2D 使用 `aligned` 版本，完全对齐官方 `BANet/banet-2d` 的训练 pipeline：
> - 数据增广：`BANetFlowAugmentor`（复刻官方 `FlowAugmentor`）
> - 优化器：AdamW(lr=8e-4, wd=1e-5, eps=1e-8)
> - 调度器：OneCycleLR(pct_start=0.01, cycle_momentum=False, anneal_strategy=linear)
> - 梯度裁剪：clip_grad_norm_=1.0
> - 输入：0-255 float（无 ImageNet Normalize）
> 
> 详细差异见 `docs/BANet_OpenStereo_vs_Official.md`。

### 多卡训练（8×A800 推荐）

```bash
# BANet2D aligned（完全对齐官方 pipeline）
torchrun --nproc_per_node=8 tools/train.py --dist_mode \
  --cfg_file cfgs/banet2d/banet2d_orbbec_merged_real_640x480_aligned.yaml \
  --extra_tag aligned_run

# LightStereo-M
torchrun --nproc_per_node=8 tools/train.py --dist_mode \
  --cfg_file cfgs/lightstereo/lightstereo_m_orbbec_merged_real_640x480.yaml \
  --extra_tag aligned_run
```

**关键提示**

- `BATCH_SIZE_PER_GPU=4`，8 卡时全局 batch=32（官方单机是 16；若要严格对齐 batch，改成 2 即可）。
- `aligned` yaml 的 `LR=8e-4` 是官方"重头训 SceneFlow"的设置；若是从 `sceneflow.pth`
  微调到 Orbbec，建议把 `OPTIMIZER.LR`（`&lr` 锚点）改成 `1e-4`。
- 输出路径现在是 `output/OrbbecDataset/...`（之前是 `output/KittiDataset/...`，
  yaml 中 `DATASET` 已改名为 `OrbbecDataset`；旧目录下的 ckpt 不会被自动复用）。

**参数说明：**
- `--extra_tag`：实验子目录名，用 `debug` 可避免冲突
- `--cover_old_exp`：覆盖同名实验目录
- `--fix_random_seed`：固定种子
- `--workers N`：dataloader worker 数（默认 8）

权重与日志：`output/<DATASET 名>/<MODEL.NAME>/<yaml 文件名>/<extra_tag>/`

## 测试

使用 `tools/infer_exp_best_test_full.py` 自动选择最佳 epoch 并推理测试集：

```bash
# LightStereo-M
python tools/infer_exp_best_test_full.py \
  --exp_dir output/OrbbecDataset/LightStereo/lightstereo_m_orbbec_merged_640x480/run_260513 \
  --save_color

# BANet2D
python tools/infer_exp_best_test_full.py \
  --exp_dir output/OrbbecDataset/BANet2D/banet2d_orbbec_merged_640x480_aligned/run_260513 \
  --save_color
```

**可选参数：**
- `--force_epoch 199`：强制使用指定 epoch
- `--cfg_file /path/to/config.yaml`：指定配置文件（默认使用实验目录下的 yaml）
- `--max-z 2.5`：深度裁切阈值（默认 2.5m，设为 0 不裁切）
- `--out_dir /path`：自定义输出目录

**输出内容：**
- `disp_png/`：KITTI uint16 视差图
- `ply/`：点云（左相机系，默认去掉 Z>2.5m）
- `metrics_per_frame.csv`：每帧指标
- `summary_test.txt`：汇总指标（EPE、D1、D3 等）

## 测试结果（orbbec_merged_real，522 样本）

**BANet2D (epoch 190):**
- EPE: 0.561
- D1 (%): 1.73
- D1 (>1px, %): 9.82
- D3 (>3px, %): 2.48

**LightStereo-M (epoch 175):**
- EPE: 0.816
- D1 (%): 2.51
- D1 (>1px, %): 17.25
- D3 (>3px, %): 3.46

**结论：** BANet2D 在所有指标上都显著优于 LightStereo-M（EPE 低 31%，D1 低 31%）

## 使用自定义测试集

如果需要在不同的测试集上评估（而不是训练时使用的数据集）：

**方法1：指定配置文件（推荐）**

创建一个临时配置文件，指向目标测试集：

```bash
# 复制训练配置
cp cfgs/banet2d/banet2d_orbbec_merged_real_640x480_aligned.yaml /tmp/test_config.yaml

# 修改配置中的 DATA_PATH 和 TESTING manifest 路径
# 然后测试
python tools/infer_exp_best_test_full.py \
  --exp_dir output/OrbbecDataset/BANet2D/banet2d_orbbec_merged_real_640x480_aligned/aligned_run \
  --cfg_file /tmp/test_config.yaml \
  --save_color
```

**方法2：使用原始数据集测试**

如果要在原始的单个数据集上测试（而不是合并后的）：

```bash
# 使用原始数据集的配置文件
python tools/infer_exp_best_test_full.py \
  --exp_dir output/OrbbecDataset/BANet2D/banet2d_orbbec_merged_640x480_aligned/run_260513 \
  --cfg_file cfgs/banet2d/banet2d_orbbec_capture_20260509_640x480_aligned.yaml \
  --save_color
```

注意：
- 配置文件中的 `DATA_PATH` 指向 `foundation_out` 目录
- `TESTING` 字段指向测试集的 manifest 文件
- 测试时会自动使用配置文件中指定的测试集
