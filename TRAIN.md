# OpenStereo 训练与测试

## 环境

```bash
source /data2/shendu/anaconda3/etc/profile.d/conda.sh
conda activate openstereo
cd /data2/shendu/code/ruoyu/openstereo_add_banet/OpenStereo
```

## 数据准备：原始 Orbbec 采集 → 训练就绪

新采集的数据集（顶层 `*.png` + `camera_intrinsics.json`，无 `foundation_out/`）需要先用
FoundationStereo 生成伪视差 GT 并切分 train/val/test。每个数据集目录里都有
`scripts/process_orbbec_all_in_one.py`（若没有，从已有数据集拷贝过去）。

```bash
source /data2/shendu/anaconda3/etc/profile.d/conda.sh && conda activate foundation_stereo

# 单卡处理（raw 图在数据集顶层时显式传 --raw_dir 指向同一目录）
CUDA_VISIBLE_DEVICES=1 python /path/to/dataset/process_orbbec_all_in_one.py \
  --session_root /path/to/dataset \
  --raw_dir /path/to/dataset \
  --skip_steps 4   # 跳过点云可视化，仅生成训练所需的 foundation_out/ 和 splits/
```

输出：
- `foundation_stereo_ir_input/{left,right}/`：配对的左右 IR
- `foundation_stereo_pseudo_gt/disp_occ_0/`：KITTI uint16 视差（disp*256）
- `foundation_out/{train,val,test}/training/{image_2,image_3,disp_occ_0}/`：训练布局
- `splits/openstereo/manifest_{train,val,test}.txt`：8:1:1 随机划分
- `disparity_stats.json`：包含建议的 MAX_DISP

两张数据集可并行处理（不同 `CUDA_VISIBLE_DEVICES`）。每个 1700 张图约 12–15 分钟。

## 合并多个数据集（推荐）

```bash
python tools/merge_datasets.py \
  --datasets /path/to/dataset1 /path/to/dataset2 /path/to/dataset3 /path/to/dataset4 \
  --output /path/to/merged_output \
  --name merged_name \
  --symlink  # 推荐：用符号链接避免重复占用磁盘
```

脚本会：
- 把所有 `foundation_out` 软链/复制到统一目录，文件名加数据集前缀避免冲突
- 生成统一的 `manifest_train.txt` / `manifest_val.txt` / `manifest_test.txt`
- 自动生成 `cfgs/lightstereo/lightstereo_m_{name}.yaml`（**忽略，已不用**）
- **注意**：脚本里 `banet_config`（非 aligned）存在一个 NameError bug。

**新工作流（推荐）**：不再为每个新合并数据集生成新 yaml。统一使用
`cfgs/banet2d/banet2d_orin.yaml` 和 `cfgs/lightstereo/lightstereo_m_orin.yaml`
作为唯一的"线上/orin 部署"配置；新数据集合并后，只改 yaml 里的 `DATA_PATH`
和三个 `DATA_SPLIT` 路径即可，不要新增 yaml 文件。脚本自动生成的
`cfgs/lightstereo/lightstereo_m_{name}.yaml` 可以直接删掉。

### 当前合并数据集：`orbbec_merged_real_5ds`（5 个数据集，2026-05-19）

```text
orbbec_capture_20260509_140955         : 1925 / 241  / 240   (train/val/test)
orbbec_capture_20260513_103730         : 2262 / 283  / 282
orbbec_capture_20260514_110538         : 1388 / 174  / 173
orbbec_capture_20260519_101407_hafo    : 1391 / 174  / 174
orbbec_capture_20260519_102145_xiandai : 1343 / 168  / 168
─────────────────────────────────────────────────────────────
合计                                     : 8309 / 1040 / 1037
```

五个采集的 `camera_intrinsics.json` 完全一致（fx=fy≈367.35, cx≈321.93, cy≈241.06,
baseline=94.97 mm，分辨率 640×480），合并安全。

合并命令（已执行）：

```bash
python tools/merge_datasets.py \
  --datasets \
    /data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_capture_20260509_140955 \
    /data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_capture_20260513_103730 \
    /data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_capture_20260514_110538 \
    /data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_capture_20260519_101407_hafo \
    /data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_capture_20260519_102145_xiandai \
  --output /data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_merged_real_5ds \
  --name orbbec_merged_real_5ds_640x480 \
  --symlink
```

使用的配置（**统一 orin 版**，直接复用，未来换数据集只改这两个 yaml 内的路径）：
- `cfgs/lightstereo/lightstereo_m_orin.yaml`
- `cfgs/banet2d/banet2d_orin.yaml`（已含 `CROP_SIZE: [320, 608]`）

## 训练（5 个数据集合并集）

### 多卡训练（推荐）

GPU 占用查看：`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`

```bash
# BANet2D aligned —— 4 卡（GPU 0,1,2,3）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 \
  tools/train.py --dist_mode \
  --cfg_file cfgs/banet2d/banet2d_orin.yaml \
  --extra_tag run_5ds_260519

# LightStereo-M —— 2 卡（GPU 5,6）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=2 --master_port=29502 \
  tools/train.py --dist_mode \
  --cfg_file cfgs/lightstereo/lightstereo_m_orin.yaml \
  --extra_tag run_5ds_260519
```

> BANet2D 使用 `aligned` 版本，完全对齐官方 `BANet/banet-2d` 的训练 pipeline：
> - 数据增广：`BANetFlowAugmentor`（复刻官方 `FlowAugmentor`，**必须配置 `CROP_SIZE`**，
>   对 480×640 输入推荐 `[320, 608]`；官方默认 `[256, 512]`）
> - 优化器：AdamW(lr=8e-4, wd=1e-5, eps=1e-8)
> - 调度器：OneCycleLR(pct_start=0.01, cycle_momentum=False, anneal_strategy=linear)
> - 梯度裁剪：clip_grad_norm_=1.0
> - 输入：0-255 float（无 ImageNet Normalize）
>
> 详细差异见 `docs/BANet_OpenStereo_vs_Official.md`。

### 单卡训练（调试用）

```bash
# BANet2D
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file cfgs/banet2d/banet2d_orin.yaml \
  --extra_tag debug

# LightStereo-M
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file cfgs/lightstereo/lightstereo_m_orin.yaml \
  --extra_tag debug
```

**关键提示**

- `BATCH_SIZE_PER_GPU=4`；BANet2D 8 卡时全局 batch=32（官方单机 16，若要严格对齐改成 2）。
- `aligned` yaml 的 `LR=8e-4` 是官方"重头训 SceneFlow"的设置；若是从 `sceneflow.pth`
  微调到 Orbbec，可把 `OPTIMIZER.LR`（`&lr` 锚点）改成 `1e-4`。
- 输出路径：`output/OrbbecDataset/<MODEL.NAME>/<yaml 文件名>/<extra_tag>/`

**参数说明：**
- `--extra_tag`：实验子目录名
- `--cover_old_exp`：覆盖同名实验目录
- `--fix_random_seed`：固定种子
- `--workers N`：dataloader worker 数（默认 8）

权重与日志：`output/<DATASET>/<MODEL.NAME>/<yaml 文件名>/<extra_tag>/`

## 测试

使用 `tools/infer_exp_best_test_full.py` 自动选择最佳 epoch 并推理测试集：

```bash
# LightStereo-M
python tools/infer_exp_best_test_full.py \
  --exp_dir output/OrbbecDataset/LightStereo/lightstereo_m_orin/run_5ds_260519 \
  --save_color

# BANet2D
python tools/infer_exp_best_test_full.py \
  --exp_dir output/OrbbecDataset/BANet2D/banet2d_orin/run_5ds_260519 \
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

## 历史结果（参考）

### 旧合并集 `orbbec_merged_real`（2 个数据集，522 测试样本）

**BANet2D (epoch 190):** EPE 0.561 | D1(%) 1.73 | D1>1px 9.82 | D3>3px 2.48
**LightStereo-M (epoch 175):** EPE 0.816 | D1(%) 2.51 | D1>1px 17.25 | D3>3px 3.46

BANet2D 在所有指标上都显著优于 LightStereo-M（EPE 低 31%，D1 低 31%）。

## 在自定义测试集上评估

如果需要在不同的测试集上评估（而不是训练时使用的数据集）：

**方法 1：指定配置文件（推荐）**

```bash
# 复制训练配置
cp cfgs/banet2d/banet2d_orin.yaml /tmp/test_config.yaml

# 修改配置中的 DATA_PATH 和 TESTING manifest 路径，然后：
python tools/infer_exp_best_test_full.py \
  --exp_dir output/OrbbecDataset/BANet2D/banet2d_orin/run_5ds_260519 \
  --cfg_file /tmp/test_config.yaml \
  --save_color
```

**方法 2：直接在 orin yaml 上改 TESTING 路径**

由于现在统一使用 `banet2d_orin.yaml` / `lightstereo_m_orin.yaml`，需要在其它数据集上测时，
最简单的做法就是临时改这两个 yaml 里的 `DATA_PATH` 或 `DATA_SPLIT.TESTING`，跑完再改回来。
要更安全可拷贝一份后用 `--cfg_file` 指定（同方法 1）。

注意：
- 配置文件中的 `DATA_PATH` 指向 `foundation_out` 目录
- `TESTING` 字段指向测试集的 manifest 文件
- 测试时会自动使用配置文件中指定的测试集
