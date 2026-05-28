# OpenStereo 训练、测试与部署导出

本文记录 Orbbec 数据从伪 GT 生成、合并、训练、测试到导出 ONNX 给 `lightstereo_cpp` 部署的完整流程。

## 1. 环境

```bash
source /data2/shendu/anaconda3/etc/profile.d/conda.sh
conda activate openstereo
cd /data2/shendu/code/ruoyu/openstereo_add_banet/OpenStereo
```

## 2. 当前推荐配置

Orin 部署相关配置集中放在以下文件中：

```text
cfgs/lightstereo/lightstereo_m_orin.yaml          # LightStereo-M, 480x640
cfgs/lightstereo/lightstereo_s_orin.yaml          # LightStereo-S, 480x640
cfgs/lightstereo/lightstereo_s_orin_256x512.yaml  # LightStereo-S, 256x512
cfgs/banet2d/banet2d_orin.yaml                    # BANet2D, 480x640 数据训练
```

输出路径规则：

```text
output/<DATASET>/<MODEL.NAME>/<yaml 文件名>/<extra_tag>/
```

例如：

```text
output/OrbbecDataset/LightStereo/lightstereo_s_orin_256x512/run_5ds_256x512_260526/
```

## 3. 数据准备：原始 Orbbec 采集到训练数据

新采集的数据集通常是顶层 `*.png` 加 `camera_intrinsics.json`，还没有 `foundation_out/`。需要先用 FoundationStereo 生成伪视差 GT，并切分 train/val/test。

每个数据集目录里应有：

```text
scripts/process_orbbec_all_in_one.py
```

如果没有，可以从已有 Orbbec 数据集目录复制。

```bash
source /data2/shendu/anaconda3/etc/profile.d/conda.sh
conda activate foundation_stereo

CUDA_VISIBLE_DEVICES=1 python /path/to/dataset/process_orbbec_all_in_one.py \
  --session_root /path/to/dataset \
  --raw_dir /path/to/dataset \
  --skip_steps 4
```

输出内容：

```text
foundation_stereo_ir_input/{left,right}/
foundation_stereo_pseudo_gt/disp_occ_0/
foundation_out/{train,val,test}/training/{image_2,image_3,disp_occ_0}/
splits/openstereo/manifest_{train,val,test}.txt
disparity_stats.json
```

两张数据集可用不同 `CUDA_VISIBLE_DEVICES` 并行处理。每个约 1700 张图的数据集通常需要 12 到 15 分钟。

## 4. 合并多个数据集

```bash
python tools/merge_datasets.py \
  --datasets /path/to/dataset1 /path/to/dataset2 /path/to/dataset3 /path/to/dataset4 \
  --output /path/to/merged_output \
  --name merged_name \
  --symlink
```

脚本会生成：

```text
foundation_out/
splits/openstereo/manifest_train.txt
splits/openstereo/manifest_val.txt
splits/openstereo/manifest_test.txt
```

当前推荐做法：不要依赖脚本自动生成的新 yaml。统一使用第 2 节的 Orin 配置；换数据集时，只改 yaml 里的 `DATA_PATH` 和 `DATA_SPLIT` 路径。

注意：`tools/merge_datasets.py` 里非 aligned 的 `banet_config` 分支存在 NameError，当前工作流不使用它。

### 当前合并数据集

当前使用的数据集：

```text
/data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_merged_real_5ds
```

组成：

```text
orbbec_capture_20260509_140955         : 1925 / 241 / 240  (train/val/test)
orbbec_capture_20260513_103730         : 2262 / 283 / 282
orbbec_capture_20260514_110538         : 1388 / 174 / 173
orbbec_capture_20260519_101407_hafo    : 1391 / 174 / 174
orbbec_capture_20260519_102145_xiandai : 1343 / 168 / 168
合计                                  : 8309 / 1040 / 1037
```

五个采集的 `camera_intrinsics.json` 一致：

```text
fx ~= 367.35
fy ~= 367.35
cx ~= 321.93
cy ~= 241.06
baseline ~= 94.97 mm
resolution = 640x480
```

已执行的合并命令：

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

## 5. 训练

GPU 状态查看：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

### LightStereo-S 256x512

这是给 `lightstereo_cpp` 现有 256x512 部署入口准备的 S 模型。

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 /data2/shendu/anaconda3/envs/openstereo/bin/python tools/train.py \
  --cfg_file cfgs/lightstereo/lightstereo_s_orin_256x512.yaml \
  --extra_tag run_5ds_256x512_260526
```

多卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29503 \
  tools/train.py --dist_mode \
  --cfg_file cfgs/lightstereo/lightstereo_s_orin_256x512.yaml \
  --extra_tag run_5ds_256x512_260526
```

输出目录：

```text
output/OrbbecDataset/LightStereo/lightstereo_s_orin_256x512/run_5ds_256x512_260526/
```

### LightStereo-S 480x640

```bash
CUDA_VISIBLE_DEVICES=0 /data2/shendu/anaconda3/envs/openstereo/bin/python tools/train.py \
  --cfg_file cfgs/lightstereo/lightstereo_s_orin.yaml \
  --extra_tag run_5ds_260526
```

输出目录：

```text
output/OrbbecDataset/LightStereo/lightstereo_s_orin/run_5ds_260526/
```

### LightStereo-M 480x640

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=2 --master_port=29502 \
  tools/train.py --dist_mode \
  --cfg_file cfgs/lightstereo/lightstereo_m_orin.yaml \
  --extra_tag run_5ds_260519
```

输出目录：

```text
output/OrbbecDataset/LightStereo/lightstereo_m_orin/run_5ds_260519/
```

### BANet2D

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 \
  tools/train.py --dist_mode \
  --cfg_file cfgs/banet2d/banet2d_orin.yaml \
  --extra_tag run_5ds_260519
```

BANet2D 使用 aligned 版本，对齐官方 `BANet/banet-2d` 训练 pipeline：

```text
数据增广：BANetFlowAugmentor，必须配置 CROP_SIZE
推荐 CROP_SIZE: [320, 608]
优化器：AdamW(lr=8e-4, wd=1e-5, eps=1e-8)
调度器：OneCycleLR(pct_start=0.01, cycle_momentum=False, anneal_strategy=linear)
梯度裁剪：clip_grad_norm_=1.0
输入：0-255 float，无 ImageNet Normalize
```

详细差异见：

```text
docs/BANet_OpenStereo_vs_Official.md
```

### 训练参数说明

```text
--extra_tag        实验子目录名
--cover_old_exp    覆盖同名实验目录
--fix_random_seed  固定随机种子
--workers N        dataloader worker 数，默认 8
```

## 6. 测试

使用 `tools/infer_exp_best_test_full.py` 自动选择最佳 epoch 并推理测试集。

LightStereo-M 示例：

```bash
python tools/infer_exp_best_test_full.py \
  --exp_dir output/OrbbecDataset/LightStereo/lightstereo_m_orin/run_5ds_260519 \
  --save_color
```

LightStereo-S 256x512 示例：

```bash
python tools/infer_exp_best_test_full.py \
  --exp_dir output/OrbbecDataset/LightStereo/lightstereo_s_orin_256x512/run_5ds_256x512_260526 \
  --save_color
```

BANet2D 示例：

```bash
python tools/infer_exp_best_test_full.py \
  --exp_dir output/OrbbecDataset/BANet2D/banet2d_orin/run_5ds_260519 \
  --save_color
```

常用参数：

```text
--force_epoch 199          强制使用指定 epoch
--cfg_file /path/to.yaml   指定测试配置，默认取实验目录里的 yaml
--max-z 2.5                深度裁切阈值，默认 2.5m，设为 0 不裁切
--out_dir /path            自定义输出目录
```

输出内容：

```text
disp_png/               KITTI uint16 视差图
ply/                    点云，左相机系，默认去掉 Z>2.5m
metrics_per_frame.csv   每帧指标
summary_test.txt        汇总指标，EPE、D1、D3 等
```

## 7. 导出 ONNX

`lightstereo_cpp` 要求 ONNX 输入输出为：

```text
left_img, right_img -> disp_pred
```

当前 `deploy/export.py` 已按这个接口导出。推荐固定尺寸导出，不使用 dynamic shape。

推荐 opset：

```text
opset 11
```

原因：当前 `lightstereo_cpp` 没有硬编码 opset，但 RKNN / TensorRT / OnnxRuntime 示例链路均可吃 opset 11；RKNN Docker 里安装的是 `onnx>=1.14.0,<1.16`。

### 导出 LightStereo-S 256x512

```bash
/data2/shendu/anaconda3/envs/openstereo/bin/python deploy/export.py \
  --config cfgs/lightstereo/lightstereo_s_orin_256x512.yaml \
  --weights output/OrbbecDataset/LightStereo/lightstereo_s_orin_256x512/run_5ds_256x512_260526/ckpt/checkpoint_epoch_xxx.pth \
  --output output/OrbbecDataset/LightStereo/lightstereo_s_orin_256x512/run_5ds_256x512_260526/lightstereo_s_orin_epochxxx_256x512.onnx \
  --imgsz 256 512 \
  --include onnx \
  --opset 11 \
  --device cpu
```

预期 ONNX：

```text
opset    ai.onnx 11
left_img  [1, 3, 256, 512]
right_img [1, 3, 256, 512]
disp_pred [1, 1, 256, 512]
```

对应 `lightstereo_cpp` 的 256x512 示例入口。

### 导出 LightStereo-S 480x640

```bash
/data2/shendu/anaconda3/envs/openstereo/bin/python deploy/export.py \
  --config cfgs/lightstereo/lightstereo_s_orin.yaml \
  --weights output/OrbbecDataset/LightStereo/lightstereo_s_orin/run_5ds_260526/ckpt/checkpoint_epoch_xxx.pth \
  --output output/OrbbecDataset/LightStereo/lightstereo_s_orin/run_5ds_260526/lightstereo_s_orin_epochxxx_480x640.onnx \
  --imgsz 480 640 \
  --include onnx \
  --opset 11 \
  --device cpu
```

### 导出 LightStereo-M 480x640

```bash
/data2/shendu/anaconda3/envs/openstereo/bin/python deploy/export.py \
  --config cfgs/lightstereo/lightstereo_m_orin.yaml \
  --weights output/OrbbecDataset/LightStereo/lightstereo_m_orin/run_5ds_260519/ckpt/checkpoint_epoch_190.pth \
  --output output/OrbbecDataset/LightStereo/lightstereo_m_orin/run_5ds_260519/lightstereo_m_orin_epoch190_480x640.onnx \
  --imgsz 480 640 \
  --include onnx \
  --opset 11 \
  --device cpu
```

## 8. 验证 ONNX

```bash
/data2/shendu/anaconda3/envs/openstereo/bin/python - <<'PY'
import onnx

path = 'path/to/model.onnx'
model = onnx.load(path)
onnx.checker.check_model(model)

print('opsets:', [(o.domain or 'ai.onnx', o.version) for o in model.opset_import])
print('inputs:', [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in model.graph.input])
print('outputs:', [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in model.graph.output])
PY
```

256x512 应输出：

```text
opsets: [('ai.onnx', 11)]
inputs: [('left_img', [1, 3, 256, 512]), ('right_img', [1, 3, 256, 512])]
outputs: [('disp_pred', [1, 1, 256, 512])]
```

480x640 应输出：

```text
opsets: [('ai.onnx', 11)]
inputs: [('left_img', [1, 3, 480, 640]), ('right_img', [1, 3, 480, 640])]
outputs: [('disp_pred', [1, 1, 480, 640])]
```

实际检查时，ONNX 可能把 `disp_pred` 的 batch/height/width 显示为符号名，但只要维度含义为 `[1, 1, H, W]` 即可。

## 9. 放入 lightstereo_cpp 部署

部署工程：

```text
/data2/shendu/code/ruoyu/lightstereo_cpp
```

C++ 侧使用的输入输出名：

```text
left_img
right_img
disp_pred
```

256x512 示例位置：

```text
stereo/stereo_lightstereo/benchmark/benchmark_stereo_lightstereo.cpp
stereo/stereo_lightstereo/test/test_stereo_lightstereo.cpp
```

部署流程：

```text
OpenStereo checkpoint .pth
-> deploy/export.py 导出 .onnx
-> 复制 ONNX 到 lightstereo_cpp Docker 的 /workspace/models/
-> 按 tools/cvt_onnx2rknn*.py 或 tools/cvt_onnx2trt.sh 转换
-> C++ 推理
```

`.pth` 不能直接放进 `lightstereo_cpp` 推理，必须先导出 ONNX；RKNN 端还需要再转 `.rknn`。

## 10. 自定义测试集评估

如果需要在不同测试集上评估，而不是训练时使用的数据集，推荐复制一份配置后改路径：

```bash
cp cfgs/banet2d/banet2d_orin.yaml /tmp/test_config.yaml
```

修改 `/tmp/test_config.yaml` 中的：

```text
DATA_CONFIG.DATA_INFOS[0].DATA_PATH
DATA_CONFIG.DATA_INFOS[0].DATA_SPLIT.TESTING
```

然后运行：

```bash
python tools/infer_exp_best_test_full.py \
  --exp_dir output/OrbbecDataset/BANet2D/banet2d_orin/run_5ds_260519 \
  --cfg_file /tmp/test_config.yaml \
  --save_color
```

也可以直接临时修改 Orin yaml 中的 `DATA_PATH` 或 `DATA_SPLIT.TESTING`，但跑完要改回。

## 11. 历史结果参考

旧合并集 `orbbec_merged_real`，测试集 522 张：

```text
BANet2D epoch 190      EPE 0.561 | D1 1.73% | D1>1px 9.82 | D3>3px 2.48
LightStereo-M epoch175 EPE 0.816 | D1 2.51% | D1>1px 17.25 | D3>3px 3.46
```

BANet2D 在该旧测试集上优于 LightStereo-M：EPE 低约 31%，D1 低约 31%。
