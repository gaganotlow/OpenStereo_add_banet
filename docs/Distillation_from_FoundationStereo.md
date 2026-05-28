# 从 FoundationStereo 蒸馏到 BANet2D / LightStereo

当前 baseline 已经在用 FoundationStereo 的最终视差作为伪 GT 训练。再往上加蒸馏 loss 之前，
必须先量化 BANet2D 与 FoundationStereo 的 gap **大小、位置、性质**，否则后续所有方案的优先
级排序都是拍脑袋。

本文只写 **第一步：诊断**。优化方案在诊断完成后另议。

---

## 诊断要回答的三个问题

| # | 问题 | 决定 |
|---|---|---|
| Q1 | BANet2D 与 FoundationStereo 在测试集上的 EPE / D1 gap 多大？ | 值不值得做复杂蒸馏 |
| Q2 | gap 集中在哪些区域：边界 / 高反 / 远景 / 细小结构 / 弱纹理？ | 用哪种蒸馏对症 |
| Q3 | gap 与 FoundationStereo 自己的概率熵是否正相关？ | teacher 是不是瓶颈 |

**决策树**（诊断完成后据此分支）：

- Q1 gap < 0.3 px → 不需要复杂蒸馏，加数据或调增广更划算
- Q2 边界占主导 → 概率分布 KL 蒸馏对症
- Q2 高反占主导 + Q3 正相关（teacher 自己也不确定的地方学生也错） → teacher 是瓶颈，
  蒸馏帮不上，转向半监督一致性 / 多 teacher 集成 / 主动光辅助监督
- Q2 分布均匀 + Q3 弱相关 → 学生容量瓶颈，概率分布 KL + 特征蒸馏组合

---

## 当前可用的输入

- 训练目录：`output/OrbbecDataset/BANet2D/banet2d_orin/run_5ds_260519/`（训练中）
- 数据集根：`/data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_merged_real_5ds/`
- 测试集 manifest：`<root>/splits/openstereo/manifest_test.txt`
- 伪 GT 视差：`<root>/foundation_out/test/training/disp_occ_0/*.png`（KITTI uint16，disp × 256）
- 已有推理脚本：`tools/infer_exp_best_test_full.py`
  - 自动按 train log 选 best ckpt
  - 输出 `disp_png/*.png`、`metrics_per_frame.csv`、`summary_test.txt`
  - **不够诊断用**：只有 per-frame 标量，没有 per-pixel error map，没有分桶

## 当前缺的输入

1. **FoundationStereo 在测试集上的概率分布或熵图**
   `foundation_out/` 里只存了最终视差，没有 `prob` / `entropy`，无法回答 Q3。
   需要补一个提取脚本。

2. **per-pixel error map 与分桶聚合工具**
   现有推理脚本只统计 per-frame，诊断需要 per-pixel 维度。

---

## 预备工作（训练同时做，互不干扰）

### P1 — FoundationStereo 熵提取脚本 ✅ 已完成

脚本：`tools/extract_foundation_entropy.py`

**做了什么**：复制了 `foundation_stereo.py:198-228` 的前半段（feature → cost volume → classifier
→ softmax），跳过 GRU 迭代，提取 init prob 的熵图。比完整 forward 省 ~80% 时间。

**输入约定**：raw uint8 → float32 RGB（不做 mean/std normalize），与生成 `foundation_out` 时
的 `process_orbbec_all_in_one.py:164` 一致。

**输出**：`<out_dir>/<stem>.npy`，shape `(H/4, W/4)` float16。

**命令样例**：
```bash
python tools/extract_foundation_entropy.py \
    --manifest /data2/.../orbbec_merged_real_5ds/splits/openstereo/manifest_test.txt \
    --data_root /data2/.../orbbec_merged_real_5ds/foundation_out \
    --teacher_ckpt /data2/shendu/code/ruoyu/FoundationStereo/<model_dir>/model_best_bp2.pth \
    --teacher_cfg  /data2/shendu/code/ruoyu/FoundationStereo/<model_dir>/cfg.yaml \
    --out_dir /data2/.../orbbec_merged_real_5ds/foundation_out/test/training/entropy
```

**用户必须提供**：
- `--teacher_ckpt`：与生成 foundation_out 时**完全一致的权重**（否则熵不可比）
- `--teacher_cfg`：与权重同目录的 `cfg.yaml`（含 vit_size、max_disp 等字段）

**工程细节**：
- `--limit N` smoke test（只跑前 N 张）
- 默认跳过已存在的 `.npy`，`--overwrite` 强制重算
- InputPadder 在 1/4 分辨率上的对齐：上采到 pad 全分辨率 → unpad → 下采回原图 1/4，避免
  padder 偏移污染熵图

**注意**：本 entropy 是 **init prob 的熵**，不是 FS 最终视差的不确定性（FS 最终视差由 GRU
迭代精化，是一个标量而非分布）。但 init prob 的熵作为 teacher 自身不确定性的代理足以回答
Q3。诊断报告里会注明这一点。

### P2 — 诊断分析脚本 ✅ 已完成

脚本：`tools/diagnose_banet_vs_foundation.py`

**输入**：
- `--pred_dir`：BANet2D 推理输出的 `disp_png/`（来自 `infer_exp_best_test_full.py`）
- `--data_root`：`foundation_out/` 根目录（用于解析 manifest 相对路径）
- `--manifest`：测试集 manifest_test.txt
- `--entropy_dir`：P1 输出目录
- `--out_dir`：诊断报告输出目录

**输出**：
- `bucket_table.csv`：每桶 (frames, pixels, pct, EPE, D1@1px, D1@3px)
- `entropy_error_curve.png`：熵分位数 vs mean EPE 折线（含 pixel count 柱状图叠加）
- `vis_topk/<stem>.png`：mean EPE 最大的 K 张，左图 + error heatmap 并排（K 默认 5）
- `report.md`：自动文字总结 + **自动决策树判定分支**

**分桶定义**（全部参数集中在脚本顶部常量，方便调）：

| 桶名 | 划分条件 | 含义 |
|---|---|---|
| `edge` | `‖∇disp_gt‖ > 1.0`，膨胀 2 px | 视差跳变边界 |
| `flat` | 非 edge 且 valid | 平坦区 |
| `near` | `disp_gt ≥ 64` | 近景 |
| `far` | `0 < disp_gt < 32` | 远景 |
| `highlight` | 灰度 > 230 且 11×11 局部方差 < 50 | 高反/过曝 |
| `lowtex` | 局部方差 < 30 且非 highlight | 弱纹理 |
| `fine` | 边界密度（11×11 box filter）> 0.3 且非 edge | 细丝/小物体内部 |
| `all` | `0 < disp_gt < max_disp_valid` | 全像素基线 |

桶之间允许重叠，分别统计。`all` 用作各桶 EPE 比例的分母。

**熵-error 相关性**：每帧随机采样 20000 个 valid 像素（CLI `--entropy_sample_per_frame` 可调），
累积到全局数组后计算 Pearson + Spearman，并按熵分位数分 10 箱画折线。Spearman 比 Pearson 更
鲁棒，决策树主要看 Spearman。

**自动决策树**（写在 `_decide_branch`）：

| 分支 | 触发条件 | 建议 |
|---|---|---|
| `A_gap_small` | 总 EPE < 0.3 px | 不值得复杂蒸馏，加数据或调增广 |
| `C_teacher_bottleneck` | highlight EPE ratio > 1.5 **且** Spearman > 0.3 | teacher 自身瓶颈，蒸馏帮不上，转向半监督一致性 / 多 teacher / 主动光 |
| `B_edge_dominant` | edge EPE ratio > 1.5 且非 C 分支 | 方案 1（概率分布 KL 蒸馏）对症 |
| `B_fine_or_far` | fine 或 far EPE ratio > 1.5 | 方案 1 + 方案 3（特征蒸馏） |
| `D_student_capacity` | 各桶都 < 1.3 且 Spearman < 0.2 | 学生容量瓶颈，方案 1 + 方案 3 |
| `E_mixed` | 都不匹配 | 人工查 bucket_table 与曲线 |

阈值都在脚本常量里，跑出来如果觉得不对可以改。

**命令样例**：
```bash
python tools/diagnose_banet_vs_foundation.py \
    --pred_dir   $EXP/infer_test_best/disp_png \
    --data_root  $ROOT/foundation_out \
    --manifest   $ROOT/splits/openstereo/manifest_test.txt \
    --entropy_dir $ROOT/foundation_out/test/training/entropy \
    --out_dir    $EXP/diagnose
```

**smoke test 用法**：加 `--limit 10` 只跑 10 张，验证 pipeline。

### P3 — 验证脚本能跑通（用部分数据）

训练还没结束，但可以拿现有任一 epoch 的 ckpt 先跑一遍 pipeline 验证脚本无误：
```bash
# 用最新 ckpt 强制跑（不等 best）
python tools/infer_exp_best_test_full.py \
    --exp_dir output/OrbbecDataset/BANet2D/banet2d_orin/run_5ds_260519 \
    --force_epoch <最新一个 epoch>
```
然后跑 P1 + P2，确认输出格式无误。**P3 只是 smoke test，结论不要用于决策。**

---

## 训练完成后的执行流程

```bash
EXP=output/OrbbecDataset/BANet2D/banet2d_orin/run_5ds_260519
ROOT=/data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_merged_real_5ds
FS=/data2/shendu/code/ruoyu/FoundationStereo/pretrained_models/23-51-11
source /data2/shendu/anaconda3/etc/profile.d/conda.sh

# 1. 跑 BANet2D best ckpt 在测试集上的推理（已有脚本，openstereo env）
conda activate openstereo
python tools/infer_exp_best_test_full.py --exp_dir $EXP

# 2. 提取 FoundationStereo 熵（P1）—— 必须用 foundation_stereo env
#    原因：脚本 import 的 vendored foundationstereo.core 依赖 trimesh/imageio/flash_attn，
#    只有 foundation_stereo env 才齐。LD_PRELOAD 是为修 flash_attn 的 GLIBCXX_3.4.29
#    （env 自带的 libstdc++ 缺该符号，base env 的 libstdc++ 有，用 LD_PRELOAD 强制加载它）。
conda activate foundation_stereo
export LD_PRELOAD=/data2/shendu/anaconda3/lib/libstdc++.so.6
CUDA_VISIBLE_DEVICES=0 python tools/extract_foundation_entropy.py \
    --manifest    $ROOT/splits/openstereo/manifest_test.txt \
    --data_root   $ROOT/foundation_out \
    --teacher_ckpt $FS/model_best_bp2.pth \
    --teacher_cfg  $FS/cfg.yaml \
    --out_dir     $ROOT/foundation_out/test/training/entropy
# 先加 --limit 5 跑 smoke test，确认有 .npy 产出再去掉跑全量 1037 张。
# 模型加载（DINOv2 backbone）需 30–60s 才会打印 [cfg]/[device]/[load]，期间无输出属正常。

# 3. 诊断分析（P2）—— 只用 cv2/numpy/matplotlib，openstereo env 即可
conda activate openstereo
python tools/diagnose_banet_vs_foundation.py \
    --pred_dir    $EXP/infer_test_best/disp_png \
    --data_root   $ROOT/foundation_out \
    --manifest    $ROOT/splits/openstereo/manifest_test.txt \
    --entropy_dir $ROOT/foundation_out/test/training/entropy \
    --out_dir     $EXP/diagnose
```

产物在 `$EXP/diagnose/`。

**环境踩坑记录**（实测 2026-05-21）：
- teacher 权重就在 `FoundationStereo/pretrained_models/23-51-11/`（唯一模型目录，含 `cfg.yaml` + `model_best_bp2.pth`），与生成 `foundation_out` 同一份。
- 第 2 步**不能**用 openstereo env：缺 joblib/trimesh/imageio/flash_attn，逐个补装是 whack-a-mole。
- 第 2 步用 foundation_stereo env 时，flash_attn 会报 `GLIBCXX_3.4.29 not found`。foundation_stereo env 自带的 libstdc++ 也缺该符号（所以 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` 没用），需 `export LD_PRELOAD=/data2/shendu/anaconda3/lib/libstdc++.so.6`（base env 的 libstdc++ 有 3.4.29）。定位命令：`find /data2/shendu/anaconda3 -name 'libstdc++.so.6*' | xargs -I{} sh -c 'strings {} | grep -q GLIBCXX_3.4.29 && echo {}'`。
- GPU 0–3 空闲，4–7 在跑训练，用 `CUDA_VISIBLE_DEVICES=0`。
- vendored `foundationstereo` 里有两处原作者机器的死路径/占位，需改 vendored 源码（仅影响这条熵提取链，BANet 训练/推理不走这里）：
  1. `depth_anything/dpt.py` 的 DINOv2 加载：原 `torch.hub.load('/file_system/vepfs/.../facebookresearch_dinov2_main', ...)`，本机 hub 副本是被污染的新版 hubconf（import 不存在的 `cell_dino`/`xray_dino`）。改为直接用 vendored builder `from dinov2.hub.backbones import dinov2_vitl14; self.pretrained = dinov2_vitl14(pretrained=False)`，架构与 teacher 权重同版本。
  2. `depth_anything/dpt.py` 的 `DepthAnything.__init__`：原无条件 `torch.load('/your_path/depth_anything_v2_{encoder}.pth')`。改为文件不存在则跳过——完整权重随 teacher ckpt 加载（strict=False）覆盖随机初始化。
- cfg.yaml 缺 `vit_size`：脚本里默认 `vitl`。注意必须用 EasyDict 属性赋值 `model_cfg.vit_size = "vitl"`，不能用 `dict.setdefault`（绕过 EasyDict 的 `__setattr__`，属性访问读不到）。
- smoke test 实测 `[load] missing=0 unexpected=0`，teacher 权重完全匹配，熵图 shape `(H/4, W/4)=(120,160)`。

**当前进度**：第 1 步推理已完成（`infer_test_best/`，1037 张，mean EPE=0.485px vs 伪 GT，> 0.3px 阈值 → 不属 A_gap_small，值得继续诊断）。第 2/3 步待跑。

---

## 诊断报告需要看的几张表 / 图

1. **总 gap**（一行）：`mean EPE`、`mean D1@1px`、`mean D1@3px`
2. **分桶 EPE 表**：每个桶的像素占比、mean EPE、与全局 EPE 的差倍数
3. **熵—error 折线**：把熵分 10 箱，每箱画 mean EPE。如果单调上升 → Q3 正相关；如果平的 → 学生瓶颈
4. **抽样可视化**：error 最大的 5 张图，叠加 error heatmap

按这几张表的形状，选定后续路径，再开始写蒸馏 loss。

---

## 注意

- 测试集 manifest 的"GT"本来就是 FoundationStereo 输出（伪 GT），所以本诊断的 EPE 不是
  对真实 GT 的，而是 **BANet2D 相对于 teacher 的拟合误差**。这正是评估蒸馏空间的正确指标。
- 如果以后能拿到一小批真实 GT（结构光 / ToF 标注），再做一次诊断，可以同时回答"teacher 本
  身的偏差"。但本步不依赖。

---

## 方案 1 实现：边界概率分布 KL 蒸馏（诊断分支 B_edge_dominant）

诊断结论：gap 几乎全在 edge 桶（EPE ×4.33），熵-error Spearman 0.374。当前训练只用 teacher
argmax 后的标量视差做 smooth-L1，丢掉了 teacher 在边界的多峰软分布。方案 1 = 把 teacher 的
视差概率分布通过 KL 蒸馏补给 student。teacher prob **离线预存**。

### bin 轴对齐（关键事实）
BANet `CostVolume` 与 FS `build_gwc_volume` 都在 1/4 分辨率把右图右移 i 像素构 cost（bin i =
1/4 分辨率视差 i）。FS 有 104 bin（max_disp=416→416/4），BANet 48 bin（192/4）。取 FS prob 前
48 bin 沿 bin 维重归一化即与 student 对齐（Orbbec disp<192 → 1/4 分辨率 <48，截断损失可忽略）。

### 代码改动
- `tools/extract_foundation_entropy.py`：加 `--save_prob_dir` + `--student_bins 48`，
  存截断+重归一化的 teacher prob `[48,H/4,W/4]` fp16（~1.84MB/帧）。熵输出不变。
- `kitti_dataset.py`：training 模式按 `DATA_INFOS.TEACHER_PROB_DIR` 读 `<stem>.npy` →
  `sample['teacher_prob']`（CHW，eval 不读）。
- `stereo_trans.py::BANetFlowAugmentor`：`spatial_transform` 返回 `(y0,x0)`，对 teacher_prob 在
  1/4 分辨率按 `(round(y0/4),round(x0/4))` 同步裁出 `[48,80,152]`，索引 clamp。
- `banet_core.py`：training return 增加 1/4 分辨率 cost volume logits `cv`（softmax 前）。
- `banet2d.py`：forward 增加 `disp_volume`；`get_loss` 加 masked KL（valid mask 最近邻降到 1/4），
  `loss = smooth_l1 + LAMBDA_KL * kl`，记录 `scalar/train/loss_kl`。`LAMBDA_KL<=0` 或无
  teacher_prob 时退化为纯 smooth_l1。
- `cfgs/banet2d/banet2d_orin_distill.yaml`：从 best ckpt(epoch 195) 精修，60 epoch。

### LAMBDA_KL 标定
从 best ckpt 精修时实测：**loss_disp≈0.2**（已很准）、**loss_kl≈8 nats**（student argmax 训出的
尖锐分布 vs teacher 软分布，差异天然大）。smoke 10 帧单 iter 的 disp=1.4 不具代表性。要让 KL 项
与 disp 同量级而不压过视差监督，默认 **LAMBDA_KL=0.02**（KL 项≈0.16≈disp），可在 [0.01,0.05] 扫。

### teacher prob 提取（仅 train，val 不用因 eval 不读 teacher_prob）
```bash
conda activate foundation_stereo
export LD_PRELOAD=/data2/shendu/anaconda3/lib/libstdc++.so.6
ROOT=/data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_merged_real_5ds
FS=/data2/shendu/code/ruoyu/FoundationStereo/pretrained_models/23-51-11
CUDA_VISIBLE_DEVICES=1 python tools/extract_foundation_entropy.py \
    --manifest    $ROOT/splits/openstereo/manifest_train.txt \
    --data_root   $ROOT/foundation_out \
    --teacher_ckpt $FS/model_best_bp2.pth \
    --teacher_cfg  $FS/cfg.yaml \
    --out_dir      $ROOT/foundation_out/train/training/entropy \
    --save_prob_dir $ROOT/foundation_out/teacher_prob \
    --student_bins 48
# 注：stdout 进度行被重定向缓冲，看 teacher_prob/ 目录文件数判断进度。
```

### 蒸馏训练 + 复测
```bash
conda activate openstereo
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29503 \
  tools/train.py --dist_mode \
  --cfg_file cfgs/banet2d/banet2d_orin_distill.yaml \
  --extra_tag run_5ds_distill
# 训完用同一套 infer + diagnose 复测，主看 bucket_table.csv 的 edge 桶 EPE（当前 2.05）是否下降，
# 全局 EPE（当前 0.473）是否改善、flat 桶不退化。
```

### 首版结果（2026-05-21，λ=0.02，60 epoch）— **基本无效**

| 桶 | baseline EPE | distill EPE | Δ |
|---|---:|---:|---:|
| all | 0.4732 | 0.4693 | −0.8% |
| edge（目标） | 2.0485 | 2.0195 | −1.4% |
| flat | 0.3625 | 0.3605 | −0.6% |
| far | 0.4422 | 0.4345 | −1.7% |
| fine | 3.2017 | 3.1397 | −1.9% |

差异全部 <2%，且 **val 反而略差**（起点 best ckpt val EPE=0.4778 → 蒸馏后 0.4814）。结论：噪声级，未真正改善。

**根因（非 λ 问题）**：config 把 `PRETRAINED_MODEL` 指向 best ckpt，却保留了从头训练的 OneCycleLR
（峰值 8e-4）。val EPE 轨迹 **0.478(起点)→0.581(epoch0,被打飞)→0.481(epoch55,爬回来)**：整个 60
epoch 预算几乎全花在“从被破坏的状态恢复”，KL 精修没机会生效。

**修正方向**：
1. fine-tune 必须用低 LR（常量 ~5e-5 或从低值 cosine 衰减），不要 OneCycle 大幅 ramp，保住好初始化。
2. KL 仅作用于全部像素被 93% flat 稀释 → 按 teacher 熵 / disp 梯度加权，把信号集中到边界。
3. LR 修好后再把 λ 上调到 0.05–0.1 扫。

### 第二版结果（2026-05-22，低 LR 隔离实验，`banet2d_orin_distill_ft.yaml`）— **仍无效，但定位了根因**

只改修正方向 1：LR 8e-4→5e-5（16×↓），PCT_START 0.01 几乎立即从 5e-5 线性衰减到 ~0，25 epoch，λ 仍 0.02。
**目的**：保住初始化后隔离验证"KL 到底有没有用"。初始化确实保住了——val EPE 轨迹 epoch0=0.4854、
中段 0.487–0.493、ep24=0.4896，全程贴着起点 0.478，没有 v1 的 0.581 打飞。复测（强制 ep24，KL 暴露最充分）：

| 桶 | baseline | v1(OneCycle) | **ft 低LR** | ft vs baseline |
|---|---:|---:|---:|---:|
| all | 0.4732 | 0.4693 | 0.4802 | +1.5% |
| edge（目标） | 2.0485 | 2.0195 | 2.0412 | **−0.4%（噪声）** |
| flat | 0.3625 | 0.3605 | 0.3706 | +2.2% |
| far | 0.4422 | 0.4345 | 0.4566 | +3.3% |
| fine | 3.2017 | 3.1397 | 3.2066 | +0.2% |

edge 桶 **−0.4%，纯噪声**；其余桶反而微升。三轮 edge 全在 ±1.5% 内互相波动，**没有任何一组真正改了边界**。

**确定的根因**（隔离实验排除了"OneCycle 打飞"这个候选）：LR 修好、初始化保住后 KL 仍无效 →
问题不在 LR，而在 **KL 信号被 93% flat 像素稀释 + λ=0.02 全像素均匀加权太弱，梯度根本传不到 edge**。
即修正方向 2 是真瓶颈。

**下一步（修正方向 2 + 3）**：把 KL 按 **teacher 熵 / disp 梯度** 加权（edge 处熵高/梯度大 → 权重大），
信号集中到边界后再把 λ 上调到 0.05–0.1。实现上在 `banet2d.py:get_loss` 的 masked KL 上乘逐像素权重图
（teacher_prob 的熵已可在线算，或离线把 entropy.npy 一起读进来）。

### 第三版结果（2026-05-22，teacher 熵加权 KL，λ=0.05）— **失败，KL 路线基本判死**

实现：`MODEL.DISTILL.KL_WEIGHT=entropy`，在 1/4 分辨率用 `H(teacher_prob)` 对 KL 逐像素加权，
`sum(w*kl)/sum(w)`；λ 从 0.02 提到 0.05。smoke 标定：`loss_kl≈7`、`loss_disp≈0.3`，
λ=0.05 → KL 项≈0.35，与 disp 同量级，且梯度集中到高熵/边界像素。训练强制 ep24 复测：

| 桶 | baseline | ft 均匀 KL | **熵加权 KL** | ew vs baseline |
|---|---:|---:|---:|---:|
| all | 0.4732 | 0.4802 | 0.4970 | **+5.0%** |
| edge（目标） | 2.0485 | 2.0412 | 2.0759 | **+1.3%** |
| flat | 0.3625 | 0.3706 | 0.3861 | +6.5% |
| near | 0.5920 | 0.6012 | 0.6191 | +4.6% |
| far | 0.4422 | 0.4566 | 0.4824 | +9.1% |
| fine | 3.2017 | 3.2066 | 3.2432 | +1.3% |

结论：**熵加权没有拉动 edge，反而全桶退化**。这排除了"均匀 KL 被 flat 稀释"这个主要候选：
即便把 KL 强信号集中到高熵/边界，student 的 argmax 边界误差仍不降。

更可能的根因：
1. **teacher init prob 不是正确监督目标**：FS 最终边界精度来自 GRU 迭代后的标量精修；init prob 的高熵多峰
   更多表示"不确定"而不是"边界应如何变准"。拟合它会让 student 更不确定，但不提升 argmax EPE。
2. **BANet 表达瓶颈**：1/4 分辨率、48-bin cost volume + soft-argmax/context upsample 对真实深度跳变表达能力有限；
   边界误差可能是架构/分辨率瓶颈，不是缺 soft label。

决策：**方案 1（概率 KL 蒸馏）到此为止，不再继续扫 λ/权重**。继续扫只是在已退化的目标上调参。
下一步若还要攻边界，应换路线：
- 方案 3：特征蒸馏（中间特征/edge-aware feature loss），不要拟合 init prob；或
- 架构侧：提高边界分辨率/细化模块，而不是继续改 KL；或
- 若目标是最终部署指标，优先回到 baseline/数据增广路线。
