# 从 FoundationStereo 蒸馏到 BANet2D / LightStereo

## 当前做法的局限

目前用 FoundationStereo 推理结果作为伪真值（pseudo label），只榨取了**最终视差图**这一个标量信号。FoundationStereo 内部还有大量信息没被利用：

- 视差**概率分布**（不只是 argmax 后的 1 个数字，而是 48 个视差候选的完整分布）
- 多尺度**特征图**（ViT backbone 提取的丰富语义特征）
- **Cost Volume** 表征（匹配代价的几何先验）
- **GRU 迭代中间结果**（每一步细化的视差）
- **置信度 / 不确定性**估计

蒸馏（distillation）就是把这些信息打包传给小模型，让它学到不只是"最终答案"，还有"老师是怎么思考的"。

---

## FoundationStereo 内部可榨取的信号

参考 `stereo/modeling/models/foundationstereo/core/foundation_stereo.py:198-265`：

```python
def forward(self, data):
    # ...
    out, vit_feat = self.feature(...)                       # [1] ViT 多尺度特征
    features_left = [o[:B] for o in out]                    # [2] 左图金字塔特征
    gwc_volume = build_gwc_volume(...)                      # [3] Group-wise cost volume
    comb_volume = self.cost_agg(comb_volume, features_left) # [4] 聚合后 cost volume
    prob = F.softmax(self.classifier(comb_volume), dim=1)   # [5] 视差概率分布 (B, 48, H/4, W/4)
    init_disp = disparity_regression(prob, max_disp//4)     # [6] 初始视差
    # GRU 迭代
    for itr in range(iters):
        ...
        disp_preds.append(disp_up)                          # [7] 每次迭代的视差
    return {'init_disp': init_disp,
            'disp_preds': disp_preds,
            'disp_pred': disp_preds[-1]}                    # [8] 最终视差（当前唯一在用）
```

按蒸馏价值从高到低：

| 信号 | 价值 | 实现难度 | 推荐 |
|---|---|---|---|
| [5] 视差概率分布 | **极高** | 低 | ⭐⭐⭐ 必做 |
| [8] 最终视差 | 高 | 低 | ⭐⭐⭐ 已在做 |
| [2] 多尺度特征 | 高 | 中 | ⭐⭐ |
| [6] 置信度/不确定性 | 高 | 低 | ⭐⭐ |
| [4] Cost Volume | 中 | 中 | ⭐ |
| [7] 迭代中间视差 | 中 | 低 | ⭐ |

---

## 蒸馏方案

### 方案 1：Soft Label 概率分布蒸馏（性价比最高）

**核心思想**：FoundationStereo 的 `prob` 是 (B, 48, H/4, W/4) 的视差概率分布；BANet2D 内部 `F.softmax(cv, dim=1)` 也是同样形状的概率分布。**直接用 KL 散度让 BANet 的概率分布拟合 FoundationStereo 的概率分布**。

为什么有效：
- argmax 后的视差丢掉了"老师对哪个视差最有信心、第二可能是哪个"的信息
- 概率分布相当于教学生"思考过程"，而不只是"标准答案"
- 高反区 FoundationStereo 的概率分布会变得**更分散（熵更大）**，BANet 学到的也是"这里不确定"，自然不会硬猜

**实现**（在 `banet2d.py::BANet2D` 里加）：

```python
# 训练前：离线把 FoundationStereo 在训练集上的 prob 存成 .npy（48×120×160 float16）
# 或在线：FoundationStereo 与 BANet 共享 dataloader，每个 batch 前向一次

def get_loss(self, model_pred, input_data):
    disp_gt = input_data["disp"].unsqueeze(1)
    mask = (disp_gt < self.max_disp) & (disp_gt > 0)

    # === 1. 原有视差 loss（pseudo gt 监督）===
    disp_pred = model_pred["disp_pred"]
    disp_coarse = model_pred["disp_coarse"]
    loss_disp = (F.smooth_l1_loss(disp_pred[mask], disp_gt[mask]) +
                 0.3 * F.smooth_l1_loss(disp_coarse[mask], disp_gt[mask]))

    # === 2. 概率分布蒸馏（新增）===
    teacher_prob = input_data["fs_prob"]            # (B, 48, H/4, W/4) FoundationStereo 输出
    student_prob = model_pred["raw_prob"]           # 需在 banet_core.py 里把 F.softmax(cv) 也返回出来
    # KL 散度 (teacher || student)
    eps = 1e-8
    log_p_s = torch.log(student_prob + eps)
    kl_loss = (teacher_prob * (torch.log(teacher_prob + eps) - log_p_s)).sum(dim=1).mean()

    # === 3. 温度软化（可选，让分布更平滑）===
    T = 2.0
    teacher_soft = F.softmax(input_data["fs_logits"] / T, dim=1)   # 若有 logits 用温度软化
    student_soft = F.softmax(model_pred["raw_cv"] / T, dim=1)
    kl_loss_T = F.kl_div(student_soft.log(), teacher_soft, reduction='batchmean') * (T ** 2)

    loss = loss_disp + 1.0 * kl_loss + 0.5 * kl_loss_T
    return loss, {'scalar/train/loss_disp': loss_disp.item(),
                  'scalar/train/loss_kl': kl_loss.item()}
```

**需要的代码改动**：

1. `banet_core.py::BANet.forward` 多返回一个 `raw_cv`（softmax 之前的 logits）或 `raw_prob`
2. `banet2d.py::BANet2D.forward` 把它放进训练输出字典
3. dataloader 加载预先生成的 `fs_prob.npy` 或在线推理 FoundationStereo

**预期收益**：EPE 提升 **0.3–0.8px**，尤其在反光区、视差跳变边界。

---

### 方案 2：边界加权 + 困难样本挖掘（零额外推理成本）

不需要 FoundationStereo 的中间输出，只用其最终视差，但**改变 loss 的权重分布**。

```python
def get_loss(self, model_pred, input_data):
    disp_gt = input_data["disp"].unsqueeze(1)
    mask = (disp_gt < self.max_disp) & (disp_gt > 0)
    disp_pred = model_pred["disp_pred"]

    # === 边界权重：视差梯度大的区域 loss × 3 ===
    grad_x = torch.abs(disp_gt[:, :, :, 1:] - disp_gt[:, :, :, :-1])
    grad_y = torch.abs(disp_gt[:, :, 1:, :] - disp_gt[:, :, :-1, :])
    grad_x = F.pad(grad_x, (0, 1, 0, 0))
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    edge_mag = grad_x + grad_y                                  # (B, 1, H, W)
    edge_weight = 1.0 + 2.0 * (edge_mag > 1.0).float()          # 边界处权重 3，其余 1

    # === 困难样本权重：error 大的像素权重更大 ===
    with torch.no_grad():
        per_pixel_err = torch.abs(disp_pred - disp_gt)          # (B, 1, H, W)
        hard_weight = 1.0 + (per_pixel_err > 2.0).float()       # error > 2px 权重 ×2

    weight = edge_weight * hard_weight
    weight = weight * mask.float()
    weight = weight / (weight.sum() + 1e-6) * mask.sum()        # 归一化保持总权重不变

    diff = F.smooth_l1_loss(disp_pred, disp_gt, reduction='none')
    loss = (diff * weight).sum() / (weight.sum() + 1e-6)
    return loss, {'scalar/train/loss': loss.item()}
```

**预期收益**：边界处 EPE 提升 **0.2–0.4px**，反光区（通常 error 大）会被自动重点关注。

---

### 方案 3：特征蒸馏（让 BANet 的中间特征对齐 FoundationStereo）

让 BANet 的 1/4 特征（`fnet` 输出的 `x4`，64 通道）拟合 FoundationStereo 的 ViT 特征（通道数不同，需要 1×1 conv 投影）。

```python
class FeatureDistillHead(nn.Module):
    """把 BANet 的 1/4 特征 (64ch) 投影到 FoundationStereo 的 1/4 特征通道数"""
    def __init__(self, student_ch=64, teacher_ch=128):
        super().__init__()
        self.proj = nn.Conv2d(student_ch, teacher_ch, 1)

    def forward(self, student_feat):
        return self.proj(student_feat)

# 在训练时
student_feat_4 = banet.banet.fnet(left)[0]                     # (B, 64, H/4, W/4)
projected = feat_head(student_feat_4)                          # (B, 128, H/4, W/4)
with torch.no_grad():
    teacher_feat = foundation_stereo.feature(left)[0][0]       # (B, 128, H/4, W/4)
loss_feat = F.smooth_l1_loss(projected, teacher_feat)
```

**预期收益**：EPE 提升 **0.1–0.3px**。对小模型容量受限时效果明显，但需要把 FoundationStereo 装在训练 pipeline 里。

---

### 方案 4：置信度引导蒸馏（高反区自动加权）

FoundationStereo 的概率分布**熵越大表示越不确定**。可以用熵作为蒸馏权重：低熵区域（FoundationStereo 自信）对 BANet 的监督更强，高熵区域（FoundationStereo 也不确定）减弱监督，避免传递错误标签。

```python
with torch.no_grad():
    teacher_prob = ...                                          # (B, 48, H/4, W/4)
    entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1)  # (B, H/4, W/4)
    # 熵小 → 老师有信心 → 监督权重大；熵大 → 老师不确定 → 监督权重小
    confidence = torch.exp(-entropy)                            # (B, H/4, W/4) ∈ (0, 1]
    confidence_full = F.interpolate(confidence.unsqueeze(1),
                                    scale_factor=4, mode='bilinear').squeeze(1)

# 用 confidence 加权所有 loss
loss = (per_pixel_loss * confidence_full * mask).sum() / (confidence_full * mask).sum()
```

**预期收益**：避免 FoundationStereo 在高反区的错误伪标签污染小模型，**间接提升高反区精度 0.2–0.5px**。

---

### 方案 5：多迭代视差 Deep Supervision

FoundationStereo 的 GRU 迭代产生 `disp_preds = [disp_1, disp_2, ..., disp_N]`，每一个都可以作为监督信号，权重逐步递增（early iter 权重小，final iter 权重大）。

```python
weights = [0.5 ** (N - 1 - i) for i in range(N)]                # [0.0625, 0.125, ..., 1.0]
loss_multi = sum(w * F.smooth_l1_loss(student_disp, teacher_disp_i)
                 for w, teacher_disp_i in zip(weights, disp_preds))
```

**预期收益**：小模型隐式学到"由粗到细的视差细化过程"，EPE 提升 **0.1–0.3px**。

---

## 推荐组合方案

按工作量从小到大递进：

### Tier 1：零额外训练开销（先做）

- 方案 2（边界加权 + 困难样本）— 不需要 FoundationStereo 中间输出，只改 loss
- 加上方案 4（置信度引导）— 离线存一份 confidence map 即可

### Tier 2：在线蒸馏（推荐）

- 方案 1（概率分布 KL 蒸馏）— 最高收益
- 方案 2 + 方案 4 一起用

### Tier 3：完整蒸馏（追求极限精度）

- 方案 1 + 方案 3（特征蒸馏）+ 方案 5（多迭代监督）
- 需要 FoundationStereo 在训练时也跑前向，显存翻倍但精度收益最大

---

## 实施步骤

### Step 1：准备蒸馏数据

```bash
# 离线生成 FoundationStereo 的中间输出（一次性）
python tools/distill_extract_teacher_outputs.py \
    --teacher_ckpt pretrained/foundationstereo/sceneflow.pth \
    --data_path /data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_merged_20260509_20260513 \
    --output_dir /data2/shendu/code/ruoyu/openstereo_add_banet/dataset/orbbec_merged_distill \
    --save prob,disp,confidence
```

每张训练图对应一个 `.npz`：
```
prob.npy        (48, H/4, W/4) float16    # 视差概率分布
disp.npy        (H, W)         float16    # 最终视差
confidence.npy  (H, W)         float16    # 置信度 (1-entropy_normalized)
```

文件大小：(48×120×160 + 480×640×2) × 2 byte ≈ **3 MB / 张**。
4187 张 × 3MB ≈ **12 GB**，磁盘可接受。

### Step 2：dataloader 加载蒸馏数据

修改 `stereo/datasets/kitti_dataset.py::KittiDataset.__getitem__`，在返回的 sample 里加：
```python
distill_path = left_img_path.replace('foundation_out', 'distill').replace('.png', '.npz')
if os.path.exists(distill_path):
    data = np.load(distill_path)
    sample['fs_prob'] = data['prob']               # (48, H/4, W/4)
    sample['fs_confidence'] = data['confidence']   # (H, W)
```

### Step 3：修改 BANet2D 的 get_loss

按上述方案 1 + 方案 2 + 方案 4 组合，参考"方案 1"的 `get_loss` 模板。

### Step 4：训练

```bash
# 使用蒸馏版 yaml（需新建）
torchrun --nproc_per_node=8 tools/train.py --dist_mode \
    --cfg_file cfgs/banet2d/banet2d_orbbec_merged_640x480_distill.yaml \
    --extra_tag distill_run
```

---

## 预期最终收益

在 Orbbec 自采数据 + 高反场景上，逐项叠加预估：

| 改进 | 预期 EPE 提升 | 累计 |
|---|---|---|
| baseline（pseudo gt 监督） | — | 100% |
| + 方案 2（边界 + 困难样本加权） | 0.2–0.4px | 95% |
| + 方案 4（置信度引导） | 0.2–0.5px | 88% |
| + 方案 1（概率分布 KL 蒸馏） | 0.3–0.8px | 78% |
| + 方案 3（特征蒸馏） | 0.1–0.3px | 73% |
| + 方案 5（多迭代 DS） | 0.1–0.3px | 70% |

**完整蒸馏后，小模型 EPE 可逼近 FoundationStereo 本身的 70-80%，但参数量只有 1/10、推理快 5-10×。** 这是知识蒸馏的核心价值。

---

## 注意事项

1. **FoundationStereo 的视差在高反区也可能错** — 用方案 4 的置信度引导能自动过滤这部分错误标签
2. **蒸馏 loss 权重需要调** — 通常 disp_loss : kl_loss = 1.0 : 1.0 起步，根据验证集调
3. **温度 T** — KL 蒸馏的温度通常 2-4 之间，T 越大分布越平滑，但太大会丢失尖锐信息
4. **不要忘记 BANet 的原 loss** — 蒸馏 loss 只是辅助，主 loss 仍是对 pseudo gt 的回归
