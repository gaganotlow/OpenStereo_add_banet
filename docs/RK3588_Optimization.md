# RK3588 部署优化方案

## 场景背景

- **应用**：加油机器人，车体高反，目标距离 0.5–2m
- **相机**：Orbbec 主动 IR 双目，480×640
- **部署平台**：RK3588（3×NPU core，原生支持 ≤7×7 Conv2d / BN / ReLU / Depthwise）
- **训练策略**：FoundationStereo 伪真值 → 轻量模型蒸馏
- **目标**：高精度 + 实时（≥30 FPS）

---

## 模型选择：BANet2D 优于 LightStereo

### RK3588 RKNN 兼容性对比

| 兼容性项 | BANet2D | LightStereo-M |
|---|---|---|
| 3D 卷积 | **无** | **无** |
| ViT / 动态 shape | **无** | **无** |
| 大核卷积 (>7×7) | **无** | **12 个** (1×11, 11×1, 1×21, 21×1) |
| InstanceNorm | **无** | **3 个** |
| ConvTranspose2d | 9 个 (可替换) | 有 (可替换) |
| 所有 op ≤ 7×7 | **是** | **否** |

**结论**：LightStereo 的大核条带卷积在 RKNN 上会 fallback 到 CPU 或被拆成多个小核，性能崩塌。BANet2D 全部是标准 2D 小核卷积，NPU 原生支持。

### 模型参数与 A800 实测延迟（参考）

| 模型 | 参数量 | A800 延迟 (480×640) | A800 FPS |
|---|---|---|---|
| BANet2D | 5.46M | 32.4 ms | 30.8 |
| LightStereo-M | 7.64M | 28.3 ms | 35.3 |

### BANet2D 各阶段耗时占比（A800 实测）

| 阶段 | 耗时 | 占比 |
|---|---|---|
| 特征提取 (MobileNetV2×2 + stem) | 10.7 ms | 33% |
| Cost Volume 构建 | 5.3 ms | 16% |
| 代价聚合 (双路 + 空间注意力) | 14.8 ms | **46%** |
| 回归 + 上采样 | 1.6 ms | 5% |

---

## 优化方案（按优先级）

### P0：零改动 / 配置改动，直接提升

#### 1. 缩小 MAX_DISP

加油场景目标距离 0.5–2m，Orbbec 基线 ~50mm，焦距 ~400px：

```
最大视差 ≈ baseline × focal / min_depth = 50 × 400 / 500 ≈ 40px
```

实际最大视差大概率 < 96。把 `MAX_DISP: 192` 改成 `MAX_DISP: 96`：

- Cost volume 从 48 通道降到 24 通道
- **聚合层计算量直接减半**（最大瓶颈，占 46% 延迟）
- 精度不降（超出范围的视差本来就匹配不到）

**操作**：yaml 里改一行 `MAX_DISP: 96`，重新训练。

#### 2. 输入分辨率降采样

480×640 → 384×512 或 320×480：

- 1/4 特征图从 120×160 → 96×128 或 80×120
- 计算量降 ~35-50%
- 视差精度按比例缩放，对加油场景（目标大、距离近）影响小

---

### P1：小改动大收益

#### 3. ConvTranspose2d → Resize + Conv2d

RKNN 对 ConvTranspose2d 支持但效率低，且容易产生棋盘格伪影。

```python
# 原来
x = conv_transpose(x)  # ConvTranspose2d(in, out, k=3, s=2)

# 替换为（RKNN 友好）
x = F.interpolate(x, scale_factor=2, mode='nearest')
x = conv(x)  # Conv2d(in, out, k=3, s=1, p=1)
```

BANet2D 有 9 个 ConvTranspose2d，全部替换后：
- RKNN 推理更快（nearest resize 是零计算）
- 数值更稳定
- 精度无损（重新训练即可）

#### 4. CostVolume for 循环 → 矩阵操作

当前 `banet_core.py:29` 用 Python for 循环逐视差计算，ONNX 导出后变成大量 Slice+Pad 节点。

```python
# 原来（48 次循环）
cv = []
for i in range(maxdisp):
    cost = reduce_mean(left[:,:,:,i:] * right[:,:,:,:-i])
    cost = F.pad(cost, (i, 0, 0, 0))
    cv.append(cost)
return torch.cat(cv, dim=1)

# 优化为（一次 shift + batch matmul）
right_shifted = torch.stack([
    F.pad(right[:,:,:,d:], (d, 0)) if d > 0 else right
    for d in range(maxdisp)
], dim=2)  # [B, C, D, H, W]
cv = (left.unsqueeze(2) * right_shifted).mean(dim=1)  # [B, D, H, W]
```

ONNX 图更简洁，RKNN 能更好地做算子融合。

#### 5. 知识蒸馏增强

当前只用 FoundationStereo 的最终视差作为伪真值（L1/SmoothL1 loss）。可以更进一步：

| 蒸馏方式 | 做法 | 收益 |
|---|---|---|
| **Soft label** | FoundationStereo 输出视差概率分布（不只 argmax）作为 soft target，用 KL 散度训练 BANet 的 softmax | 让小模型学到"不确定性"，高反区不硬猜 |
| **边界加权 loss** | 在视差梯度大的区域（Sobel 检测）加大 loss 权重 ×3 | 让小模型重点学边界，弥补容量不足 |
| **困难样本挖掘** | 每 epoch 统计 per-pixel error，下一 epoch 对 error > 2px 的区域加权 | 高反区自动被重点关注 |

---

### P2：架构微调（需重新训练）

#### 6. 聚合层通道剪枝

BANet 双路聚合各 32 通道 → 剪到 24 通道：

- 参数从 3.29M → ~1.85M
- 聚合延迟降 ~25%
- 配合蒸馏训练，精度损失可控制在 <0.2px EPE

#### 7. Backbone 轻量化

MobileNetV2 → MobileNetV3-Small 或 ShuffleNetV2：

- 参数从 1.95M → ~1.2M
- 特征提取延迟降 ~30%
- RK3588 对 depthwise conv + SE block 有专门硬件加速

#### 8. 双路聚合 → 单路 + 残差

BANet 的 `cost_agg0` 和 `cost_agg1` 结构完全相同（各 1.644M）。改成：

```
fused_cv = shared_agg(cv) + lightweight_residual(α * cv)
```

- 省掉一半聚合计算（1.644M → 共享 1.644M + 残差 ~0.2M）
- 保留边界感知能力（α 分流仍在）
- 需要重新训练验证

---

### P3：部署工程优化

#### 9. RKNN 量化策略

BANet2D 全是 Conv2d + BN + ReLU6，非常适合 INT8 量化：

| 量化方式 | 精度损失 | 速度提升 |
|---|---|---|
| Post-training INT8 | -0.5~1.0px | 2-3× |
| QAT (量化感知训练) | -0.2~0.3px | 2-3× |
| 混合精度 (softmax 回归 FP16，其余 INT8) | -0.1~0.2px | 2× |

**推荐**：最后 10 epoch 加入 fake quantize 做 QAT，比 post-training 量化精度高 0.3-0.5px。

#### 10. 三核流水线并行

RK3588 有 3 个 NPU core（各 2 TOPS），可以分段：

```
Core 0: 左图特征提取 (fnet)
Core 1: 右图特征提取 (fnet)
Core 2: Cost Volume + 聚合 + 回归 + 上采样
```

三段流水线并行，理论吞吐量提升 2-3×（延迟不变但帧率翻倍）。

#### 11. 模型分段导出

RKNN 对单个大模型优化有限，分成 2-3 个子模型分别转换：

```
Part A: left_img, right_img → left_feat, right_feat  (纯 CNN，INT8)
Part B: left_feat, right_feat → cost_volume           (含 shift 操作，INT8)
Part C: cost_volume → disparity                       (含 softmax，FP16)
```

每段独立量化，softmax 段保持 FP16 避免精度崩塌。

---

## 预期性能（RK3588 NPU）

| 优化配置 | 预估延迟 | 精度影响 | 工作量 |
|---|---|---|---|
| 原始 BANet2D (FP16, 480×640, D=192) | ~150-200ms | baseline | — |
| + MAX_DISP=96 | ~100-130ms | 无损 | 5 分钟 |
| + ConvTranspose→Resize+Conv | ~90-110ms | 无损 | 1 天 |
| + INT8 量化 (QAT) | ~50-70ms | -0.2px | 2 天 |
| + 通道剪枝 24ch + 蒸馏 | ~35-50ms | -0.3px (蒸馏补回) | 3 天 |
| + 输入 384×512 | ~25-40ms | 视差按比例缩放 | 配置改动 |
| + 三核流水线 | 吞吐 60+ FPS | 同上 | 2 天 |

**目标 30 FPS（33ms）在做完 P0 + P1 + INT8 量化后可达。**

---

## 推荐执行路径

```
Step 1: 测实际数据最大视差 → 确定 MAX_DISP 能缩到多少
Step 2: 替换 ConvTranspose2d → 确保 ONNX 导出干净
Step 3: ONNX → RKNN 转换验证 → 确认所有 op 走 NPU
Step 4: INT8 QAT 训练 → FoundationStereo 伪真值 + 蒸馏
Step 5: 按需做通道剪枝 / backbone 替换
Step 6: 三核流水线部署
```

---

## 附：LightStereo 若要部署 RK3588 需要的额外改动

如果仍想尝试 LightStereo，需要：

1. **大核条带卷积 (1×21, 21×1) → 多个 1×7 级联**：3 个 1×7 depthwise conv 级联 = 等效 1×21 感受野，但每个都是 RKNN 原生支持
2. **InstanceNorm → BatchNorm**：重新训练，精度可能微降
3. 改完后本质上已经不是原版 LightStereo，且大核的"一次性大感受野"优势被稀释

**因此对 RK3588 部署，直接优化 BANet2D 是更务实的路径。**
