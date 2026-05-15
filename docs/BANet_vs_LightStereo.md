# BANet2D 与 LightStereo 架构对比

本文对比 OpenStereo 中两个轻量立体匹配模型：

- `stereo/modeling/models/lightstereo/`
- `stereo/modeling/models/banet2d/`

两者在仓库里被刻意做成**相同的数据字典与损失接口**（输入 `left/right/disp`，训练输出 `disp_pred` + 一份辅助粗视差），可直接互换做消融。

---

## 1. 总览对比

| 维度 | LightStereo | BANet2D |
|---|---|---|
| 设计理念 | 轻量、单路、靠大核注意力捕获全局上下文 | 边界感知，双路并行聚合，显式区分边界 / 平滑区 |
| Backbone | MobileNetV2 + 4 级 FPN（p2/p3/p4/c5）| MobileNetV2 + 3 级 Deconv；1/4 特征额外拼接 image stem |
| Backbone 输出通道 | [24, 32, 96, 160] | [64(=32+32), 64, 192] |
| Cost Volume | `correlation_volume`（**group-wise**，48 通道） | 自定义：`conv+desc` → 点积 → `reduce_mean` 逐通道平均 → 1 通道 / disp，拼接得 48 通道 |
| 代价聚合 | **单路** U 形 2D Mobile 残差 | **双路并行**：`cost_agg0` + `cost_agg1`，按空间注意力 α 切分与融合 |
| 左图引导注意力 | `AttentionModule`：大核条带卷积 (1×7/7×1、1×11、1×21) + 残差 | `Guided_Cost_Volume_Excitation`：1×1 通道激励（更轻） |
| 空间分流 | 无 | `SpatialAttention`：左图三尺度特征预测 sigmoid 掩膜 α |
| 视差上采样 | SPx convex upsample（refine_1 → FPN → deconv → 9 通道 softmax） | 同样结构（`spx_4` → `spx_2` → `spx`） + `context_upsample` |
| 训练损失 | `SmoothL1(full) + 0.3 · SmoothL1(1/4 双线性上采)` | 同形式，系数 `1.0 + 0.3` |
| 输出键 | `disp_pred` / `disp_4` | `disp_pred` / `disp_coarse` |

---

## 2. 数据流对比

### LightStereo

```
left, right
   │
   ├── Backbone (MobileNetV2 + FPN, 4 级输出 [p2, p3, p4, c5])
   │
   ▼
correlation_volume(features_left[0], features_right[0], D/4)   # group-wise, 48 ch
   │
   ▼
Aggregation (单路 U-Net)
   ├── conv0 + att0(left p2)         # 大核条带注意力
   ├── conv1↓ → conv2 + att2(left p3)
   ├── conv3↓ → conv4 + att4(left p4)
   ├── conv5↑ + redir2(conv2)
   └── conv6↑ + redir1(x)            → [B, 48, H/4, W/4]
   │
   ▼  reshape→softmax→disparity_regression
init_disp  [B, 1, H/4, W/4]
   │
   ▼  SPx convex upsample（refine_1/2/3）
disp_pred  [B, 1, H, W]
```

辅助分支：训练时把 `init_disp` 双线性上采到全分辨率得到 `disp_4`，用于辅助 loss。

### BANet2D

```
left, right
   │
   ├── FeatureNet (MobileNetV2 + 3 级 Deconv) → [x4, x8, x16]
   ├── stem_2 + stem_4 (额外 image stem, 1/4 32 ch)
   │     └── 与 fnet[0] 在通道维 concat   → 64 ch
   │
   ▼
CostVolume(left_1/4, right_1/4, D/4)        # 点积 + 逐通道平均，48 ch
   │
   ▼ cost_stem (3×3 conv, 48→32)
cv  [B, 32, D/4, H/4, W/4 视为通道堆叠]
   │
   ▼ SpatialAttention(features_left[0..2])  → α  (sigmoid)
   │
   ├──  α · cv   ── cost_agg0(+ Guided_Cost_Volume_Excitation × 3 尺度)
   └── (1-α)·cv  ── cost_agg1(+ Guided_Cost_Volume_Excitation × 3 尺度)
                       │
                       ▼ α · cv0 + (1-α) · cv1
                     cv_fused (48 ch)
   │
   ▼ softmax + disparity_regression
disp  [B, 1, H/4, W/4]
   │
   ▼ SPx convex upsample（spx_4 / spx_2 / spx）+ context_upsample
disp_up  [B, 1, H, W]   →  ×4 得最终视差
```

辅助分支：训练时把 1/4 视差双线性上采到全分辨率作为 `disp_coarse`。

---

## 3. 关键模块差异详解

### 3.1 Cost Volume

- **LightStereo**：`correlation_volume(...)`，**group-wise correlation**。把左右特征按 group 切分后做点积，输出 48 个通道，能保留更丰富的相似度向量。
- **BANet2D**：`CostVolume` 内部先 `BasicConv(64→32) + 1×1 desc`，逐 disp 做 `left * right`，再用一个权重固定为 `1/32` 的 1×1 卷积当 "channel mean"，得到 1 通道相似度，最后按 disparity 维度拼接成 48 通道。本质是**经典相关代价**，更轻、表达力弱于 gwc。

### 3.2 注意力机制

- **LightStereo `AttentionModule`**：
  - `1×1` 通道映射后，并联三组**深度可分离条带卷积**：(1×7, 7×1)、(1×11, 11×1)、(1×21, 21×1)。
  - 三路相加再 `1×1`，与 cost 相乘。
  - 优势：**大感受野**、参数省；对薄结构、大无纹理区域友好。

- **BANet2D `Guided_Cost_Volume_Excitation`**：
  - 仅两层 `1×1`（im_chan→im_chan→cv_chan），输出与 cost 逐元素相乘。
  - 是一种**通道激励**式注入，更轻；空间建模的活儿交给前面的 `SpatialAttention`。

### 3.3 空间分流（BANet 独有）

`SpatialAttention` 从左图特征 `[x4, x8, x16]` 中各取一份 1/4 上采、拼接、卷积、`sigmoid` 得到一张 α∈(0,1)。
之后：

```
cv_0 = α · cv          → cost_agg0  (一路专注 α 高的区域，如边界/前景)
cv_1 = (1-α) · cv      → cost_agg1  (一路专注 α 低的区域，如平滑/背景)
cv   = α · cv_0 + (1-α) · cv_1
```

这就是 BANet 名字里的 **Boundary-Aware**：用同一份 cost 走两条并行聚合路径，分别学习两类区域的视差先验，再融合。

LightStereo 不做这件事，所有像素共用同一条聚合路径。

### 3.4 额外的 image stem（BANet 独有）

```python
features_left[0]  = cat(features_left[0],  stem_4(stem_2(left)),  dim=1)
features_right[0] = cat(features_right[0], stem_4(stem_2(right)), dim=1)
```

把一条独立的 `Conv 3→16→32` stem 直接拼到 1/4 特征上，让低层纹理信息**绕过 backbone 主干**进入 cost volume，对细节边缘有利。LightStereo 直接用 backbone 的 p2 输出。

### 3.5 上采样与损失

两者几乎一致：

- 都用 SPx convex upsample（9 通道 softmax + `context_upsample`）把 1/4 视差升到全分辨率。
- 损失都是 `1.0·SmoothL1(full) + 0.3·SmoothL1(1/4 上采)`，掩膜都是 `0 < disp_gt < max_disp`。

差异仅在辅助分支的命名（`disp_4` vs `disp_coarse`）。

---

## 4. 一句话总结

- **LightStereo**：单路 U 形聚合 + 大核条带注意力，参数量小、感受野大，强调**轻量与全局上下文**。
- **BANet2D**：空间注意力 α 把 cost 分成**边界 / 非边界两路并行聚合**再融合，加上额外 image stem 注入低层纹理，强调**对视差跳变与细节边界的显式建模**。

二者其余模块（stem、上采样、损失、I/O 接口）刻意对齐，便于在同一训练管线里互换/对比。

---

## 5. 场景适配分析：加油机器人 + 车体高反

### 5.1 场景特征

加油机器人使用 Orbbec 主动 IR 双目相机，目标是精确定位油箱盖并引导机械臂插枪。核心视觉挑战：

| 挑战 | 表现 |
|---|---|
| 大块过曝 | 车顶、引擎盖、油箱盖在 IR 投影下整片饱和 → 无纹理白板 |
| 强反光边缘 | 车窗边框、油口金属圈反射高光 → 局部超强梯度 |
| 左右反光不对称 | 左右相机视角不同，同一反光斑在两图位置/强度不同 → 打破亮度一致性假设 |
| 小物体精度 | 油枪、油盖、车牌等定位精度直接决定机器人能否插准 |
| 实时性 | 闭环控制要求低延迟 |

### 5.2 两模型应对能力

| 挑战 | LightStereo-M | BANet2D | 判断依据 |
|---|---|---|---|
| 大块过曝区视差延拓 | **★★★** | ★★ | 大核条带卷积 (1×21、11×1) 感受野极大，专门 hallucinate 缺纹理区 |
| 强反光边缘锐利度 | ★★ | **★★★** | Spatial Attention 双路聚合显式区分边界/平滑区 |
| 左右反光不对称 | ★★ | ★★ | 两者都靠左图引导注意力，对左右光度差异都不算特别强健 |
| 小物体（油枪、油盖） | **★★★** | ★★ | 4 级 FPN p2 通道更精细 vs 3 级 |
| 实时性（闭环控制） | **★★★** | ★ | 单路轻量 ≈30 FPS vs 双路并行慢 1.5–2× |
| 小数据集 (~4k) 泛化 | **★★★** | ★★ | 参数少不易过拟合 vs 双 aggregation 容量大 |

### 5.3 推荐方案

**主用 LightStereo-M**，原因（按权重排序）：

1. **实时性**：加油机器人是闭环实时任务，视差延迟直接影响插枪精度与安全。
2. **大块无纹理**：高反核心问题是"大块过曝无纹理"，恰好是 LightStereo 大核条带卷积的最强项。
3. **小数据集**：仅 ~4k 训练样本，BANet 容量更大反而容易过拟合。
4. **小物体**：4 级 FPN 在 1/4 输出上保留更多细节，有利于油口定位。

**BANet 留作对照实验**，适用场景：
- LightStereo 在车窗边框、油口金属圈等**强反光锐边**处视差跳变严重
- 实时性可以放宽（不做闭环控制，只做开环规划）

### 5.4 模型选择之外的建议

以下措施对精度的影响往往**远大于换模型**：

1. **数据增广加过曝模拟**：`aligned` yaml 里 `GAMMA: [1,1,1,1]` 是 identity，改成 `[0.7, 1.5, 1.0, 1.0]` 可模拟过曝/欠曝。
2. **采集困难样本**：刻意采逆光、强日光、金属车（白漆/银漆）、深色玻璃下的样本。
3. **置信度后处理**：用左右一致性检查 + 视差梯度阈值过滤反光不可靠区，给上层规划器返回"不确定"标志，比让模型硬猜更安全。
4. **传感器层**：Orbbec 主动 IR 投影器有助于规避漆面颜色差异，但对**金属镜面反射**无效。如果允许，加偏振滤光片或调整投影器到相机基线/光轴关系，常常比模型层改进显著得多。
