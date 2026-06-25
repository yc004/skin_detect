# 基于改进 ConvNeXt 的多类别皮肤疾病图像分类研究

## 技术文档 · Technical Report

> 本文档为论文撰写提供完整的技术方案、实验设计和结果分析框架。文中指标为占位符，请在训练完成后替换为实际数据。

---

## 摘要

皮肤疾病是全球最常见的健康问题之一，早期准确诊断对治疗效果至关重要。本文提出一种基于改进 ConvNeXt 架构的 22 类皮肤疾病自动分类方法。针对细粒度皮肤病分类中类间差异小、类内差异大、数据分布不均衡等挑战，本文系统性地研究了四种改进策略：（1）ConvNeXt V2 骨干网络引入全局响应归一化（GRN）以缓解特征坍缩；（2）广义均值池化（GeM Pooling）替代传统平均池化以增强细节特征表达；（3）多尺度特征融合（Multi-Scale Fusion）聚合不同感受野的特征信息；（4）指数滑动平均（EMA）权重平滑提升模型泛化能力。在包含 15,444 张图像的 Skin Disease Dataset 上进行消融实验，结果表明 [填入最佳配置] 达到 [填入准确率] 的分类准确率，较基线模型提升 [填入提升幅度]。

**关键词**: 皮肤疾病分类；ConvNeXt；广义均值池化；多尺度特征融合；消融研究

---

## 1. 引言

皮肤病是全球疾病负担的重要组成部分，影响各年龄段人群。传统诊断依赖皮肤科医生的目视检查和皮肤镜检查，受限于医疗资源分布不均和医生经验差异。基于深度学习的自动分类系统能够辅助临床决策，提高筛查效率。

近年来，卷积神经网络（CNN）在医学图像分析领域取得了显著进展。从 ResNet 到 EfficientNet，再到 Vision Transformer，模型架构不断演进。ConvNeXt 由 Liu 等人 (2022) 提出，将现代 CNN 的设计理念系统性地推向极致，在 ImageNet 分类任务上取得了与 Transformer 相当的性能。ConvNeXt V2 进一步引入全局响应归一化（GRN）和全卷积掩码自编码器（FCMAE）预训练，解决了大模型训练中的特征坍缩问题。

然而，将通用图像分类模型直接应用于皮肤疾病分类面临三个核心挑战：

1. **细粒度识别**: 不同皮肤病（如光化性角化病 vs. 脂溢性角化病）在外观上高度相似，需要模型捕获细微的纹理和颜色差异。
2. **类内差异大**: 同一疾病在不同患者、不同部位、不同阶段的形态差异显著。
3. **数据不均衡**: 常见病（正常皮肤 1,651 例）与罕见病（念珠菌病 248 例）样本量差异达 6 倍以上。

针对上述挑战，本文从池化策略、特征融合、权重平滑和骨干网络四个维度对 ConvNeXt 进行系统性的改进和消融研究。

---

## 2. 数据集

### 2.1 数据来源

实验使用 Skin Disease Dataset，一个公开的皮肤疾病分类数据集。数据集包含 22 类皮肤疾病的临床和皮肤镜图像。

### 2.2 数据分布

| 类别（英文） | 风险等级 | 训练集 | 测试集 | 合计 |
|-------------|---------|--------|--------|------|
| Skin Cancer | HIGH | 693 | 77 | 770 |
| Actinic Keratosis | MEDIUM | 748 | 83 | 831 |
| Lupus | MEDIUM | 311 | 34 | 345 |
| Vasculitis | MEDIUM | 461 | 52 | 513 |
| Bullous | MEDIUM | 504 | 55 | 559 |
| Acne | LOW | 593 | 65 | 658 |
| Benign Tumors | LOW | 1,093 | 121 | 1,214 |
| Candidiasis | LOW | 248 | 27 | 275 |
| Drug Eruption | LOW | 547 | 61 | 608 |
| Eczema | LOW | 1,010 | 112 | 1,122 |
| Infestations/Bites | LOW | 524 | 60 | 584 |
| Lichen | LOW | 553 | 61 | 614 |
| Moles | LOW | 361 | 40 | 401 |
| Psoriasis | LOW | 820 | 88 | 908 |
| Rosacea | LOW | 254 | 28 | 282 |
| Seborrheic Keratoses | LOW | 455 | 51 | 506 |
| Sun/Sunlight Damage | LOW | 312 | 34 | 346 |
| Tinea | LOW | 923 | 102 | 1,025 |
| Unknown/Normal | LOW | 1,651 | 189 | 1,840 |
| Vascular Tumors | LOW | 543 | 60 | 603 |
| Vitiligo | LOW | 714 | 82 | 796 |
| Warts | LOW | 580 | 64 | 644 |
| **合计** | | **13,898** | **1,546** | **15,444** |

### 2.3 数据特点分析

- **类别不均衡**: 样本最多的 Unknown/Normal（1,840 例）与最少的 Candidiasis（275 例）相差约 6.7 倍。实验中使用 Label Smoothing 缓解长尾效应，未采用过采样以避免过拟合。
- **细粒度分类**: 部分类别在视觉上高度相似（如 Actinic Keratosis 与 Seborrheic Keratoses，Benign Tumors 与 Moles），对模型的判别能力提出了较高要求。
- **临床风险分级**: 根据医学意义将 22 类划分为 HIGH（1 类）、MEDIUM（4 类）、LOW（17 类）三个风险等级，有助于临床分诊。

### 2.4 数据预处理与增强

训练阶段采用以下数据增强策略：

| 增强方法 | 参数 | 作用 |
|---------|------|------|
| RandomResizedCrop | scale=(0.7, 1.0) | 尺度不变性 |
| RandomHorizontalFlip | p=0.5 | 左右翻转不变性 |
| RandomVerticalFlip | p=0.3 | 上下翻转（皮肤镜方向不固定） |
| RandomRotation | 20° | 旋转不变性 |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.1 | 光照/采集条件变化 |
| RandomAffine | translate=(0.05, 0.05) | 平移不变性 |
| Normalize | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] | ImageNet 标准化 |

验证/测试阶段：Resize(256) → CenterCrop(224) → Normalize。

---

## 3. 方法

### 3.1 基线模型: ConvNeXt-Tiny

ConvNeXt 由 Liu 等人 (2022) 提出，其核心设计理念是"现代化 ResNet"——系统性地将 Swin Transformer 的设计元素（stage 计算比例、patchify stem、大卷积核、Layer Normalization、GELU 激活函数等）引入 CNN 架构，在不改变卷积本质的前提下取得了与 Transformer 相匹配的性能。

ConvNeXt-Tiny 的架构参数：
- 四个 stage，通道数分别为 [96, 192, 384, 768]
- 各 stage 的 block 数: [3, 3, 9, 3]
- 总参数量: ~28M
- 输入尺寸: 224×224

基线模型使用 ImageNet-22k 预训练权重初始化，最后全局平均池化（Global Average Pooling）后接全连接分类层。

### 3.2 ConvNeXt V2: 全局响应归一化（GRN）

ConvNeXt V2（Woo et al., 2023）针对大模型训练中的"特征坍缩"（feature collapse）问题，提出了两个关键改进：

**全局响应归一化（Global Response Normalization, GRN）**:

$$\text{GRN}(X_i) = \gamma \cdot \frac{X_i}{\sqrt{\|X_i\|^2 + \epsilon}} + \beta$$

其中 $X_i \in \mathbb{R}^{C}$ 是空间位置 $i$ 的特征向量，$\|\cdot\|$ 为 L2 范数，$\gamma, \beta$ 为可学习参数。GRN 通过对每个空间位置的特征进行通道维度的归一化，增强了特征多样性，抑制了特征坍缩。

**全卷积掩码自编码器（FCMAE）**: ConvNeXt V2 使用 FCMAE 自监督预训练，掩码比例高达 60%，稀疏卷积实现高效训练。

本实验使用 ConvNeXtV2-Tiny（FCMAE 预训练 + ImageNet-22k 微调）作为骨干网络的升级选项。

### 3.3 广义均值池化（GeM Pooling）

传统全局平均池化（GAP）对所有空间位置一视同仁，可能稀释判别性细节。广义均值池化（Generalized Mean Pooling）由 Radenović 等人 (2018) 在图像检索中提出，通过可学习的幂参数 $p$ 在平均池化和最大池化之间连续调节：

$$f^{(g)} = \left[ \frac{1}{|\Omega|} \sum_{x \in \Omega} x^p \right]^{\frac{1}{p}}$$

其中 $\Omega$ 为空间域，$p \geq 1$ 为可学习参数。当 $p=1$ 时退化为平均池化，$p \to \infty$ 时逼近最大池化。GeM 通过增大 $p$ 赋予高响应区域更大的权重，同时保留全局信息。

本文初始化 $p=3.0$，约束 $p \in [1, 10]$ 以保证数值稳定性。

### 3.4 多尺度特征融合（Multi-Scale Feature Fusion）

ConvNeXt 各 stage 提取不同粒度的特征：浅层（Stage 1-2）编码纹理、边缘等局部细节，深层（Stage 3-4）编码语义、形状等全局信息。对于细粒度皮肤疾病分类，单一尺度特征可能不足以同时捕获全局病灶形态和局部纹理差异。

本文提出的多尺度融合策略：
1. 提取 Stage 2、Stage 3、Stage 4 的特征图，空间尺寸分别为 H/8、H/16、H/32
2. 各 stage 特征通过 1×1 卷积投影到统一的 256 维空间，接 BatchNorm + SiLU
3. 投影后的特征经池化（AvgPool 或 GeM）得到 (B, 256) 向量
4. 三个 stage 的特征拼接为 (B, 768) 的多尺度描述子
5. 通过两层 MLP（768 → 512 → 22）进行分类

### 3.5 指数滑动平均（EMA）

EMA（Exponential Moving Average）在训练过程中维护模型参数的滑动平均副本：

$$\theta_{\text{shadow}}^{(t)} = \alpha \cdot \theta_{\text{shadow}}^{(t-1)} + (1-\alpha) \cdot \theta^{(t)}$$

其中 $\alpha = 0.999$ 为衰减率。推理时使用 shadow 权重，具有以下优势：
- 平滑训练过程中的参数波动，降低对单 batch 噪声的敏感性
- 等效于模型集成（Polyak averaging），提升泛化能力
- 零推理开销（仅多存储一份权重副本）

### 3.6 训练策略

| 超参数 | 值 |
|--------|-----|
| 优化器 | AdamW |
| 初始学习率 | 1e-4 |
| 权重衰减 | 0.05 |
| 学习率调度 | CosineAnnealingWarmRestarts, T_0=epochs/3, T_mult=2 |
| 最小学习率 | 1e-6 |
| 损失函数 | CrossEntropyLoss(label_smoothing=0.1) |
| 混合精度 | torch.cuda.amp (GradScaler) |
| Batch Size | 32 |
| 最大 Epoch | 50 |
| Early Stopping | patience=10, monitor=val_acc |
| 随机种子 | 42 |

**Label Smoothing**: 将硬标签 $y \in \{0, 1\}$ 平滑为 $\tilde{y} = (1-\epsilon) \cdot y + \epsilon / K$，其中 $\epsilon=0.1$，$K=22$。该技术减轻过拟合，提升模型校准度，有助于处理类别不均衡。

---

## 4. 实验设计

### 4.1 消融研究设计

为系统评估各组件的贡献，设计以下消融实验：

| 实验 | 骨干网络 | 池化 | 多尺度 | EMA | 分辨率 | 说明 |
|------|---------|------|--------|-----|--------|------|
| E1 (Baseline) | ConvNeXt V1 | Avg | ✗ | ✗ | 224 | 基线 |
| E2 | ConvNeXt V2 | Avg | ✗ | ✗ | 224 | 仅 V2 |
| E3 | ConvNeXt V2 | GeM | ✗ | ✗ | 224 | V2 + GeM |
| E4 | ConvNeXt V2 | GeM | ✗ | ✓ | 224 | V2 + GeM + EMA |
| E5 | ConvNeXt V2 | GeM | ✓ | ✓ | 224 | 全组件 |
| E6 | ConvNeXt V2 | GeM | ✓ | ✓ | 384 | 高分辨率 |
| E7 | ConvNeXt V1 | GeM | ✓ | ✓ | 224 | V1 对照 |

### 4.2 评估指标

- **总体准确率** (Overall Accuracy)
- **精确率** (Precision, per-class & macro-avg)
- **召回率** (Recall, per-class & macro-avg)
- **F1 分数** (F1-score, per-class & macro-avg)
- **AUC** (One-vs-Rest ROC 曲线下面积, per-class & micro-avg)
- **混淆矩阵** (Confusion Matrix)
- **参数量** (Model Parameters)
- **推理时间** (Inference Time, ms/image)

### 4.3 训练命令

```bash
# E1: 基线
python train.py --model convnext_tiny --pooling avg --output runs/E1_baseline

# E2: V2
python train.py --model convnextv2_tiny --pooling avg --output runs/E2_v2

# E3: V2 + GeM
python train.py --model convnextv2_tiny --pooling gem --output runs/E3_v2_gem

# E4: V2 + GeM + EMA
python train.py --model convnextv2_tiny --pooling gem --ema --output runs/E4_v2_gem_ema

# E5: 全组件
python train.py --model convnextv2_tiny --pooling gem --ema --multi-scale --output runs/E5_full

# E6: 高分辨率
python train.py --model convnextv2_tiny --pooling gem --ema --multi-scale --img-size 384 --output runs/E6_hires

# E7: V1 对照
python train.py --model convnext_tiny --pooling gem --ema --multi-scale --output runs/E7_v1_multiscale
```

---

## 5. 实验结果（模板）

> ⚠️ **请在训练完成后将实际数据填入以下表格。** 所有实验指标均自动保存于 `runs/<实验名>/experiment_summary.json`。

### 5.1 主要结果

| 实验 | Val Acc | Test Acc | Macro F1 | Micro AUC | 参数量 | 训练时间 |
|------|---------|----------|----------|-----------|--------|----------|
| E1 (Baseline) | _._ _ _ | _._ _ _ | _._ _ _ | _._ _ _ | 28.6M | _h |
| E2 (V2) | _._ _ _ | _._ _ _ | _._ _ _ | _._ _ _ | 28.6M | _h |
| E3 (V2+GeM) | _._ _ _ | _._ _ _ | _._ _ _ | _._ _ _ | 28.6M | _h |
| E4 (V2+GeM+EMA) | _._ _ _ | _._ _ _ | _._ _ _ | _._ _ _ | 28.6M | _h |
| E5 (Full) | _._ _ _ | _._ _ _ | _._ _ _ | _._ _ _ | _._M | _h |
| E6 (384px) | _._ _ _ | _._ _ _ | _._ _ _ | _._ _ _ | _._M | _h |
| E7 (V1对照) | _._ _ _ | _._ _ _ | _._ _ _ | _._ _ _ | _._M | _h |

### 5.2 消融分析模板

**V1 → V2 的影响**: [V2 引入 GRN 层，对比 E1 和 E2 的差异，分析 GRN 对细粒度分类的作用]

**Avg → GeM 的影响**: [对比 E2 和 E3 的差异，分析可学习池化参数的收敛行为]

**无 EMA → EMA 的影响**: [对比 E3 和 E4 的差异，分析训练曲线平滑度和泛化能力变化]

**单尺度 → 多尺度的影响**: [对比 E4 和 E5 的差异，分析各 stage 特征的互补性]

**分辨率提升的影响**: [对比 E5 和 E6 的差异，分析细节信息增益与计算成本的权衡]

### 5.3 易混淆类别分析

[分析混淆矩阵中高频误分类对，如 Actinic Keratosis ↔ Seborrheic Keratoses，讨论其视觉相似性原因和可能的改进方向]

### 5.4 风险分级性能

| 风险等级 | Precision | Recall | F1 | 说明 |
|----------|-----------|--------|-----|------|
| HIGH (1类) | _._ _ _ | _._ _ _ | _._ _ _ | Skin Cancer 识别能力 |
| MEDIUM (4类) | _._ _ _ | _._ _ _ | _._ _ _ | 中等风险疾病 |
| LOW (17类) | _._ _ _ | _._ _ _ | _._ _ _ | 良性/常见疾病 |

---

## 6. 讨论

### 6.1 关键发现

[填入关键实验结果和发现]

### 6.2 局限性与未来工作

1. **数据集局限**: 当前数据集为公开通用皮肤病数据集，部分类别样本量偏少（<300 例）。未来可结合多个公开数据集（如 ISIC、PAD-UFES-20）以增加样本多样性和覆盖范围。

2. **仅分类信息**: 模型仅输出类别，不包含病灶定位。后续可引入 Grad-CAM 作为弱监督分割信号，同时输出诊断类别和病灶区域。

3. **无元数据融合**: 真实临床场景中，患者年龄、性别、病灶部位等信息对诊断具有辅助作用。未来的多模态模型可融合图像特征和临床元数据。

4. **可解释性**: 当前 Grad-CAM 提供了初步的可视化解释，但对于高风险决策，需要更严格的解释方法（如概念瓶颈模型、对比解释）。

5. **前瞻性验证**: 模型仅在回顾性数据集上评估，实际临床部署前需进行前瞻性验证。

---

## 7. 结论

本文系统性地研究了 ConvNeXt 架构在 22 类皮肤疾病分类任务上的改进方案。通过引入 ConvNeXt V2 骨干网络、广义均值池化、多尺度特征融合和 EMA 权重平滑，[填入主要结论]。实验结果表明，[填入最佳配置] 在 [数据集名称] 上达到了 [最佳准确率] 的分类性能，验证了各改进组件的有效性。

---

## 8. 项目代码

项目代码已开源：`https://github.com/yc004/skin_detect`

主要文件说明：
- `train.py` — 训练脚本，支持所有消融变体
- `detect.py` — 推理脚本，支持 Grad-CAM 可视化
- `demo.py` — Gradio Web 演示界面
- `models/modules.py` — GeM Pooling、多尺度融合、EMA 实现
- `utils/visualize.py` — 混淆矩阵、ROC 曲线、per-class 指标图表生成
- `runs/<实验名>/` — 每个实验的完整输出（模型、指标、图表）

---

## 参考文献

[1] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. *CVPR 2022*.

[2] Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I. S., & Xie, S. (2023). ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders. *CVPR 2023*.

[3] Radenović, F., Tolias, G., & Chum, O. (2018). Fine-tuning CNN Image Retrieval with No Human Annotation. *IEEE TPAMI*, 41(7), 1655-1668.

[4] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.

[5] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR 2016*.

[6] Polyak, B. T., & Juditsky, A. B. (1992). Acceleration of Stochastic Approximation by Averaging. *SIAM Journal on Control and Optimization*, 30(4), 838-855.

[7] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.

[8] Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.

[9] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*.

---

> 📅 文档版本: v1.0  
> 👤 作者: [你的名字]  
> 📧 联系方式: [你的邮箱]  
> 🏫 单位: [你的学校/机构]
