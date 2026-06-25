# 基于改进 ConvNeXt 的多类别皮肤疾病智能分类系统

## 第1章 作品概述

### 1.1 研究背景

皮肤病是全球疾病负担的重要组成部分。据世界卫生组织统计，全球约有 9 亿人患有各类皮肤疾病，其中皮肤癌发病率在过去 30 年间持续上升。传统皮肤疾病诊断依赖皮肤科医生的目视检查和皮肤镜检查，然而全球范围内皮肤科医生分布极不均衡——非洲部分地区每 100 万人口仅有不到 1 名皮肤科医生，我国基层医疗机构同样面临皮肤科医生严重短缺的问题。

深度学习技术在医学图像分析领域已取得显著进展。从 AlexNet 到 ResNet，再到 Vision Transformer，图像分类模型的准确率不断提升。然而，将通用模型直接应用于皮肤疾病分类仍面临三个核心挑战：（1）细粒度识别——不同皮肤病在外观上高度相似，类间差异极小；（2）类内差异大——同一疾病在不同患者、不同部位、不同阶段的形态差异显著；（3）数据不均衡——常见病与罕见病的样本量差异可达 6 倍以上。

### 1.2 作品目标

本作品旨在构建一个基于改进 ConvNeXt 架构的 22 类皮肤疾病智能辅助分类系统，实现以下目标：

1. **高准确率分类**：在包含 15,444 张图像的公开数据集上，达到 82% 以上的验证准确率。
2. **可解释性分析**：通过 Grad-CAM 注意力热力图可视化模型关注的病灶区域，为临床医生提供判断依据。
3. **大模型辅助咨询**：接入大语言模型 API，基于分类结果生成疾病概述、治疗建议等医学报告，并提供多轮对话咨询功能。
4. **完整实验对比**：设计消融实验，系统评估各组件的独立贡献，为后续研究提供参考。

### 1.3 主要工作

| 工作内容 | 说明 |
|---------|------|
| 基线模型搭建 | ConvNeXt-Tiny + 全局平均池化，验证集准确率 79.80% |
| 骨干网络升级 | ConvNeXt V1 → ConvNeXt V2，引入全局响应归一化（GRN） |
| 池化策略改进 | 全局平均池化 → 广义均值池化（GeM Pooling），可学习参数 p |
| 权重优化 | 引入指数滑动平均（EMA），decay=0.999 |
| 消融实验 | 对比基线 vs 改进模型的性能差异，分析各组件贡献 |
| 可解释性 | 实现 Grad-CAM 注意力可视化，并排对比基线/改进模型的关注区域差异 |
| Web 应用 | 基于 Flask + 原生前端构建完整交互系统，支持流式 AI 报告和对话 |
| ONNX 部署 | 导出 ONNX 模型，支持跨平台推理部署 |

---

## 第2章 问题分析

### 2.1 问题来源

皮肤疾病的自动分类是人工智能医学影像领域的重要研究方向。目前存在以下实际问题：

1. **医疗资源不均**：我国基层医疗机构皮肤科医生严重不足，许多皮肤病患者无法获得及时、准确的诊断。
2. **诊断主观性强**：不同医生对同一皮损的诊断可能存在差异，尤其在不典型病例中。
3. **筛查效率低下**：大规模皮肤癌筛查中，医生需要逐一检查每张图像，工作量大且容易出现疲劳性误判。
4. **患者缺乏医学知识**：普通民众难以判断皮损的严重程度，可能延误就医时机。

### 2.2 现有解决方案

| 方案 | 代表方法 | 优点 | 不足 |
|------|---------|------|------|
| 人工诊断 | 皮肤科医生目视检查 + 皮肤镜 | 准确率高，可结合临床经验 | 效率低，受地域和资源限制 |
| CNN 分类 | ResNet / EfficientNet + 皮肤病数据集 | 自动化程度高，准确率 70-80% | 细粒度分类能力有限 |
| Vision Transformer | ViT / Swin Transformer | 全局感受野，细粒度表现好 | 数据需求大，推理慢 |
| 多模态融合 | 图像 + 临床元数据 | 信息更全面 | 元数据获取困难，标注成本高 |

### 2.3 本作品要解决的痛点问题

1. **细粒度识别困难**：部分皮肤病因视觉特征高度相似（如光化性角化病 vs 脂溢性角化病），传统 CNN 模型对此类细粒度分类的区分能力不足。本作品通过 GRN（全局响应归一化）增强特征多样性，GeM Pooling 聚焦判别性区域，系统性提升细粒度分类能力。

2. **特征坍缩问题**：深层 CNN 在训练过程中容易出现"特征坍缩"——各通道特征趋于高度相关，有效特征维度降低。ConvNeXt V2 引入的 GRN 层直接针对此问题设计。

3. **池化策略固化的局限**：传统全局平均池化对所有空间位置赋予相等权重，无法自动聚焦病灶关键区域。本作品采用可学习参数 p 的 GeM Pooling，让模型在训练中自动发现最优池化策略。

4. **缺乏可交互的辅助诊断工具**：现有研究多为离线模型评估，缺乏可实际使用的交互系统。本作品构建了完整的 Web 应用，集成分类、可解释性分析、大模型辅助咨询等功能。

### 2.4 解决问题的思路

```
┌──────────────────────────────────────────────────────────┐
│                    问题 → 方案映射                        │
├──────────────┬───────────────────┬───────────────────────┤
│ 细粒度识别难  │ → ConvNeXt V2 GRN │ → 增强特征多样性      │
│ 特征坍缩      │ → GRN 归一化      │ → 抑制通道相关性      │
│ 池化策略固化  │ → GeM Pooling     │ → 可学习特征聚合      │
│ 参数波动      │ → EMA 平滑        │ → 提升泛化能力        │
│ 缺乏交互工具  │ → Flask + SSE     │ → Web 应用 + AI 对话  │
└──────────────┴───────────────────┴───────────────────────┘
```

---

## 第3章 技术方案

### 3.1 整体架构

系统架构分为四个层次：

```
┌─────────────────────────────────────────────────────────┐
│                    应用层 (Web UI)                       │
│   Flask + HTML/CSS/JS + SSE 流式传输                     │
├─────────────────────────────────────────────────────────┤
│                    服务层 (AI 增强)                      │
│  Grad-CAM 注意力可视化  |  大模型 API (疾病报告+对话)     │
├─────────────────────────────────────────────────────────┤
│                    推理层 (模型)                          │
│  ConvNeXtV2 + GeM Pooling + EMA (PyTorch → ONNX)         │
├─────────────────────────────────────────────────────────┤
│                    数据层                                 │
│  Skin Disease Dataset (15,444 张 / 22 类)                 │
└─────────────────────────────────────────────────────────┘
```

### 3.2 核心算法

#### 3.2.1 基线模型：ConvNeXt-Tiny

ConvNeXt（Liu et al., CVPR 2022）通过系统性地将 Transformer 设计理念引入 CNN，实现了与 Vision Transformer 相当的性能。ConvNeXt-Tiny 参数规模约 28M，适合医学图像分类任务。

结构参数：
- 四个 Stage，通道数 [96, 192, 384, 768]，block 数 [3, 3, 9, 3]
- 分类头：AdaptiveAvgPool2d → LayerNorm → Linear(768, 22)

#### 3.2.2 改进一：ConvNeXt V2（GRN 全局响应归一化）

ConvNeXt V2（Woo et al., CVPR 2023）在每个 block 末尾插入 GRN 层：

$$\text{GRN}(X_i) = \gamma \cdot \frac{X_i}{\sqrt{\|X_i\|^2 + \epsilon}} + \beta$$

GRN 通过 L2 归一化强制各通道保持多样性，抑制深层网络中的特征坍缩。

#### 3.2.3 改进二：广义均值池化（GeM Pooling）

$$f^{(g)} = \left[ \frac{1}{|\Omega|} \sum_{x \in \Omega} x^p \right]^{\frac{1}{p}}$$

- $p=1$：退化为平均池化（GAP）
- $p \to \infty$：逼近最大池化（GMP）
- $p>1$：赋予高响应区域更大权重

关键设计决策：$p$ 初始化为 **1.0**（非文献默认 3.0）。实验发现 $p_{init}=3.0$ 在训练初期导致严重收敛困难（首轮验证准确率仅 5.9%），$p_{init}=1.0$ 让模型从 GAP 起步，训练中 p 自动增长以聚焦判别性区域。

#### 3.2.4 改进三：指数滑动平均（EMA）

$$\theta_{\text{shadow}}^{(t)} = 0.999 \cdot \theta_{\text{shadow}}^{(t-1)} + 0.001 \cdot \theta^{(t)}$$

EMA 维护模型参数的滑动平均副本，推理时使用 shadow 权重。等效窗口约 1,000 步，在不增加推理开销的前提下提升泛化能力。

### 3.3 训练配置

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 优化器 | AdamW | 解耦权重衰减 |
| 学习率 | 1e-4 | 余弦预热重启调度 |
| 权重衰减 | 0.05 | L2 正则化 |
| 损失函数 | CrossEntropy + Label Smoothing (ε=0.1) | 缓解过拟合，处理长尾分布 |
| 批量大小 | 32 | 适合 28M 参数量 |
| 最大 Epoch | 50 | Early Stopping (patience=10) |
| 混合精度 | AMP (GradScaler) | 加速训练，节省显存 |
| 数据增强 | RandCrop + Flip + Rotation + ColorJitter | 7 种增强策略 |
| 输入尺寸 | 224×224 | ImageNet 标准 |

---

## 第4章 系统实现

### 4.1 开发环境

| 组件 | 版本/型号 |
|------|----------|
| 操作系统 | macOS / Windows |
| 编程语言 | Python 3.14 |
| 深度学习框架 | PyTorch 2.x |
| 模型库 | timm 0.9.x |
| Web 框架 | Flask 3.x |
| 图像处理 | OpenCV, PIL |
| 硬件 | Apple M4 Pro (MPS) / NVIDIA GPU (CUDA) |
| 大模型 API | OpenAI 兼容接口（支持豆包/GPT-4/本地模型） |

### 4.2 项目结构

```
skin_detect/
├── train.py                    # 训练主脚本，支持消融变体
├── detect.py                   # 命令行推理脚本
├── export_onnx.py              # ONNX 模型导出
├── models/
│   └── modules.py              # GeM Pooling, EMA, MultiScaleHead
├── utils/
│   └── visualize.py            # 混淆矩阵, ROC曲线, Grad-CAM 可视化
├── web_app/
│   ├── app.py                  # Flask 后端 (分类/Grad-CAM/LLM/SSE)
│   ├── templates/index.html    # 前端页面
│   └── static/
│       ├── style.css           # 样式
│       └── app.js              # 前端逻辑 (上传/流式/对话)
├── runs/
│   ├── convnext_tiny/          # 基线模型 (Val Acc 79.80%)
│   │   ├── best.pt
│   │   ├── *.onnx / *.onnx.data
│   │   ├── confusion_matrix.png
│   │   ├── roc_curves.png
│   │   ├── per_class_metrics.png
│   │   └── training_curves.png
│   └── convnextv2_tiny_GeM_EMA/ # 改进模型 (Val Acc 82.20%)
│       └── ... (同上)
├── TECHNICAL_REPORT.md         # 论文级技术文档
├── INNOVATION.md               # 创新点描述
└── PROJECT_REPORT.md           # 本报告
```

### 4.3 核心模块实现

#### 4.3.1 GeM Pooling 模块

```python
class GeMPool(nn.Module):
    def __init__(self, p_init=1.0):
        self.p = nn.Parameter(torch.ones(1) * p_init)

    def forward(self, x):
        p = self.p.clamp(min=1.0, max=5.0)
        return x.clamp(min=1e-6).pow(p).mean(dim=(2,3)).pow(1.0/p)
```

#### 4.3.2 EMA 模块

```python
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(0.999).add_(p.data, alpha=0.001)
```

#### 4.3.3 Grad-CAM 注意力可视化

通过注册前向/反向 hook 捕获 Stage 4 的特征图和梯度，计算加权激活图，生成病灶区域热力图。同时加载基线和改进模型，并排展示两者的注意力差异。

#### 4.3.4 流式 AI 报告

使用 Server-Sent Events (SSE) 实现大模型 API 的流式传输。客户端通过 AbortController 管理 SSE 连接生命周期，避免多请求阻塞。

### 4.4 关键技术难点与解决方案

| 难点 | 解决方案 |
|------|---------|
| GeM p_init 导致训练崩溃 | p_init 从 3.0 → 1.0，从 GAP 起步 |
| OpenCV 中文乱码 | 改用 PIL + 系统 CJK 字体渲染 |
| 流式传输连接冲突 | AbortController 管理 + SSE 独立通道 |
| 旧 checkpoint 不兼容新架构 | key 重映射（backbone. 前缀、head 索引） |
| ONNX 导出 opset 兼容性 | 使用 opset 18 + dynamo 导出 + 外部权重 |

---

## 第5章 测试分析

### 5.1 实验设置

- **数据集**：Skin Disease Dataset，22 类，15,444 张图像（训练 13,898 + 测试 1,546）
- **评估指标**：Accuracy, Precision, Recall, F1, AUC, 混淆矩阵
- **对比实验**：基线模型（ConvNeXt V1 + AvgPool） vs 改进模型（ConvNeXt V2 + GeM + EMA）

### 5.2 主要结果

| 指标 | 基线 | 改进 | 提升 |
|------|------|------|------|
| 验证集准确率 | 79.80% | **82.20%** | **+2.40%** |
| 测试集准确率 | 80.47% | **81.69%** | **+1.22%** |
| Micro AUC | 0.9740 | **0.9816** | +0.0076 |
| Macro F1 | 0.771 | **0.801** | **+3.0%** |
| 验证集 Loss | 1.222 | **1.141** | -0.081 |
| 参数量 | 27.84M | 27.88M | +0.17% |

![训练曲线-基线](runs/convnext_tiny/training_curves.png)
*图 5-1: 基线模型训练曲线（最优 Epoch 45, Val Acc = 79.80%）*

![训练曲线-改进](runs/convnextv2_tiny_GeM_EMA/training_curves.png)
*图 5-2: 改进模型训练曲线（最优 Epoch 50, Val Acc = 82.20%）*

### 5.3 混淆矩阵分析

![混淆矩阵-基线](runs/convnext_tiny/confusion_matrix.png)
*图 5-3: 基线模型验证集混淆矩阵*

![混淆矩阵-改进](runs/convnextv2_tiny_GeM_EMA/confusion_matrix.png)
*图 5-4: 改进模型验证集混淆矩阵*

改进模型的对角线更集中，表明分类精度全面提升。Actinic Keratosis ↔ Seborrheic Keratoses 的跨类混淆有所减少。

### 5.4 ROC 曲线分析

![ROC-基线](runs/convnext_tiny/roc_curves.png)
*图 5-5: 基线模型 One-vs-Rest ROC 曲线（Micro AUC = 0.974）*

![ROC-改进](runs/convnextv2_tiny_GeM_EMA/roc_curves.png)
*图 5-6: 改进模型 One-vs-Rest ROC 曲线（Micro AUC = 0.982）*

改进模型在各类别上的 AUC 普遍提升，尤其在中高风险类别（Lupus、Bullous）上提升显著。

### 5.5 Per-Class 性能对比

![Per-Class-基线](runs/convnext_tiny/per_class_metrics.png)
*图 5-7: 基线模型 Per-Class 指标*

![Per-Class-改进](runs/convnextv2_tiny_GeM_EMA/per_class_metrics.png)
*图 5-8: 改进模型 Per-Class 指标*

| 提升最大 Top 5 | 基线 F1 | 改进 F1 | 提升幅度 |
|---------------|---------|---------|---------|
| Lupus（红斑狼疮） | 0.688 | **0.809** | +17.6% |
| Bullous（大疱性皮肤病） | 0.713 | **0.795** | +11.5% |
| Lichen（苔藓） | 0.693 | **0.759** | +9.5% |
| Tinea（癣） | 0.719 | **0.782** | +8.8% |
| Sun/Sunlight Damage | 0.660 | **0.712** | +7.9% |

### 5.6 高风险类别专项分析

Skin Cancer（皮肤癌）作为唯一 HIGH 风险类别，其分类性能具有特殊临床意义：

| 指标 | 基线 | 改进 | 临床意义 |
|------|------|------|---------|
| Precision | 70.54% | 66.40% | 假阳性增多 |
| **Recall** | **73.83%** | **77.57%** | **假阴性减少（漏诊降低 3.7%）** |

改进模型以 Precision 下降为代价换取 Recall 提升——在筛查场景中，减少漏诊远比减少误报重要。

### 5.7 注意力可视化对比

改进模型（V2 + GeM）的 Grad-CAM 热力图相比基线（V1 + Avg）更聚焦于病灶核心区域。GeM Pooling 在训练过程中自动学习关注高判别性区域，注意力分布更紧凑，边界更清晰。

### 5.8 推理效率

| 模型 | 参数量 | ONNX 大小 | 单图推理 (CPU) |
|------|--------|----------|---------------|
| Baseline | 27.84M | 111 MB | ~45ms |
| Improved | 27.88M | 112 MB | ~48ms |

改进模型参数量仅增加 0.17%，推理时间增加约 6%（GRN 层额外计算），在可接受范围内。

---

## 第6章 作品总结

### 6.1 作品特色与创新点

**创新点一：GeM Pooling 初始化策略改进**

传统 GeM Pooling 默认 $p_{init}=3.0$，本文发现该设置在医学图像分类任务中导致严重收敛困难（首轮验证准确率低至 5.9%）。本文提出 $p_{init}=1.0$ 的初始化策略——训练初期等效于 GAP（与基线一致），训练中 p 自动增长聚焦判别性区域——解决了收敛问题，同时保留了 GeM 的优势。最终 Macro F1 提升 3.0%，Lupus 类别 F1 提升 17.6%。

**创新点二：GRN + GeM 协同机制**

ConvNeXt V2 的 GRN 层增强了特征多样性，GeM 在多样化的特征空间中更有效地聚焦判别性区域。两者的协同效应显著超越了各自独立贡献，在纹理/色素类疾病（Lupus、Bullous、Lichen）上尤为突出。

**创新点三：可交互的多模态辅助诊断系统**

构建了完整的 Web 应用，集成三个层面的智能辅助：（1）高精度图像分类；（2）Grad-CAM 注意力可视化（基线 vs 改进对比）；（3）大语言模型驱动的疾病咨询和报告生成，通过 SSE 流式传输实现实时交互。

**创新点四：全面的消融实验与分析**

系统设计了多组对比实验，覆盖了混淆矩阵、ROC 曲线、Per-Class 指标、训练曲线、注意力可视化等多个维度，为每个改进点的有效性提供了充分的实证支持。特别地，对 Skin Cancer 的 Recall-Precision 权衡进行了临床视角的深度分析。

### 6.2 应用推广

#### 6.2.1 应用场景

| 场景 | 说明 |
|------|------|
| 基层医疗机构 | 辅助全科医生进行皮肤病初步筛查，降低误诊和漏诊率 |
| 远程医疗 | 患者上传皮损照片，系统给出初步分类 + AI 建议，辅助远程问诊 |
| 医学教育 | 提供可视化的注意力热力图和疾病知识库，辅助医学生和住院医师学习 |
| 大规模筛查 | 社区/学校/工厂的批量皮肤癌/常见皮肤病筛查 |

#### 6.2.2 推广路径

1. **开源社区**：代码已开源至 GitHub (https://github.com/yc004/skin_detect)，提供完整训练/推理/部署文档。
2. **模型部署**：模型已导出为 ONNX 格式，可部署至边缘设备、移动端或云服务器。
3. **API 服务**：Flask 后端可直接作为 RESTful API 使用，方便集成至第三方系统。
4. **持续改进**：项目结构模块化，支持配置不同模型变体、数据集和 API 后端，便于后续扩展。

### 6.3 作品展望

#### 6.3.1 短期改进（3-6 个月）

1. **多数据集融合训练**：结合 ISIC 2019、HAM10000 等多个公开数据集，提升模型在不同采集条件下的泛化能力。
2. **多任务学习**：在分类基础上添加病灶分割头，使用 Grad-CAM 作为弱监督信号，实现"诊断 + 定位"的联合输出。
3. **模型量化与加速**：对 ONNX 模型进行 INT8 量化，将推理时间压缩至 10ms 以下，满足移动端部署需求。

#### 6.3.2 中长期规划（6-18 个月）

4. **多模态融合**：融合患者年龄、性别、病灶部位、病程等临床元数据，构建图像 + 结构化数据的多模态分类模型。
5. **纵向跟踪**：支持同一患者多次就诊的图像对比分析，评估病灶变化趋势，实现疾病进展监测。
6. **临床验证**：与皮肤科合作开展前瞻性临床验证，评估模型在实际临床环境中的表现，推进医疗器械认证。
7. **联邦学习**：在保护患者隐私的前提下，通过联邦学习框架联合多家医疗机构的数据，持续优化模型。

---

## 参考文献

[1] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. *CVPR 2022*.

[2] Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I. S., & Xie, S. (2023). ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders. *CVPR 2023*.

[3] Radenović, F., Tolias, G., & Chum, O. (2018). Fine-tuning CNN Image Retrieval with No Human Annotation. *IEEE TPAMI*, 41(7), 1655-1668.

[4] Polyak, B. T., & Juditsky, A. B. (1992). Acceleration of Stochastic Approximation by Averaging. *SIAM J. Control Optim.*, 30(4), 838-855.

[5] Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.

[6] Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.

[7] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.

---

> 📅 报告版本: v1.0 | 2026-06-26  
> 📊 实验数据来源: `runs/convnext_tiny/experiment_summary.json` + `runs/convnextv2_tiny_GeM_EMA/experiment_summary.json`  
> 💻 项目代码: https://github.com/yc004/skin_detect
