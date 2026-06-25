# 🩺 皮肤疾病智能分类系统

**基于 ConvNeXt 的 22 类皮肤疾病辅助诊断**

---

## 项目简介

本项目构建了一个面向皮肤疾病自动分类的深度学习系统。采用 ConvNeXt 作为骨干网络，在 22 类皮肤疾病数据集上进行训练，支持多种架构变体的对比实验（消融研究），并提供完整的训练、推理、可视化与交互式演示功能。

### 核心特性

- 🧠 **多架构支持**: ConvNeXt V1 / V2，GeM Pooling，多尺度特征融合，EMA 权重平滑
- 🔬 **消融实验**: 模块化设计，任意组合组件进行对照实验
- 📊 **完整报告**: 自动生成混淆矩阵、ROC 曲线、per-class 指标、训练曲线
- 🎯 **风险分级**: HIGH / MEDIUM / LOW 三级临床风险提示
- 🔍 **可解释性**: Grad-CAM 热力图可视化模型关注区域
- 🖥️ **交互式 Demo**: 基于 Gradio 的 Web 界面

---

## 数据集

| 属性 | 值 |
|------|-----|
| 类别数 | 22 |
| 训练集 | 13,898 张 |
| 测试集 | 1,546 张 |
| 总计 | 15,444 张 |
| 数据来源 | Skin Disease Dataset (公开数据集) |

<details>
<summary><b>22 类完整列表（点击展开）</b></summary>

| 类别 | 风险等级 | 训练 | 测试 |
|------|---------|------|------|
| Skin Cancer | 🔴 HIGH | 693 | 77 |
| Actinic Keratosis | 🟡 MEDIUM | 748 | 83 |
| Lupus | 🟡 MEDIUM | 311 | 34 |
| Vasculitis | 🟡 MEDIUM | 461 | 52 |
| Bullous | 🟡 MEDIUM | 504 | 55 |
| Acne | 🟢 LOW | 593 | 65 |
| Benign Tumors | 🟢 LOW | 1,093 | 121 |
| Candidiasis | 🟢 LOW | 248 | 27 |
| Drug Eruption | 🟢 LOW | 547 | 61 |
| Eczema | 🟢 LOW | 1,010 | 112 |
| Infestations/Bites | 🟢 LOW | 524 | 60 |
| Lichen | 🟢 LOW | 553 | 61 |
| Moles | 🟢 LOW | 361 | 40 |
| Psoriasis | 🟢 LOW | 820 | 88 |
| Rosacea | 🟢 LOW | 254 | 28 |
| Seborrheic Keratoses | 🟢 LOW | 455 | 51 |
| Sun/Sunlight Damage | 🟢 LOW | 312 | 34 |
| Tinea | 🟢 LOW | 923 | 102 |
| Unknown/Normal | 🟢 LOW | 1,651 | 189 |
| Vascular Tumors | 🟢 LOW | 543 | 60 |
| Vitiligo | 🟢 LOW | 714 | 82 |
| Warts | 🟢 LOW | 580 | 64 |

</details>

---

## 项目结构

```
skin_detect/
├── dataset/                         # 数据集（不纳入版本控制）
│   └── SkinDisease/
│       └── SkinDisease/
│           ├── Train/               # 训练集（按类分文件夹）
│           └── Test/                # 测试集
├── models/
│   ├── __init__.py
│   └── modules.py                   # GeM Pooling / 多尺度融合 / EMA
├── utils/
│   ├── __init__.py
│   ├── visualize.py                 # 可视化：混淆矩阵、ROC、Grad-CAM
│   └── pseudo_label.py              # 伪标签生成（用于无标注数据）
├── runs/                            # 训练输出（每个实验一个子目录）
│   └── <experiment_name>/
│       ├── best.pt                  # 最佳模型权重
│       ├── experiment_summary.json  # 完整实验指标汇总
│       ├── classification_report.txt
│       ├── training_history.csv     # 每 epoch 训练日志
│       ├── confusion_matrix.png
│       ├── confusion_matrix_test.png
│       ├── roc_curves.png
│       ├── roc_curves_test.png
│       ├── per_class_metrics.png
│       └── training_curves.png
├── results/                         # 推理输出
├── train.py                         # 训练主脚本
├── detect.py                        # 推理脚本
├── demo.py                          # Gradio Web Demo
├── requirements.txt
├── README.md
└── TECHNICAL_REPORT.md              # 论文级技术文档
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练（基线模型）
python train.py --model convnext_tiny

# 3. 训练（推荐配置：V2 + GeM + EMA）
python train.py --model convnextv2_tiny --pooling gem --ema

# 4. 推理
python detect.py --image 皮肤图片.jpg --cam

# 5. 启动 Web Demo
python demo.py
```

---

## 架构变体与消融实验

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `--model` | `convnext_tiny` / `convnextv2_tiny` / `convnext_small` | 骨干网络 |
| `--pooling` | `avg` / `gem` | 池化方式 |
| `--multi-scale` | flag | 多尺度特征融合 |
| `--ema` | flag | EMA 权重平滑 |
| `--img-size` | 224 / 384 | 输入分辨率 |

### 典型实验命令

```bash
# 实验 1: 基线
python train.py --model convnext_tiny --pooling avg

# 实验 2: V2 对比
python train.py --model convnextv2_tiny

# 实验 3: V2 + GeM
python train.py --model convnextv2_tiny --pooling gem

# 实验 4: V2 + GeM + EMA（推荐）
python train.py --model convnextv2_tiny --pooling gem --ema

# 实验 5: 多尺度融合
python train.py --model convnext_tiny --multi-scale

# 实验 6: 高分辨率
python train.py --model convnext_tiny --img-size 384

# 快速验证（5 epoch）
python train.py --model convnextv2_tiny --quick
```

每个实验自动输出到 `runs/<变体名>/`，互不覆盖。

---

## 训练配置

| 参数 | 值 |
|------|-----|
| 架构 | ConvNeXt-Tiny (28M) |
| 预训练 | ImageNet-22k |
| 输入尺寸 | 224×224（可选 384） |
| 优化器 | AdamW, lr=1e-4 |
| 学习率调度 | Cosine Warm Restarts |
| 损失函数 | CrossEntropy + Label Smoothing (0.1) |
| Batch Size | 32 |
| Epochs | 50（Early Stopping, patience=10） |
| 数据增强 | RandomResizedCrop, Flip, Rotation, ColorJitter |
| 混合精度 | AMP (GradScaler) |
| 设备 | MPS / CUDA / CPU |

---

## 输出指标说明

训练完成后，`runs/<实验名>/` 目录下生成：

| 文件 | 用途 |
|------|------|
| `experiment_summary.json` | 所有指标、配置、per-class AUC 汇总 |
| `classification_report.txt` | sklearn 格式的 P/R/F1 报告 |
| `training_history.csv` | 每 epoch 的 train/val loss 与 accuracy |
| `confusion_matrix.png` | 验证集混淆矩阵（含数量+百分比） |
| `confusion_matrix_test.png` | 测试集混淆矩阵 |
| `roc_curves.png` | One-vs-Rest ROC 曲线 + AUC 值 |
| `per_class_metrics.png` | 每类 Precision/Recall/F1 柱状图 |
| `training_curves.png` | Loss 和 Accuracy 训练曲线 |

---

## 论文撰写

详细的技术方案、数据集分析、消融实验设计和结果讨论请参阅：

👉 **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**

---

## 依赖

- PyTorch ≥ 2.0
- timm ≥ 0.9
- torchvision
- opencv-python
- scikit-learn
- matplotlib, seaborn
- Gradio (Demo)
- tqdm

---

## ⚠️ 免责声明

**本系统为学术研究原型，非医疗器械。** 所有预测结果仅供研究参考，不可用于临床诊断。任何医疗决策应由合格的皮肤科医生做出。
