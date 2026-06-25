#!/usr/bin/env python3
"""
Skin Disease Classification — Gradio Interactive Demo.

Features:
  - 22-class skin disease classification with ConvNeXt
  - Chinese + English bilingual display
  - LLM API for disease description and treatment advice
  - Grad-CAM heatmap, batch analysis, risk triage

Usage:
    python demo.py                              # reads config.json
    python demo.py --api-key sk-xxx             # override API key
    python demo.py --api-key sk-xxx --api-base https://api.openai.com/v1

Config file (config.json):
    {"llm": {"api_key": "sk-xxx", "api_base": "...", "api_model": "gpt-4o-mini"}}
    CLI arguments override config file values.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import gradio as gr

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm

from models.modules import ConvNeXtWithFeatures
from utils.visualize import draw_classification_result, draw_gradcam


# ============================================================
# Built-in Chinese Fallback
# ============================================================

CLASS_NAMES_ZH_FALLBACK = {
    "Acne": "痤疮",
    "Actinic Keratosis": "光化性角化病",
    "Benign Tumors": "良性肿瘤",
    "Bullous": "大疱性皮肤病",
    "Candidiasis": "念珠菌病",
    "Drug Eruption": "药疹",
    "Eczema": "湿疹",
    "Infestations/Bites": "寄生虫/虫咬",
    "Lichen": "苔藓",
    "Lupus": "红斑狼疮",
    "Moles": "痣",
    "Psoriasis": "银屑病",
    "Rosacea": "玫瑰痤疮",
    "Seborrheic Keratoses": "脂溢性角化病",
    "Skin Cancer": "皮肤癌",
    "Sun/Sunlight Damage": "日光性损伤",
    "Tinea": "癣",
    "Unknown/Normal": "未知/正常",
    "Vascular Tumors": "血管性肿瘤",
    "Vasculitis": "血管炎",
    "Vitiligo": "白癜风",
    "Warts": "疣",
}

# ============================================================
# Disease Knowledge Base
# ============================================================

DISEASE_KB = {
    "Acne": {
        "overview": "痤疮是一种常见的毛囊皮脂腺慢性炎症性皮肤病，好发于面部、胸背部。",
        "symptoms": "粉刺（黑头/白头）、炎性丘疹、脓疱、结节、囊肿，可遗留色素沉着或瘢痕。",
        "treatment": "轻度：外用维A酸/过氧化苯甲酰；中度：外用+口服抗生素；重度：口服异维A酸、光动力疗法。",
        "precautions": "避免挤压皮损；使用非致痘性护肤品；减少高糖/高脂饮食；规律作息。",
    },
    "Actinic Keratosis": {
        "overview": "光化性角化病是由长期日晒引起的癌前病变，表现为粗糙鳞屑性斑片，有恶变为鳞状细胞癌的风险。",
        "symptoms": "皮肤粗糙、干燥、鳞屑性红斑或棕黄色斑片，触之如砂纸，好发于面部、手背等曝光部位。",
        "treatment": "冷冻治疗、外用5-氟尿嘧啶/咪喹莫特、光动力疗法、激光。多发或顽固皮损需皮肤科医生评估。",
        "precautions": "严格防晒（SPF50+）；定期皮肤科随访（每6-12个月）；发现新皮损或原有皮损变化及时就诊。",
    },
    "Benign Tumors": {
        "overview": "良性皮肤肿瘤包括脂肪瘤、纤维瘤、表皮囊肿等多种类型，生长缓慢，不发生转移。",
        "symptoms": "皮下结节或肿块，质地软至韧，边界清楚，通常无痛，大小较稳定。",
        "treatment": "多数无需治疗，定期观察即可；若影响美观或功能可手术切除；需皮肤科医生明确诊断排除恶性。",
        "precautions": "观察有无快速增大、颜色改变、破溃等异常变化；发现上述变化及时就医。",
    },
    "Bullous": {
        "overview": "大疱性皮肤病是一组以皮肤出现水疱和大疱为特征的自身免疫性或遗传性疾病。",
        "symptoms": "皮肤和/或黏膜出现大小不等的水疱、大疱，可伴有瘙痒或灼痛，Nikolsky征可能阳性。",
        "treatment": "需皮肤科专科诊治；常用糖皮质激素、免疫抑制剂、生物制剂（利妥昔单抗等）；需住院治疗重症患者。",
        "precautions": "保持皮肤清洁干燥；避免摩擦和外伤；定期专科复诊；不可自行停药。",
    },
    "Candidiasis": {
        "overview": "念珠菌病是由白色念珠菌等引起的真菌感染，可累及皮肤、黏膜和内脏器官。",
        "symptoms": "皮肤：红斑、丘疹、脓疱，边缘有鳞屑；黏膜：口腔白斑、阴道豆腐渣样分泌物、瘙痒。",
        "treatment": "外用抗真菌药（克霉唑、咪康唑等）；口服氟康唑/伊曲康唑；需寻找并纠正易感因素（糖尿病、免疫力低下等）。",
        "precautions": "保持皮肤干燥通风；控制血糖；避免滥用抗生素；注意个人卫生。",
    },
    "Drug Eruption": {
        "overview": "药疹是由药物引起的皮肤不良反应，表现多样，从轻度皮疹到危及生命的重症药疹。",
        "symptoms": "皮疹形态多样（麻疹样、荨麻疹样、固定型等），可伴瘙痒、发热、黏膜受累。",
        "treatment": "立即停用可疑药物；抗组胺药止痒；重症需住院使用糖皮质激素/免疫球蛋白；需皮肤科急诊评估。",
        "precautions": "记录过敏药物并终身避免使用；就诊时主动告知医生药物过敏史。",
    },
    "Eczema": {
        "overview": "湿疹（特应性皮炎）是一种慢性复发性炎症性皮肤病，与遗传过敏体质相关。",
        "symptoms": "皮肤干燥、红斑、丘疹、水疱、渗出、结痂、苔藓化；剧烈瘙痒；呈对称性分布。",
        "treatment": "保湿剂（基础治疗）；外用糖皮质激素/钙调磷酸酶抑制剂；口服抗组胺药；重症可用生物制剂（度普利尤单抗）或JAK抑制剂。",
        "precautions": "坚持每日保湿；避免搔抓；避免已知过敏原和刺激物；温水洗浴。",
    },
    "Infestations/Bites": {
        "overview": "寄生虫感染或虫咬引起的皮肤病，包括疥疮、虱病、螨虫皮炎等。",
        "symptoms": "瘙痒性丘疹、水疱、隧道（疥疮）、抓痕；皮损多见于指缝、腕部、腰部等。",
        "treatment": "外用杀虫药（硫磺软膏、氯菊酯等）；口服伊维菌素；家庭成员同时治疗；衣物被褥消毒。",
        "precautions": "注意个人和环境卫生；避免共用毛巾衣物；宠物定期驱虫。",
    },
    "Lichen": {
        "overview": "苔藓样皮肤病包括扁平苔藓、硬化萎缩性苔藓等，病因可能与免疫异常相关。",
        "symptoms": "紫红色多角形扁平丘疹（扁平苔藓）；白色萎缩性斑片（硬化萎缩性苔藓）；可累及皮肤和黏膜。",
        "treatment": "外用强效糖皮质激素；口服维A酸/免疫抑制剂；光疗；需皮肤科专科长期管理。",
        "precautions": "避免搔抓（同形反应）；定期皮肤科随访；口腔和外阴受累需多学科协作。",
    },
    "Lupus": {
        "overview": "红斑狼疮是一种自身免疫性疾病，可累及皮肤（皮肤型）或多个器官系统（系统性）。",
        "symptoms": "面部蝶形红斑、盘状红斑、光敏感、口腔溃疡、关节痛、发热、乏力。",
        "treatment": "需风湿免疫科/皮肤科综合管理；羟氯喹（基础用药）；糖皮质激素；免疫抑制剂；生物制剂。",
        "precautions": "严格防晒；定期复查血常规/尿常规/补体/自身抗体；避免劳累和感染；不可自行停药。",
    },
    "Moles": {
        "overview": "痣（色素痣）是黑色素细胞的良性增生，绝大多数为良性，极少数可发生恶变。",
        "symptoms": "棕色至黑色的斑点或丘疹，可平坦或隆起，边界清楚，形态规则，大小稳定。",
        "treatment": "绝大多数无需治疗；如需切除（美容原因/可疑恶变），应手术完整切除并送病理检查。",
        "precautions": "按ABCDE原则自查（不对称、边缘不规则、颜色不均、直径>6mm、进展变化）；可疑痣及时皮肤科就诊；避免暴晒。",
    },
    "Psoriasis": {
        "overview": "银屑病（牛皮癣）是一种免疫介导的慢性复发性皮肤病，可伴有银屑病性关节炎。",
        "symptoms": "红色斑块上覆银白色鳞屑，Auspitz征阳性；可累及头皮、指甲/趾甲、关节。",
        "treatment": "轻中度：外用糖皮质激素/维生素D3衍生物、光疗（NB-UVB）；中重度：口服甲氨蝶呤/维A酸、生物制剂、小分子靶向药。",
        "precautions": "避免诱因（感染、外伤、精神压力、酒精）；坚持皮肤保湿；定期专科复诊。",
    },
    "Rosacea": {
        "overview": "玫瑰痤疮是一种慢性面部炎症性疾病，好发于30-50岁女性，表现为面部潮红、毛细血管扩张。",
        "symptoms": "面中部阵发性潮红、持续性红斑、毛细血管扩张、丘疹脓疱、鼻赘（晚期）。",
        "treatment": "避免诱因（日晒、辛辣食物、酒精、极端温度）；外用甲硝唑/伊维菌素/溴莫尼定；口服多西环素；激光/强脉冲光。",
        "precautions": "严格防晒；使用温和护肤品；避免使用含酒精/香精的化妆品；记录并避免个人诱因。",
    },
    "Seborrheic Keratoses": {
        "overview": "脂溢性角化病（老年斑）是常见的良性表皮增生，多见于中老年人，无恶变风险。",
        "symptoms": "棕褐色至黑色的疣状斑块，表面呈油脂状或颗粒状，\"贴上去的\"外观，大小数毫米至数厘米。",
        "treatment": "良性无需治疗；影响美观可冷冻、激光或刮除去除；需由皮肤科医生确诊排除恶性。",
        "precautions": "观察有无快速增大、颜色不均、破溃（如有则需排除恶变）；定期皮肤检查。",
    },
    "Skin Cancer": {
        "overview": "皮肤癌包括基底细胞癌、鳞状细胞癌和黑色素瘤等，是全球最常见的恶性肿瘤之一。早期发现治愈率很高。",
        "symptoms": "不愈合的溃疡、进行性增大的结节、色素性皮损ABCDE改变（不对称/边缘不规则/颜色不均/直径>6mm/进展变化）。",
        "treatment": "需尽快皮肤科就诊！手术完整切除（Mohs手术为首选）；根据病理类型和分期决定后续治疗（放疗、靶向治疗、免疫治疗）。",
        "precautions": "此为紧急情况！立即预约皮肤科医生；严格防晒；终身定期皮肤检查（每3-6个月）；检查全身皮肤和淋巴结。",
    },
    "Sun/Sunlight Damage": {
        "overview": "日光性损伤是长期紫外线暴露导致的皮肤退行性改变，包括日光性角化、色素沉着、皱纹等。",
        "symptoms": "皮肤粗糙、色素不均（日光性黑子、雀斑）、皱纹、毛细血管扩张、皮肤松弛。",
        "treatment": "严格防晒；外用维A酸/果酸改善肤质；激光/IPL治疗色素和血管问题；定期皮肤科筛查排除癌变。",
        "precautions": "每日使用广谱防晒霜（SPF50+PA++++）；避免10:00-16:00户外暴晒；穿戴防护衣物。",
    },
    "Tinea": {
        "overview": "癣是由皮肤癣菌（真菌）引起的浅表感染，根据部位分为头癣、体癣、股癣、手足癣、甲癣等。",
        "symptoms": "环状红斑，边缘隆起有鳞屑，中央趋向自愈；瘙痒；甲癣表现为甲板增厚、变色、碎裂。",
        "treatment": "局限性：外用抗真菌药（特比萘芬、克霉唑等）；泛发性/甲癣：口服特比萘芬/伊曲康唑（需肝功能监测）。",
        "precautions": "保持皮肤干燥；不共用毛巾拖鞋；宠物查治癣病；足疗程治疗（甲癣需数月）。",
    },
    "Unknown/Normal": {
        "overview": "正常皮肤或临床表现不典型、无法明确分类的皮损。",
        "symptoms": "无明显异常皮损，或皮损特征不足以明确诊断。",
        "treatment": "无需特殊治疗。如有疑虑，建议皮肤科医生面诊评估。",
        "precautions": "保持良好的皮肤护理习惯；如有新发皮损或原有皮损变化，及时就医。",
    },
    "Vascular Tumors": {
        "overview": "血管性肿瘤包括血管瘤、血管肉瘤等，可为先天性或获得性，多数为良性。",
        "symptoms": "红色至紫红色的斑块或结节，可平坦或隆起；婴幼儿血管瘤有增殖-消退自然过程。",
        "treatment": "多数婴幼儿血管瘤可观察；需治疗者可用口服普萘洛尔/外用噻吗洛尔；成人需皮肤科评估排除恶性血管肿瘤。",
        "precautions": "观察有无快速增大、破溃、出血；婴幼儿血管瘤位于眼周/气道需紧急评估。",
    },
    "Vasculitis": {
        "overview": "血管炎是血管壁炎症导致的一系列疾病，可局限于皮肤或累及多系统。",
        "symptoms": "可触性紫癜、结节、溃疡、网状青斑；可伴发热、关节痛、腹痛、血尿等系统症状。",
        "treatment": "需风湿免疫科/皮肤科综合诊治；根据严重程度使用糖皮质激素/免疫抑制剂/生物制剂。",
        "precautions": "定期复查尿液和肾功能；不可自行停药；发现新症状及时告知医生。",
    },
    "Vitiligo": {
        "overview": "白癜风是一种色素脱失性皮肤病，由黑色素细胞破坏引起，全球患病率约1%。",
        "symptoms": "边界清楚的乳白色或瓷白色斑片，大小形态不一；可发生于任何部位，常见于面部、手背、腋窝等。",
        "treatment": "外用糖皮质激素/他克莫司；NB-UVB光疗（一线方案）；稳定期可考虑表皮移植；JAK抑制剂（芦可替尼乳膏）。",
        "precautions": "防晒（白斑区易晒伤）；避免外伤（Koebner现象）；心理支持（影响外观可导致心理压力）。",
    },
    "Warts": {
        "overview": "疣是由人乳头瘤病毒（HPV）感染引起的良性皮肤增生，常见于手、足部位。",
        "symptoms": "表面粗糙的角化性丘疹或斑块，可见小黑点（血栓性毛细血管）；可单发或多发；可有压痛（跖疣）。",
        "treatment": "外用含水杨酸制剂；冷冻治疗；激光；光动力；部分可自行消退（尤其儿童）。",
        "precautions": "避免搔抓（自身接种传播）；公共场所穿拖鞋；洗手后保持干燥；免疫力低下者需积极治疗。",
    },
}

# ============================================================
# LLM Client (OpenAI-compatible)
# ============================================================

class LLMClient:
    """OpenAI-compatible API client with streaming support."""

    def __init__(self, api_key: str, api_base: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self._available = None

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self.api_base}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            urllib.request.urlopen(req, timeout=10)
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def _build_prompt(self, disease_en: str, disease_zh: str,
                      confidence: float, risk: str) -> str:
        kb_entry = DISEASE_KB.get(disease_en, {})
        kb_text = ""
        if kb_entry:
            kb_text = f"""已知疾病信息:
- 概述: {kb_entry['overview']}
- 症状: {kb_entry['symptoms']}
- 治疗: {kb_entry['treatment']}
- 注意事项: {kb_entry['precautions']}"""

        return f"""你是皮肤科AI助手。根据分类结果生成简洁医学报告。

分类: {disease_en}（{disease_zh}） | 置信度: {confidence:.1%} | 风险: {risk}

{kb_text}

用中文输出，每段不超过3行:

【疾病概述】
（1-2句）

【可能症状】
（2-3个要点）

【建议措施】
（2-3条建议）

【就医指引】
（科室+具体建议）

末尾加: ⚠️ 本报告AI生成仅供参考，不能替代专业医疗诊断。"""

    def generate_report(self, disease_en: str, disease_zh: str,
                        confidence: float, risk: str) -> str:
        """Non-streaming report generation."""
        if not self._check_available():
            return None
        try:
            import urllib.request
            data = json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是皮肤科AI助手，提供专业准确的医学信息。"},
                    {"role": "user", "content": self._build_prompt(disease_en, disease_zh, confidence, risk)},
                ],
                "temperature": 0.3, "max_tokens": 600, "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.api_base}/chat/completions",
                data=data,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[AI调用失败: {e}]"

    def generate_report_stream(self, disease_en: str, disease_zh: str,
                                confidence: float, risk: str):
        """Streaming report generation — yields token strings."""
        if not self._check_available():
            yield "[AI服务不可用]"
            return
        try:
            import urllib.request
            data = json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是皮肤科AI助手。"},
                    {"role": "user", "content": self._build_prompt(disease_en, disease_zh, confidence, risk)},
                ],
                "temperature": 0.3, "max_tokens": 600, "stream": True,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.api_base}/chat/completions",
                data=data,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=60)
            for line in resp:
                line = line.decode("utf-8").strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        choice = chunk.get("choices", [{}])[0]
                        # Skip final chunk — some APIs send full text in it
                        if choice.get("finish_reason") is not None:
                            continue
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"\n[AI调用失败: {e}]"


# ============================================================
# Model Discovery
# ============================================================

def find_models(runs_dir: str = "runs") -> list:
    runs = Path(runs_dir)
    models = []
    if not runs.exists():
        return models
    for pt_file in sorted(runs.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        summary_file = pt_file.parent / "experiment_summary.json"
        info = {}
        if summary_file.exists():
            try:
                info = json.loads(summary_file.read_text())
            except Exception:
                pass
        display_name = info.get("variant", pt_file.parent.name)
        val_acc = info.get("val_accuracy") or info.get("best_val_acc")
        acc_str = f"Val Acc: {val_acc:.2%}" if val_acc else ""
        models.append({
            "path": str(pt_file), "name": display_name,
            "dir": str(pt_file.parent.name), "val_acc": val_acc,
            "acc_str": acc_str, "num_params": info.get("num_params", "?"),
            "config": info.get("config", {}),
        })
    return models


# ============================================================
# Model Manager
# ============================================================

class ModelManager:
    def __init__(self):
        self.cache = {}
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def load(self, model_path: str):
        if model_path in self.cache:
            return self.cache[model_path]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading: {model_path} ...")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        class_names = checkpoint.get("class_names", [])
        class_names_zh = checkpoint.get("class_names_zh") or CLASS_NAMES_ZH_FALLBACK
        class_risk = checkpoint.get("class_risk", {})

        if not class_names:
            config_path = Path(model_path).parent / "experiment_summary.json"
            if config_path.exists():
                c = json.loads(config_path.read_text())
                class_names = c.get("class_names", [])
                class_names_zh = c.get("class_names_zh") or CLASS_NAMES_ZH_FALLBACK
                class_risk = c.get("class_risk", {})

        num_classes = len(class_names)
        model_cfg = checkpoint.get("config", {})
        model_name = model_cfg.get("model_name", "convnext_tiny")
        pooling = model_cfg.get("pooling", "avg")
        use_multi_scale = model_cfg.get("multi_scale", False)
        img_size = model_cfg.get("img_size", 224)

        MODEL_MAP = {
            "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
            "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
            "convnext_small": "convnext_small.fb_in22k_ft_in1k",
        }
        timm_name = MODEL_MAP.get(model_name, "convnext_tiny.fb_in22k_ft_in1k")

        backbone = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
        model = ConvNeXtWithFeatures(
            backbone=backbone, num_classes=num_classes,
            dropout=0.3, use_multi_scale=use_multi_scale, pooling=pooling,
        )
        if "ema_state" in checkpoint:
            from models.modules import ModelEMA
            ema = ModelEMA(model, decay=0.999)
            ema.load_state_dict(checkpoint["ema_state"])
            ema.apply_shadow(model)

        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("backbone.") for k in state_dict.keys()):
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        if "head.0.weight" in state_dict and "head.2.weight" not in state_dict:
            remap = {}
            for k in list(state_dict.keys()):
                if k.startswith("head.0."):
                    remap[k] = k.replace("head.0.", "head.2.")
                elif k.startswith("head.4."):
                    remap[k] = k
            for old_k, new_k in remap.items():
                state_dict[new_k] = state_dict.pop(old_k)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

        info = {
            "model_name": model_name, "pooling": pooling,
            "multi_scale": use_multi_scale,
            "ema": model_cfg.get("ema", False) or "ema_state" in checkpoint,
            "img_size": img_size, "num_classes": num_classes,
            "val_acc": checkpoint.get("val_acc"), "epoch": checkpoint.get("epoch"),
        }
        self.cache[model_path] = (model, class_names, class_names_zh, class_risk, img_size, transform, info)
        print(f"  ✓ Loaded: {model_name} | pooling={pooling}")
        return self.cache[model_path]


manager = ModelManager()


# ============================================================
# Inference
# ============================================================

def classify(image, model_path, show_cam, top_k, chat_state):
    """Generator: yields (annotated, report_html, cam_image, model_info, chat_state, chatbot)."""
    info_html = _model_info_html(model_path)
    empty = _empty_report()

    if image is None:
        yield None, empty, None, info_html, {}, []
        return

    try:
        model, class_names, class_names_zh, class_risk, img_size, transform, info = manager.load(model_path)
    except Exception as e:
        yield image, f"<p style='color:red'>Failed to load: {e}</p>", None, info_html, {}, []
        return

    # --- Phase 1: Classification ---
    with torch.no_grad():
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = transform(image_rgb).unsqueeze(0).to(manager.device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

        k = min(top_k, len(class_names))
        topk_probs, topk_indices = torch.topk(probs, k)
        topk_probs = topk_probs.cpu().numpy()[0]
        topk_indices = topk_indices.cpu().numpy()[0]

    predictions = []
    for prob, idx in zip(topk_probs, topk_indices):
        name_en = class_names[idx]
        name_zh = class_names_zh.get(name_en, name_en)
        display_name = f"{name_zh}（{name_en}）" if name_zh != name_en else name_en
        risk = class_risk.get(name_en, "LOW")
        predictions.append({"class": name_en, "class_zh": display_name, "confidence": float(prob), "risk": risk})

    annotated = draw_classification_result(image, predictions, class_risk)

    cam_image = None
    if show_cam:
        cam_image = draw_gradcam(model, image, class_names, transform, manager.device)

    # Build context for chat
    top = predictions[0]
    ctx = {
        "predictions": predictions,
        "top_en": top["class"],
        "top_zh": class_names_zh.get(top["class"], top["class"]),
        "top_conf": top["confidence"],
        "top_risk": top["risk"],
        "model_info": info,
    }

    # --- Phase 2: Stream LLM report ---
    if llm_client is not None:
        loading_html = _build_report(predictions, info, "⏳ 正在生成AI建议...")
        yield annotated, loading_html, cam_image, info_html, ctx, []

        accumulated = ""
        for token in llm_client.generate_report_stream(
            top["class"], class_names_zh.get(top["class"], top["class"]),
            top["confidence"], top["risk"]
        ):
            accumulated += token
            stream_html = _build_report(predictions, info, accumulated)
            yield annotated, stream_html, cam_image, info_html, ctx, []
    else:
        report_html = _build_report(predictions, info, None)
        yield annotated, report_html, cam_image, info_html, ctx, []


# ============================================================
# Chat
# ============================================================

def chat(message: str, chat_history: list, ctx: dict):
    """Streaming chat: user asks questions about the classified disease.
    chat_history uses Gradio tuple format: [(user_msg, bot_msg), ...]
    """
    if not ctx or not llm_client:
        chat_history.append((message, "请先上传图像进行分析。"))
        yield chat_history
        return

    top_en = ctx["top_en"]
    top_zh = ctx["top_zh"]
    top_conf = ctx["top_conf"]
    top_risk = ctx["top_risk"]

    kb_entry = DISEASE_KB.get(top_en, {})
    kb_context = ""
    if kb_entry:
        kb_context = f"{kb_entry['overview']} 症状: {kb_entry['symptoms']} 治疗: {kb_entry['treatment']}"

    system_msg = f"""你是皮肤科AI助手。用户上传的图像被分类为: {top_en}（{top_zh}），置信度{top_conf:.1%}，风险等级{top_risk}。
{kb_context}
基于以上结果回答用户问题。用中文，简洁专业。"""

    # Build messages: convert tuple history to API format
    messages = [{"role": "system", "content": system_msg}]
    for user_msg, bot_msg in chat_history[-3:]:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    try:
        import urllib.request
        data = json.dumps({
            "model": llm_client.model,
            "messages": messages,
            "temperature": 0.5, "max_tokens": 500, "stream": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{llm_client.api_base}/chat/completions",
            data=data,
            headers={"Authorization": f"Bearer {llm_client.api_key}", "Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=60)

        chat_history.append((message, ""))
        yield chat_history

        for line in resp:
            line = line.decode("utf-8").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    choice = chunk.get("choices", [{}])[0]
                    # Skip final chunk — some APIs send full text in it
                    if choice.get("finish_reason") is not None:
                        continue
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        user_msg, bot_msg = chat_history[-1]
                        chat_history[-1] = (user_msg, bot_msg + content)
                        yield chat_history
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        chat_history.append((message, f"[AI调用失败: {e}]"))
        yield chat_history

# ============================================================
# HTML Reports
# ============================================================

def _build_report(predictions: list, info: dict, llm_report: str = None) -> str:
    top = predictions[0]
    risk = top["risk"]
    risk_color = {"HIGH": "#ef5350", "MEDIUM": "#ff9800", "LOW": "#66bb6a"}[risk]

    # Confidence bars
    bar_rows = ""
    colors = ["#4fc3f7", "#81c784", "#ce93d8", "#ffb74d", "#e57373"]
    for i, pred in enumerate(predictions):
        color = colors[min(i, len(colors)-1)]
        bar_rows += f"""
        <div style="margin: 6px 0;">
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:1px;">
                <span>{'→' if i == 0 else ' '} {pred.get('class_zh', pred['class'])}</span>
                <span style="color:#aaa;">{pred['confidence']:.1%}</span>
            </div>
            <div style="background:#2a2a3e; border-radius:3px; height:6px;">
                <div style="background:{color}; border-radius:3px; height:6px; width:{pred['confidence']*100:.0f}%;"></div>
            </div>
        </div>"""

    risk_cfg = {
        "HIGH":   ("🚨 高风险警告", "建议立即就医，由皮肤科医生进行专业诊断。", "#3a1a1a", "#ef5350"),
        "MEDIUM": ("⚠️ 中等风险", "建议安排临床随访，进一步评估。", "#2a2a1a", "#ff9800"),
        "LOW":    ("✅ 低风险", "倾向良性表现，建议常规观察。", "#1a2a1a", "#66bb6a"),
    }
    risk_title, risk_msg, bg, border = risk_cfg[risk]

    # LLM report section
    llm_html = ""
    if llm_report:
        llm_html = f"""
        <div style="margin-top:12px; padding:12px; background:#1a2a1a; border-radius:8px; border-left:3px solid #4fc3f7; font-size:12px;">
            <b style="color:#4fc3f7;">🤖 AI 辅助建议</b>
            <pre style="margin:8px 0 0 0; white-space:pre-wrap; font-family:system-ui,sans-serif; color:#ddd; line-height:1.6;">{llm_report}</pre>
        </div>"""

    return f"""
    <div style="padding:16px; background:#1a1a2e; border-radius:12px; color:#e0e0e0; font-family:system-ui,sans-serif; max-height:750px; overflow-y:auto;">
        <h3 style="margin:0 0 4px 0; color:#4fc3f7; font-size:16px;">🔬 分类结果</h3>

        <div style="padding:10px; background:{bg}; border-radius:8px; margin:8px 0; border-left:4px solid {border};">
            <div style="font-size:14px; font-weight:bold; color:{border};">{risk_title}: {top.get('class_zh', top['class'])}</div>
            <p style="color:#bbb; margin:3px 0 0 0; font-size:11px;">{risk_msg}</p>
        </div>

        {bar_rows}
        {llm_html}

        <div style="margin-top:10px; padding:6px 8px; background:#111; border-radius:4px; font-size:9px; color:#666; text-align:center;">
            ⚠️ 本系统为研究原型，不可用于临床诊断 | 请咨询皮肤科医生
        </div>
    </div>
    """


def _empty_report() -> str:
    return """
    <div style="padding:20px; background:#1a1a2e; border-radius:12px; color:#e0e0e0; font-family:system-ui,sans-serif;">
        <h3 style="margin:0; color:#4fc3f7;">🔬 分类结果</h3>
        <p style="color:#888; margin-top:20px; text-align:center;">请上传皮肤图像以开始分析</p>
    </div>
    """


def _model_info_html(model_path: str) -> str:
    if not model_path or not os.path.exists(model_path):
        return "<div style='color:#666;font-size:11px;'>未加载模型</div>"
    cached = manager.cache.get(model_path)
    if cached is None:
        return "<div style='color:#666;font-size:11px;'>加载中...</div>"
    _, _, _, _, _, _, info = cached
    variant = info.get("model_name", "?").replace("_", " ").title()
    val_acc = info.get("val_acc")
    return f"""
    <div style="padding:10px; background:#1a1a2e; border-radius:6px; color:#ccc; font-size:11px; font-family:system-ui,sans-serif;">
        <b style="color:#4fc3f7;">模型信息</b>
        <table style="width:100%; margin-top:4px; font-size:10px;">
            <tr><td style="color:#888;">架构</td><td>{variant}</td></tr>
            <tr><td style="color:#888;">池化</td><td>{info.get('pooling','?').upper()}</td></tr>
            <tr><td style="color:#888;">多尺度</td><td>{'✓' if info.get('multi_scale') else '✗'}</td></tr>
            <tr><td style="color:#888;">EMA</td><td>{'✓' if info.get('ema') else '✗'}</td></tr>
            <tr><td style="color:#888;">验证准确率</td><td style="color:#81c784;">{val_acc:.2%}</td></tr>
        </table>
    </div>
    """


# ============================================================
# Gradio UI
# ============================================================

def build_ui(models: list, theme, css: str):

    if models:
        model_choices = [m["path"] for m in models]
        default_model = model_choices[0]
    else:
        model_choices = ["runs/convnext_tiny/best.pt"]
        default_model = model_choices[0]

    with gr.Blocks(title="皮肤疾病智能分类系统") as demo:
        gr.Markdown("""
        <div style="text-align:center; margin-bottom:0;">
            <h1 style="margin:0 0 4px 0;">🩺 皮肤疾病智能辅助分类系统</h1>
            <p style="color:#888; font-size:14px; margin:0;">ConvNeXt · 22类皮肤病 · {'🤖 AI建议已启用' if llm_client else '💡 配置API key以启用AI建议'}</p>
        </div>
        """)

        with gr.Row(equal_height=True):
            # ── Left Column: Settings ──
            with gr.Column(scale=1, min_width=180, elem_id="col-settings"):
                gr.Markdown("#### ⚙️ 设置")
                model_dropdown = gr.Dropdown(
                    choices=model_choices, value=default_model,
                    label="模型选择", interactive=True,
                )
                top_k_slider = gr.Slider(
                    1, 5, 3, step=1, label="显示 Top-K",
                )
                cam_checkbox = gr.Checkbox(
                    label="Grad-CAM 热力图", value=False,
                )
                gr.Markdown("---")
                model_info = gr.HTML(value=_model_info_html(default_model))

            # ── Center Column: Image I/O ──
            with gr.Column(scale=3):
                with gr.Row():
                    input_image = gr.Image(
                        label="上传皮肤图像", type="numpy", height=380,
                        sources=["upload", "clipboard", "webcam"],
                    )
                    output_image = gr.Image(
                        label="分析结果", type="numpy", height=380,
                    )

                classify_btn = gr.Button("🔬 开始分析", variant="primary", size="lg")
                cam_output = gr.Image(
                    label="Grad-CAM 激活热力图 (模型关注区域)", type="numpy",
                    height=200, visible=True,
                )

            # ── Right Column: Report ──
            with gr.Column(scale=2, min_width=280, elem_id="col-report"):
                report_html = gr.HTML(value=_empty_report())
                # Chat state and UI
                chat_state = gr.State({})
                chatbot = gr.Chatbot(label="💬 AI 对话咨询", height=260)
                chat_input = gr.Textbox(
                    placeholder="输入问题... 如: 这个病严重吗？怎么治疗？会传染吗？",
                    label="咨询AI助手", scale=4,
                )
                chat_btn = gr.Button("发送", variant="secondary", scale=1)

        # --- Events ---
        classify_inputs = [input_image, model_dropdown, cam_checkbox, top_k_slider, chat_state]
        classify_outputs = [output_image, report_html, cam_output, model_info, chat_state, chatbot]

        classify_btn.click(fn=classify, inputs=classify_inputs, outputs=classify_outputs)
        input_image.change(fn=classify, inputs=classify_inputs, outputs=classify_outputs)

        # Chat events
        chat_btn.click(
            fn=chat, inputs=[chat_input, chatbot, chat_state], outputs=[chatbot]
        ).then(lambda: "", outputs=[chat_input])
        chat_input.submit(
            fn=chat, inputs=[chat_input, chatbot, chat_state], outputs=[chatbot]
        ).then(lambda: "", outputs=[chat_input])

        # Footer
        gr.Markdown("""
        <div style="text-align:center; color:#555; font-size:10px; margin-top:8px;">
        ⚠️ 本系统为学术研究原型，非医疗器械。所有结果仅供参考，不可用于临床诊断。
        </div>
        """)

    return demo


# ============================================================
# Main
# ============================================================

llm_client = None
llm_model = ""

def _load_config() -> dict:
    """Load config.json (or config.example.json) from project root."""
    for name in ["config.json", "config.example.json"]:
        config_path = Path(__file__).parent / name
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
                # Skip if it's the example placeholder
                api_key = cfg.get("llm", {}).get("api_key", "")
                if "your-api-key" in api_key or "sk-your" in api_key:
                    continue  # skip placeholder, try next file
                return cfg
            except Exception:
                continue
    return {}


def main():
    global llm_client, llm_model

    # Load config file first (lower priority than CLI)
    cfg = _load_config()
    llm_cfg = cfg.get("llm", {})

    parser = argparse.ArgumentParser(description="Skin Disease Classification Demo")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--api-key", type=str, default=llm_cfg.get("api_key"),
                        help="OpenAI-compatible API key (falls back to config.json)")
    parser.add_argument("--api-base", type=str, default=llm_cfg.get("api_base", "https://api.openai.com/v1"),
                        help="API base URL")
    parser.add_argument("--api-model", type=str, default=llm_cfg.get("api_model", "gpt-4o-mini"),
                        help="LLM model name")
    args = parser.parse_args()

    # Init LLM client
    if args.api_key:
        llm_client = LLMClient(args.api_key, args.api_base, args.api_model)
        llm_model = args.api_model
        print(f"🤖 LLM enabled: {args.api_model} @ {args.api_base}")
    else:
        print("💡 提示: 创建 config.json 或使用 --api-key 启用AI大模型")
        print("   复制 config.example.json 为 config.json 并填入API key")

    models = find_models("runs")
    if args.model and not any(m["path"] == args.model for m in models):
        models.insert(0, {"path": args.model, "name": Path(args.model).parent.name, "acc_str": ""})

    if not models:
        print("⚠️  未找到已训练模型，请先运行: python train.py")
    else:
        print(f"📦 发现 {len(models)} 个模型")

    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    css = """
    footer { visibility: hidden; }
    .gradio-container { max-width: 1400px !important; }
    #col-settings { background: #1e1e2e; border-radius: 12px; padding: 16px; }
    #col-report { background: #1e1e2e; border-radius: 12px; padding: 4px; max-height: 720px; overflow-y: auto; }
    """
    demo = build_ui(models, theme, css)
    demo.queue(max_size=20)
    demo.launch(server_port=args.port, share=args.share, show_error=True,
                theme=theme, css=css)


if __name__ == "__main__":
    main()
