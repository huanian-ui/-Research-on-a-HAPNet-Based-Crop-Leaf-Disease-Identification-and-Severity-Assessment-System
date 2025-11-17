# -*- coding: utf-8 -*-
"""
Task 4 辅助评估脚本：对比多任务 (Task4) 与单任务 (Task2/Task3) 的协同效应

功能：
- 使用同一份验证集，对以下模型进行评估：
    1）Task2：病害分类单任务模型（61 类）
    2）Task3：严重度分级单任务模型（4 档有序）
    3）Task4：多任务模型（同时输出病害 + 严重度）
- 输出：
    - 病害分类：Accuracy, Macro-F1（Task2 vs Task4）
    - 严重度分级：Accuracy, Macro-F1（Task3 vs Task4）
- 将结果保存为 CSV，便于在论文或报告中画表、写协同效应分析。

依赖：
    pip install timm seaborn matplotlib scikit-learn pandas tqdm
"""

import os
import json
import random
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import timm

from sklearn.metrics import accuracy_score, f1_score

# ==========================
# 路径配置（根据你的环境修改）
# ==========================
BASE_DIR = r"D:\PyCharmproject\2025ShuWeiCup\Data"
IMG_DIR = os.path.join(BASE_DIR, "AgriculturalDisease_trainingset", "images")
ANN_JSON = os.path.join(
    BASE_DIR, "AgriculturalDisease_trainingset", "AgriculturalDisease_train_annotations_fixed.json"
)

# 模型权重路径（请根据实际情况确认或修改）
TASK2_CKPT = r"D:\PyCharmproject\2025ShuWeiCup\问题二\outputs\best_model.pth"       # 病害单任务模型
TASK3_CKPT = r".\hapnet_sg_outputs\best_model_task3.pth"                             # 严重度单任务模型（问题三）
TASK4_CKPT = r".\hapnet_mtl_outputs\best_model_task4.pth"                            # 多任务模型（问题四）

# 结果输出目录
OUTPUT_DIR = "./hapnet_mtl_eval_compare"

# ==========================
# 配置
# ==========================
NUM_DISEASE_CLASSES = 61
NUM_SEVERITY_CLASSES = 4
NUM_THRESHOLDS = NUM_SEVERITY_CLASSES - 1

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
VAL_RATIO = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2025

# Backbone 名称（需与训练时一致，如有不同请修改）
TASK2_BACKBONE = "swin_tiny_patch4_window7_224"
TASK3_BACKBONE = "tf_efficientnet_b0_ns"
TASK4_BACKBONE = "tf_efficientnet_b0_ns"


# ==========================
# 工具函数
# ==========================
def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# ==========================
# 从 disease_class → severity 的映射（需与题三、题四保持一致）
# ==========================
def map_disease_to_severity(disease_id: int) -> int:
    """
    占位示意：
        0        : healthy
        1  - 20  : mild
        21 - 40  : moderate
        41 - 60  : severe

    ⚠ 若真实题意中严重度由其它字段决定，请统一修改此函数，
      使 Task3/Task4/本文件中的映射逻辑完全一致。
    """
    if disease_id == 0:
        sev = 0
    elif 1 <= disease_id <= 20:
        sev = 1
    elif 21 <= disease_id <= 40:
        sev = 2
    else:
        sev = 3
    return sev


# ==========================
# 数据集 & transform
# ==========================
class SeverityDataset(Dataset):
    """
    验证集统一使用的多任务数据集：
    - 图像
    - 病害标签
    - 严重度标签（由 disease_class 映射）
    """
    def __init__(self, img_dir: str, ann_list: List[dict]):
        super().__init__()
        self.img_dir = img_dir
        self.samples = []
        for ann in ann_list:
            img_name = ann["image_id"]
            disease = int(ann["disease_class"])
            severity = map_disease_to_severity(disease)
            self.samples.append((img_name, disease, severity))
        print(f"[SeverityDataset] {len(self.samples)} samples in this split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, disease, severity = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        return img, disease, severity


def collate_fn_pil(batch):
    imgs, disease, severity = zip(*batch)
    disease = torch.tensor(disease, dtype=torch.long)
    severity = torch.tensor(severity, dtype=torch.long)
    return list(imgs), disease, severity


def get_val_transform(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_tf = T.Compose([
        T.Resize(int(img_size * 1.1)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return val_tf


# ==========================
# 模型定义（需与训练时结构保持一致）
# ==========================

# 1) Task2: 病害单任务模型（Swin-Tiny，61 类）
class DiseaseOnlyNet(nn.Module):
    def __init__(self, backbone_name: str, num_disease: int):
        super().__init__()
        # 直接用 timm 创建分类模型
        self.model = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=num_disease
        )

    def forward(self, x: torch.Tensor):
        logits = self.model(x)
        return logits


# 2) Task3: 严重度单任务模型（HAPNet-SG 结构）
class HAPNetSG(nn.Module):
    def __init__(self, backbone_name: str, num_disease: int,
                 num_severity: int, num_thresholds: int):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )

        if hasattr(self.backbone, "num_features"):
            feat_dim = self.backbone.num_features
        else:
            feat_dim = 1280

        self.head_disease = nn.Linear(feat_dim, num_disease)
        self.head_severity = nn.Linear(feat_dim, num_thresholds)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        logits_dis = self.head_disease(feat)
        logits_sev = self.head_severity(feat)
        return logits_dis, logits_sev, feat


# 3) Task4: 多任务模型（HAPNet-MTL，与前面问题四训练脚本一致）
class HAPNetMTL(nn.Module):
    def __init__(self, backbone_name: str, num_disease: int,
                 num_severity: int, num_thresholds: int):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )

        if hasattr(self.backbone, "num_features"):
            feat_dim = self.backbone.num_features
        else:
            feat_dim = 1280

        self.head_disease = nn.Linear(feat_dim, num_disease)
        self.head_severity = nn.Linear(feat_dim, num_thresholds)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        logits_dis = self.head_disease(feat)
        logits_sev = self.head_severity(feat)
        return logits_dis, logits_sev, feat


# ==========================
# 严重度有序回归：logit -> preds/probs
# ==========================
def severity_from_logits(logits_sev: torch.Tensor) -> torch.Tensor:
    """
    与题三/题四一致：
    - 给定 K 个阈值 logit，每个 logit_k 对应 P(y >= k+1)
    - 预测严重度：sum(I(sigmoid(logit_k)>0.5))
    """
    probs = torch.sigmoid(logits_sev)
    preds_bin = (probs > 0.5).long()
    severity_pred = preds_bin.sum(dim=1)
    return severity_pred


def severity_probs_from_logits(logits_sev: torch.Tensor) -> torch.Tensor:
    """
    将 ordinal 阈值概率转换为 4 类概率：
        p1 = P(y>=1), p2 = P(y>=2), p3 = P(y>=3)
        P0=1-p1; P1=p1-p2; P2=p2-p3; P3=p3
    """
    p = torch.sigmoid(logits_sev)  # [B,K], K=3
    p1 = p[:, 0]
    p2 = p[:, 1]
    p3 = p[:, 2]
    P0 = 1.0 - p1
    P1 = p1 - p2
    P2 = p2 - p3
    P3 = p3
    probs = torch.stack([P0, P1, P2, P3], dim=1)
    probs = torch.clamp(probs, min=0.0)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    return probs


# ==========================
# 评估函数
# ==========================
@torch.no_grad()
def eval_task2_disease_only(model: DiseaseOnlyNet, loader, val_tf):
    model.eval()
    all_true = []
    all_pred = []

    pbar = tqdm(loader, desc="Eval Task2 (Disease-only)", ncols=120)
    for imgs_pil, disease, severity in pbar:
        disease = disease.to(DEVICE, non_blocking=True)
        imgs = torch.stack([val_tf(img) for img in imgs_pil]).to(DEVICE)

        logits = model(imgs)
        pred = logits.argmax(dim=1)

        all_true.append(disease.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    acc = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro")
    return acc, macro_f1


@torch.no_grad()
def eval_task3_severity_only(model: HAPNetSG, loader, val_tf):
    model.eval()
    all_true = []
    all_pred = []

    pbar = tqdm(loader, desc="Eval Task3 (Severity-only)", ncols=120)
    for imgs_pil, disease, severity in pbar:
        severity = severity.to(DEVICE, non_blocking=True)
        imgs = torch.stack([val_tf(img) for img in imgs_pil]).to(DEVICE)

        logits_dis, logits_sev, feat = model(imgs)
        pred_sev = severity_from_logits(logits_sev)

        all_true.append(severity.cpu().numpy())
        all_pred.append(pred_sev.cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    acc = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro")
    return acc, macro_f1


@torch.no_grad()
def eval_task4_multitask(model: HAPNetMTL, loader, val_tf):
    model.eval()
    all_dis_true, all_dis_pred = [], []
    all_sev_true, all_sev_pred = [], []

    pbar = tqdm(loader, desc="Eval Task4 (Multi-task)", ncols=120)
    for imgs_pil, disease, severity in pbar:
        disease = disease.to(DEVICE, non_blocking=True)
        severity = severity.to(DEVICE, non_blocking=True)
        imgs = torch.stack([val_tf(img) for img in imgs_pil]).to(DEVICE)

        logits_dis, logits_sev, feat = model(imgs)
        pred_dis = logits_dis.argmax(dim=1)
        pred_sev = severity_from_logits(logits_sev)

        all_dis_true.append(disease.cpu().numpy())
        all_dis_pred.append(pred_dis.cpu().numpy())
        all_sev_true.append(severity.cpu().numpy())
        all_sev_pred.append(pred_sev.cpu().numpy())

    all_dis_true = np.concatenate(all_dis_true, axis=0)
    all_dis_pred = np.concatenate(all_dis_pred, axis=0)
    all_sev_true = np.concatenate(all_sev_true, axis=0)
    all_sev_pred = np.concatenate(all_sev_pred, axis=0)

    acc_dis = accuracy_score(all_dis_true, all_dis_pred)
    macro_f1_dis = f1_score(all_dis_true, all_dis_pred, average="macro")

    acc_sev = accuracy_score(all_sev_true, all_sev_pred)
    macro_f1_sev = f1_score(all_sev_true, all_sev_pred, average="macro")

    return acc_dis, macro_f1_dis, acc_sev, macro_f1_sev


# ==========================
# 主函数：统一评估 & 协同效应对比
# ==========================
def main():
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)

    print(f"Use device: {DEVICE}")
    print(f"Output dir: {OUTPUT_DIR}")

    # 1. 统一划分验证集（与 Task3/4 类似）
    with open(ANN_JSON, "r", encoding="utf-8") as f:
        anns = json.load(f)
    random.shuffle(anns)
    total = len(anns)
    val_len = int(total * VAL_RATIO)
    val_anns = anns[:val_len]

    val_dataset = SeverityDataset(IMG_DIR, val_anns)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_pil,
    )
    val_tf = get_val_transform(IMG_SIZE)

    # 2. 构建并加载 Task2 模型（病害单任务）
    print("\n========== Load Task2 (Disease-only) model ==========")
    if not os.path.exists(TASK2_CKPT):
        print(f"[Warning] Task2 ckpt not found: {TASK2_CKPT}")
        print("如果你没有训练或路径不同，请修改 TASK2_CKPT 或暂时跳过此部分。")
        acc2, f12 = None, None
    else:
        model2 = DiseaseOnlyNet(TASK2_BACKBONE, NUM_DISEASE_CLASSES).to(DEVICE)
        state2 = torch.load(TASK2_CKPT, map_location=DEVICE)
        model2.load_state_dict(state2, strict=True)
        acc2, f12 = eval_task2_disease_only(model2, val_loader, val_tf)
        print(f"[Task2] Disease-only  Acc: {acc2:.4f}, Macro-F1: {f12:.4f}")

    # 3. 构建并加载 Task3 模型（严重度单任务）
    print("\n========== Load Task3 (Severity-only) model ==========")
    if not os.path.exists(TASK3_CKPT):
        print(f"[Warning] Task3 ckpt not found: {TASK3_CKPT}")
        print("如果你没有训练或路径不同，请修改 TASK3_CKPT 或暂时跳过此部分。")
        acc3, f13 = None, None
    else:
        model3 = HAPNetSG(TASK3_BACKBONE, NUM_DISEASE_CLASSES,
                          NUM_SEVERITY_CLASSES, NUM_THRESHOLDS).to(DEVICE)
        state3 = torch.load(TASK3_CKPT, map_location=DEVICE)
        model3.load_state_dict(state3, strict=True)
        acc3, f13 = eval_task3_severity_only(model3, val_loader, val_tf)
        print(f"[Task3] Severity-only  Acc: {acc3:.4f}, Macro-F1: {f13:.4f}")

    # 4. 构建并加载 Task4 模型（多任务）
    print("\n========== Load Task4 (Multi-task) model ==========")
    if not os.path.exists(TASK4_CKPT):
        print(f"[Error] Task4 ckpt not found: {TASK4_CKPT}")
        print("请先运行问题四的训练脚本，生成 best_model_task4.pth。")
        return
    model4 = HAPNetMTL(TASK4_BACKBONE, NUM_DISEASE_CLASSES,
                       NUM_SEVERITY_CLASSES, NUM_THRESHOLDS).to(DEVICE)
    state4 = torch.load(TASK4_CKPT, map_location=DEVICE)
    model4.load_state_dict(state4, strict=True)
    acc4_dis, f14_dis, acc4_sev, f14_sev = eval_task4_multitask(model4, val_loader, val_tf)
    print(f"[Task4] Multi-task Disease  Acc: {acc4_dis:.4f}, Macro-F1: {f14_dis:.4f}")
    print(f"[Task4] Multi-task Severity Acc: {acc4_sev:.4f}, Macro-F1: {f14_sev:.4f}")

    # 5. 汇总为对比表
    rows = []

    # 病害任务对比（Task2 vs Task4）
    rows.append({
        "Task": "Disease-only (Task2)",
        "Type": "Disease",
        "Accuracy": acc2 if acc2 is not None else "N/A",
        "MacroF1": f12 if f12 is not None else "N/A",
    })
    rows.append({
        "Task": "Multi-task (Task4)",
        "Type": "Disease",
        "Accuracy": acc4_dis,
        "MacroF1": f14_dis,
    })

    # 严重度任务对比（Task3 vs Task4）
    rows.append({
        "Task": "Severity-only (Task3)",
        "Type": "Severity",
        "Accuracy": acc3 if acc3 is not None else "N/A",
        "MacroF1": f13 if f13 is not None else "N/A",
    })
    rows.append({
        "Task": "Multi-task (Task4)",
        "Type": "Severity",
        "Accuracy": acc4_sev,
        "MacroF1": f14_sev,
    })

    # 保存 CSV
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "task4_multitask_vs_single_compare.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n对比结果已保存到：{csv_path}")
    print("\n========== 协同效应示例解读建议 ==========")
    print("1）对比 Disease-only vs Multi-task(Disease)：")
    print("   若 Multi-task 的 Macro-F1 提升，说明严重度任务帮助了病害特征学习。")
    print("2）对比 Severity-only vs Multi-task(Severity)：")
    print("   若 Multi-task 的 Macro-F1 提升，说明病害语义对严重度分级有正向迁移。")
    print("你可以直接把这张表的结果写进论文/报告的“多任务协同效应分析”小节。")


if __name__ == "__main__":
    main()
