# -*- coding: utf-8 -*-
"""
Task 4: 多任务联合学习与可解释性诊断 (HAPNet-MTL)

功能概述：
- 同一模型中同时完成：
    1）病害分类（61类）
    2）严重度分级（4类：healthy/mild/moderate/severe，采用Ordinal Regression）
- 使用共享 backbone 的多任务框架
- 训练完成后：
    - 在验证集上评估病害分类和严重度分级指标（Acc, Macro-F1）
    - 生成部分样本的诊断报告（包含置信度、严重度、病灶位置/覆盖率等）
    - 输出针对诊断报告的 Grad-CAM 叠加图

依赖：
    pip install timm seaborn matplotlib scikit-learn pandas tqdm
"""

import os
import json
import random
import math
import time
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import timm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, \
    f1_score, accuracy_score

# ==========================
# 科研配色
# ==========================
COLORS = ['#90D8A6', '#83A1E7', '#E992A9', '#D2CAF8',
          '#F7AF7F', '#B0D9F9', '#E7B6BC', '#B0CDED']
sns.set_palette(COLORS)

# ==========================
# 路径配置（根据你的环境）
# ==========================
BASE_DIR = r"D:\PyCharmproject\2025ShuWeiCup\Data"
IMG_DIR = os.path.join(BASE_DIR, "AgriculturalDisease_trainingset", "images")
ANN_JSON = os.path.join(
    BASE_DIR, "AgriculturalDisease_trainingset", "AgriculturalDisease_train_annotations_fixed.json"
)

OUTPUT_DIR = "./hapnet_mtl_outputs"

# 可选：从问题一加载 backbone 的预训练权重（如果不想用，设为 None）
TASK1_CKPT = r"D:\PyCharmproject\2025ShuWeiCup\问题一\outputs\best_model.pth"

# ==========================
# 超参数
# ==========================
NUM_DISEASE_CLASSES = 61
NUM_SEVERITY_CLASSES = 4      # healthy/mild/moderate/severe
NUM_THRESHOLDS = NUM_SEVERITY_CLASSES - 1  # ordinal 阈值个数

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_WORKERS = 4

VAL_RATIO = 0.1

BASE_LR = 3e-4
WEIGHT_DECAY = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2025

# 多任务权重：严重度 loss 相对于病害 loss 的权重
LAMBDA_SEV = 0.5

# Backbone 名称（timm）
BACKBONE_NAME = "tf_efficientnet_b0_ns"


# ==========================
# 工具函数
# ==========================
def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# ==========================
# 从 disease_class → severity 的映射（可以根据题意修改）
# ==========================
def map_disease_to_severity(disease_id: int) -> int:
    """
    !!! 关键函数：根据官方文档/JSON 标注设计 !!!
    当前示意策略（需你根据真实任务调整）：
        0        : healthy
        1  - 20  : mild
        21 - 40  : moderate
        41 - 60  : severe
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
# 数据集定义
# ==========================
class MultiTaskDataset(Dataset):
    """
    多任务数据集：同时返回 img, disease, severity
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
        print(f"[MultiTaskDataset] {len(self.samples)} samples.")

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


# ==========================
# 数据增强
# ==========================
def get_transforms(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.2),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    val_tf = T.Compose([
        T.Resize(int(img_size * 1.1)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ==========================
# 多任务网络：backbone + disease head + severity ordinal head
# ==========================
class HAPNetMTL(nn.Module):
    """
    HAPNet-MTL: Hierarchical & Attentive Prototype-like backbone for
    Multi-Task Learning (disease + severity)
    这里用 EfficientNet 作为 backbone，双头输出：
      - disease: 多类 CE
      - severity: ordinal regression (K 阈值)
    """
    def __init__(
        self,
        backbone_name: str,
        num_disease: int,
        num_severity: int,
        num_thresholds: int,
        task1_ckpt: str = None,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,        # 输出特征
            global_pool="avg"
        )

        if hasattr(self.backbone, "num_features"):
            feat_dim = self.backbone.num_features
        else:
            feat_dim = 1280

        self.head_disease = nn.Linear(feat_dim, num_disease)
        self.head_severity = nn.Linear(feat_dim, num_thresholds)

        # 尝试从问题一加载 backbone 权重（可选）
        if task1_ckpt is not None and os.path.exists(task1_ckpt):
            print(f"[HAPNetMTL] Try to load Task1 backbone from: {task1_ckpt}")
            try:
                state = torch.load(task1_ckpt, map_location="cpu")
                missing, unexpected = self.backbone.load_state_dict(state, strict=False)
                print(f"Loaded Task1 weights to backbone (strict=False). "
                      f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            except Exception as e:
                print(f"[Warning] Loading Task1 weights failed: {e}")
                print("[HAPNetMTL] Fallback to ImageNet pretrained backbone.")
        else:
            print("[HAPNetMTL] Task1 checkpoint not found or not set, using ImageNet pretrained backbone.")

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)              # [B, D]
        logits_dis = self.head_disease(feat) # [B, num_disease]
        logits_sev = self.head_severity(feat) # [B, K]
        return logits_dis, logits_sev, feat


# ==========================
# 严重度：有序回归辅助函数
# ==========================
def ordinal_targets(severity: torch.Tensor, num_thresholds: int) -> torch.Tensor:
    """
    severity: [B] ∈ {0,1,2,3}
    输出：targets: [B,K], y_k = 1(y>=k+1)
    """
    bsz = severity.size(0)
    device = severity.device
    thresholds = torch.arange(1, num_thresholds + 1, device=device).view(1, -1)  # [1,K]
    sev = severity.view(-1, 1)  # [B,1]
    targets = (sev >= thresholds).float()
    return targets


def ordinal_loss(logits_sev: torch.Tensor, severity: torch.Tensor):
    """
    多阈值二分类交叉熵之和
    """
    num_thresholds = logits_sev.size(1)
    targets = ordinal_targets(severity, num_thresholds)  # [B,K]
    loss = F.binary_cross_entropy_with_logits(logits_sev, targets, reduction="mean")
    return loss


def severity_from_logits(logits_sev: torch.Tensor) -> torch.Tensor:
    """
    阈值 -> 严重度预测： \hat{y} = sum(I(sigmoid(logit_k)>0.5))
    """
    probs = torch.sigmoid(logits_sev)
    preds_bin = (probs > 0.5).long()
    severity_pred = preds_bin.sum(dim=1)
    return severity_pred


def severity_probs_from_logits(logits_sev: torch.Tensor) -> torch.Tensor:
    """
    将 ordinal 阈值概率转换为 4 类概率分布：
      P0=1-p1, P1=p1-p2, P2=p2-p3, P3=p3
    """
    p = torch.sigmoid(logits_sev)  # [B,K], K=3
    p1 = p[:, 0]
    p2 = p[:, 1]
    p3 = p[:, 2]
    P0 = 1.0 - p1
    P1 = p1 - p2
    P2 = p2 - p3
    P3 = p3
    probs = torch.stack([P0, P1, P2, P3], dim=1)  # [B,4]
    probs = torch.clamp(probs, min=0.0)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    return probs


# ==========================
# 训练 & 验证
# ==========================
def train_one_epoch(
    epoch: int,
    model: HAPNetMTL,
    optimizer,
    scheduler,
    train_loader,
    train_tf,
):
    model.train()
    running_loss = 0.0
    running_loss_dis = 0.0
    running_loss_sev = 0.0
    running_acc_dis = 0.0
    running_acc_sev = 0.0
    n_samples = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", ncols=120)
    for imgs_pil, disease, severity in pbar:
        disease = disease.to(DEVICE, non_blocking=True)
        severity = severity.to(DEVICE, non_blocking=True)
        bsz = severity.size(0)

        imgs = torch.stack([train_tf(img) for img in imgs_pil]).to(DEVICE)

        logits_dis, logits_sev, feat = model(imgs)

        loss_dis = F.cross_entropy(logits_dis, disease)
        loss_sev = ordinal_loss(logits_sev, severity)
        loss = loss_dis + LAMBDA_SEV * loss_sev

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # statistics
        n_samples += bsz
        running_loss += loss.item() * bsz
        running_loss_dis += loss_dis.item() * bsz
        running_loss_sev += loss_sev.item() * bsz

        with torch.no_grad():
            pred_dis = logits_dis.argmax(dim=1)
            acc_dis = (pred_dis == disease).float().mean().item()
            pred_sev = severity_from_logits(logits_sev)
            acc_sev = (pred_sev == severity).float().mean().item()
            running_acc_dis += acc_dis * bsz
            running_acc_sev += acc_sev * bsz

        avg_loss = running_loss / n_samples
        avg_acc_dis = running_acc_dis / n_samples
        avg_acc_sev = running_acc_sev / n_samples
        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc_dis=f"{avg_acc_dis:.4f}",
            acc_sev=f"{avg_acc_sev:.4f}"
        )

    epoch_loss = running_loss / n_samples
    epoch_loss_dis = running_loss_dis / n_samples
    epoch_loss_sev = running_loss_sev / n_samples
    epoch_acc_dis = running_acc_dis / n_samples
    epoch_acc_sev = running_acc_sev / n_samples

    return epoch_loss, epoch_loss_dis, epoch_loss_sev, epoch_acc_dis, epoch_acc_sev


@torch.no_grad()
def validate(
    epoch: int,
    model: HAPNetMTL,
    val_loader,
    val_tf,
):
    model.eval()
    running_loss = 0.0
    n_samples = 0

    all_dis_true = []
    all_dis_pred = []
    all_sev_true = []
    all_sev_pred = []

    pbar = tqdm(val_loader, desc=f"Val   Epoch {epoch}", ncols=120)
    for imgs_pil, disease, severity in pbar:
        disease = disease.to(DEVICE, non_blocking=True)
        severity = severity.to(DEVICE, non_blocking=True)
        bsz = severity.size(0)

        imgs = torch.stack([val_tf(img) for img in imgs_pil]).to(DEVICE)

        logits_dis, logits_sev, feat = model(imgs)

        loss_dis = F.cross_entropy(logits_dis, disease)
        loss_sev = ordinal_loss(logits_sev, severity)
        loss = loss_dis + LAMBDA_SEV * loss_sev

        running_loss += loss.item() * bsz
        n_samples += bsz

        pred_dis = logits_dis.argmax(dim=1)
        pred_sev = severity_from_logits(logits_sev)

        all_dis_true.append(disease.cpu().numpy())
        all_dis_pred.append(pred_dis.cpu().numpy())
        all_sev_true.append(severity.cpu().numpy())
        all_sev_pred.append(pred_sev.cpu().numpy())

    epoch_loss = running_loss / n_samples

    all_dis_true = np.concatenate(all_dis_true, axis=0)
    all_dis_pred = np.concatenate(all_dis_pred, axis=0)
    all_sev_true = np.concatenate(all_sev_true, axis=0)
    all_sev_pred = np.concatenate(all_sev_pred, axis=0)

    acc_dis = accuracy_score(all_dis_true, all_dis_pred)
    macro_f1_dis = f1_score(all_dis_true, all_dis_pred, average="macro")

    acc_sev = accuracy_score(all_sev_true, all_sev_pred)
    macro_f1_sev = f1_score(all_sev_true, all_sev_pred, average="macro")

    print(
        f"Val Epoch [{epoch}] Loss: {epoch_loss:.4f}, "
        f"Dis-Acc: {acc_dis:.4f}, Dis-MacroF1: {macro_f1_dis:.4f}, "
        f"Sev-Acc: {acc_sev:.4f}, Sev-MacroF1: {macro_f1_sev:.4f}"
    )

    return epoch_loss, acc_dis, macro_f1_dis, acc_sev, macro_f1_sev, \
        all_dis_true, all_dis_pred, all_sev_true, all_sev_pred


# ==========================
# Grad-CAM 用于诊断解释
# ==========================
class GradCAM:
    def __init__(self, model: HAPNetMTL, target_layer_name: str = "backbone.conv_head"):
        self.model = model
        self.model.eval()

        module = model
        for attr in target_layer_name.split("."):
            module = getattr(module, attr)
        self.target_layer = module

        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, img_tensor: torch.Tensor, use_disease_head: bool = True):
        """
        img_tensor: [1,3,H,W]
        use_disease_head: True 用 disease logit 做 Grad-CAM；
                          False 用 severity 阈值 logit 之和。
        """
        self.model.zero_grad()
        logits_dis, logits_sev, feat = self.model(img_tensor)

        if use_disease_head:
            # 以预测的 disease logit 作为目标
            pred_dis = logits_dis.argmax(dim=1)[0]
            score = logits_dis[0, pred_dis]
        else:
            # 使用所有阈值 logit 之和，偏向严重度相关区域
            score = logits_sev.sum()

        score.backward(retain_graph=True)

        act = self.activations  # [1,C,H,W]
        grad = self.gradients   # [1,C,H,W]

        weights = grad.mean(dim=(2, 3), keepdim=True)  # [1,C,1,1]
        cam = (weights * act).sum(dim=1, keepdim=True) # [1,1,H,W]
        cam = F.relu(cam)

        cam = cam.squeeze(0).squeeze(0)  # [H,W]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


# ==========================
# 诊断报告生成模块
# ==========================
def summarize_cam_region(cam: np.ndarray, threshold: float = 0.5) -> Tuple[str, float]:
    """
    根据 CAM 热力图估计病灶覆盖比例和位置（上/中/下）。
    cam: [H,W] ∈ [0,1]
    """
    H, W = cam.shape
    mask = cam > threshold
    coverage = mask.sum() / (H * W + 1e-8)  # 覆盖比例

    if coverage < 0.05:
        coverage_desc = "病斑范围较小"
    elif coverage < 0.2:
        coverage_desc = "病斑范围有限"
    elif coverage < 0.5:
        coverage_desc = "病斑范围中等"
    else:
        coverage_desc = "病斑范围较大"

    # 简单按垂直方向质心估计病灶主要位置
    if mask.sum() > 0:
        ys, xs = np.where(mask)
        y_mean = ys.mean() / H
        if y_mean < 1 / 3:
            loc_desc = "主要集中在叶片上部"
        elif y_mean < 2 / 3:
            loc_desc = "主要集中在叶片中部"
        else:
            loc_desc = "主要集中在叶片下部"
    else:
        loc_desc = "未出现明显集中病灶区域"

    desc = f"{coverage_desc}，{loc_desc}"
    return desc, float(coverage)


def severity_label_to_str(sev_id: int) -> str:
    mapping = {
        0: "健康",
        1: "轻度",
        2: "中度",
        3: "重度"
    }
    return mapping.get(int(sev_id), f"未知等级({sev_id})")


def generate_diagnosis_report(
    img_pil: Image.Image,
    disease_probs: np.ndarray,
    severity_probs: np.ndarray,
    disease_id_to_name: dict,
    cam: np.ndarray,
    topk: int = 3,
) -> str:
    """
    根据模型输出构造中文诊断报告：
      - 主要病害 + 严重度
      - 置信度
      - 次要候选
      - CAM 病灶位置/覆盖描述
    """
    # 病害 top-k
    dis_probs = disease_probs
    topk_idx = np.argsort(dis_probs)[-topk:][::-1]
    top1 = topk_idx[0]
    conf_dis = dis_probs[top1]
    dis_name = disease_id_to_name.get(int(top1), f"病害ID-{top1}")

    # 严重度
    sev_probs = severity_probs
    sev_id = int(sev_probs.argmax())
    conf_sev = float(sev_probs[sev_id])
    sev_name = severity_label_to_str(sev_id)

    # 混淆候选
    candidates = []
    for idx in topk_idx[1:]:
        candidates.append((int(idx), float(dis_probs[idx])))
    candidate_strs = [
        f"{disease_id_to_name.get(cid, f'病害ID-{cid}')} (概率 {p:.2f})"
        for cid, p in candidates
    ]
    if len(candidate_strs) == 0:
        conf_candidates = "无明显其他候选病害。"
    else:
        conf_candidates = "可能的次要候选包括：" + "，".join(candidate_strs) + "。"

    # CAM 描述
    cam_desc, coverage = summarize_cam_region(cam, threshold=0.5)

    report = []
    report.append(f"模型判定该叶片主要病害为【{dis_name}】，严重程度为【{sev_name}】。")
    report.append(f"病害分类置信度为 {conf_dis:.2f}，严重度置信度为 {conf_sev:.2f}。")
    report.append(f"基于 Grad-CAM 的病灶区域分析：{cam_desc}，估计病斑覆盖比例约为 {coverage*100:.1f}%。")
    report.append(conf_candidates)

    return "\n".join(report)


# ==========================
# 主训练 + 诊断示例
# ==========================
def main():
    set_seed(SEED)

    out_dir = prepare_dir(OUTPUT_DIR)
    print(f"Use device: {DEVICE}")
    print(f"Output dir: {out_dir}")

    # 1. 读取 JSON，划分 train/val
    with open(ANN_JSON, "r", encoding="utf-8") as f:
        anns = json.load(f)

    random.shuffle(anns)
    total = len(anns)
    val_len = int(total * VAL_RATIO)
    val_anns = anns[:val_len]
    train_anns = anns[val_len:]

    train_dataset = MultiTaskDataset(IMG_DIR, train_anns)
    val_dataset = MultiTaskDataset(IMG_DIR, val_anns)
    train_dataset.img_dir = IMG_DIR
    val_dataset.img_dir = IMG_DIR

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_pil,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_pil,
    )

    train_tf, val_tf = get_transforms(IMG_SIZE)

    # 2. 构建多任务模型
    model = HAPNetMTL(
        backbone_name=BACKBONE_NAME,
        num_disease=NUM_DISEASE_CLASSES,
        num_severity=NUM_SEVERITY_CLASSES,
        num_thresholds=NUM_THRESHOLDS,
        task1_ckpt=TASK1_CKPT,
    ).to(DEVICE)

    # 3. 优化器 & 学习率调度
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = len(train_loader) * 5

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 4. 训练循环
    best_val_score = 0.0  # 可以用 disease Macro-F1 + severity Macro-F1 的和
    history = []

    csv_path = os.path.join(out_dir, "training_history_task4.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,train_loss_dis,train_loss_sev,train_acc_dis,train_acc_sev,"
                    "val_loss,val_acc_dis,val_macro_f1_dis,val_acc_sev,val_macro_f1_sev\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_loss_dis, train_loss_sev, train_acc_dis, train_acc_sev = train_one_epoch(
            epoch, model, optimizer, scheduler, train_loader, train_tf
        )

        val_loss, acc_dis, macro_f1_dis, acc_sev, macro_f1_sev, \
            dis_true, dis_pred, sev_true, sev_pred = validate(
                epoch, model, val_loader, val_tf
        )

        epoch_time = time.time() - start_time

        # 综合评分（可以自定义，这里简单相加）
        score = macro_f1_dis + macro_f1_sev
        if score > best_val_score:
            best_val_score = score
            best_path = os.path.join(out_dir, "best_model_task4.pth")
            torch.save(model.state_dict(), best_path)
            print(f"*** New best model (task4) at epoch {epoch}, score={best_val_score:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_loss_dis": float(train_loss_dis),
            "train_loss_sev": float(train_loss_sev),
            "train_acc_dis": float(train_acc_dis),
            "train_acc_sev": float(train_acc_sev),
            "val_loss": float(val_loss),
            "val_acc_dis": float(acc_dis),
            "val_macro_f1_dis": float(macro_f1_dis),
            "val_acc_sev": float(acc_sev),
            "val_macro_f1_sev": float(macro_f1_sev),
        })

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_loss_dis:.6f},{train_loss_sev:.6f},"
                f"{train_acc_dis:.6f},{train_acc_sev:.6f},"
                f"{val_loss:.6f},{acc_dis:.6f},{macro_f1_dis:.6f},{acc_sev:.6f},{macro_f1_sev:.6f}\n"
            )

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"TrainLoss: {train_loss:.4f}, TrainAccDis: {train_acc_dis:.4f}, TrainAccSev: {train_acc_sev:.4f}, "
            f"ValLoss: {val_loss:.4f}, DisAcc: {acc_dis:.4f}, DisF1: {macro_f1_dis:.4f}, "
            f"SevAcc: {acc_sev:.4f}, SevF1: {macro_f1_sev:.4f}, Time: {epoch_time:.1f}s"
        )

    with open(os.path.join(out_dir, "training_history_task4.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"Training finished. Best Val Score (F1_dis+F1_sev) = {best_val_score:.4f}")

    # 5. 保存最终一次验证的详细分类报告
    report_dis = classification_report(
        dis_true, dis_pred,
        labels=list(range(NUM_DISEASE_CLASSES)),
        output_dict=True,
        zero_division=0
    )
    report_sev = classification_report(
        sev_true, sev_pred,
        labels=list(range(NUM_SEVERITY_CLASSES)),
        target_names=[f"severity_{i}" for i in range(NUM_SEVERITY_CLASSES)],
        output_dict=True,
        zero_division=0
    )

    with open(os.path.join(out_dir, "classification_report_task4_disease.json"), "w", encoding="utf-8") as f:
        json.dump(report_dis, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "classification_report_task4_severity.json"), "w", encoding="utf-8") as f:
        json.dump(report_sev, f, indent=2, ensure_ascii=False)

    # 6. 简单构造 disease_id -> name 的映射（真实项目中可根据官方表/字典替换）
    disease_id_to_name = {i: f"病害类别-{i}" for i in range(NUM_DISEASE_CLASSES)}

    # 7. 使用最佳模型加载，生成部分验证样本的诊断报告 + Grad-CAM 可视化
    best_model_path = os.path.join(out_dir, "best_model_task4.pth")
    if os.path.exists(best_model_path):
        state = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(state, strict=True)
        print(f"[Diag] Loaded best model from {best_model_path} for diagnosis demo.")

    cam_dir = os.path.join(out_dir, "diagnosis_samples")
    os.makedirs(cam_dir, exist_ok=True)

    cam_gen = GradCAM(model, target_layer_name="backbone.conv_head")

    # 随机从验证集中选 N 个样本做诊断报告
    num_demo = 5
    indices = np.random.choice(len(val_dataset), size=min(num_demo, len(val_dataset)), replace=False)

    for idx in indices:
        img_name, disease, severity = val_dataset.samples[idx]
        img_path = os.path.join(val_dataset.img_dir, img_name)
        img_pil = Image.open(img_path).convert("RGB")

        img_t = val_tf(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits_dis, logits_sev, feat = model(img_t)
            probs_dis = F.softmax(logits_dis, dim=1)[0].cpu().numpy()
            probs_sev = severity_probs_from_logits(logits_sev)[0].cpu().numpy()
            pred_dis = int(probs_dis.argmax())
            pred_sev = int(probs_sev.argmax())

        # Grad-CAM（这里使用 disease head）
        cam = cam_gen.generate_cam(img_t, use_disease_head=True)

        # 生成诊断报告
        report_text = generate_diagnosis_report(
            img_pil,
            disease_probs=probs_dis,
            severity_probs=probs_sev,
            disease_id_to_name=disease_id_to_name,
            cam=cam,
            topk=3,
        )

        # 保存图像 + CAM 叠加 + 文本报告
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(img_pil.size, resample=Image.BILINEAR)
        cam_np = np.array(cam_img) / 255.0

        plt.figure(figsize=(7, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_pil)
        plt.axis("off")
        plt.title(f"GT: dis={disease}, sev={severity}")

        plt.subplot(1, 2, 2)
        plt.imshow(img_pil)
        plt.imshow(cam_np, cmap="jet", alpha=0.4)
        plt.axis("off")
        plt.title(f"Pred: dis={pred_dis}, sev={pred_sev}")

        img_save_path = os.path.join(cam_dir, f"sample_{idx}_cam.png")
        plt.tight_layout()
        plt.savefig(img_save_path, dpi=300)
        plt.close()

        report_save_path = os.path.join(cam_dir, f"sample_{idx}_report.txt")
        with open(report_save_path, "w", encoding="utf-8") as f:
            f.write(f"图像文件：{img_name}\n")
            f.write(report_text)

        print(f"[Diag] Saved CAM to {img_save_path}")
        print(f"[Diag] Saved report to {report_save_path}")
        print("------诊断报告示例------")
        print(report_text)
        print("-----------------------")

    print("Task4 multi-task training & diagnosis demo finished.")


if __name__ == "__main__":
    main()
