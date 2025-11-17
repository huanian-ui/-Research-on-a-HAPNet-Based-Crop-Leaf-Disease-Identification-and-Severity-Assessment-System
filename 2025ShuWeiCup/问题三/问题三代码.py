# -*- coding: utf-8 -*-
"""
Task 3: Disease Severity Grading (HAPNet-SG)

功能：
- 从 JSON 解析 disease_class，并根据自定义规则映射到 severity（0:healthy,1:mild,2:moderate,3:severe）
- 使用轻量级 backbone（EfficientNet-B0） + 严重度有序回归头 + 可选病害辅助头
- 尝试从问题一训练好的 best_model.pth 加载 backbone 参数做迁移学习
- 训练输出：Accuracy、macro-F1、每类 recall 等
- 可视化：Grad-CAM 可视化模型关注的病灶区域

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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# ==========================
# 科研配色
# ==========================
COLORS = ['#90D8A6', '#83A1E7', '#E992A9', '#D2CAF8',
          '#F7AF7F', '#B0D9F9', '#E7B6BC', '#B0CDED']
sns.set_palette(COLORS)

# ==========================
# 路径配置（按你的本地环境修改）
# ==========================
BASE_DIR = r"D:\PyCharmproject\2025ShuWeiCup\Data"

IMG_DIR = os.path.join(BASE_DIR, "AgriculturalDisease_trainingset", "images")
ANN_JSON = os.path.join(BASE_DIR, "AgriculturalDisease_trainingset", "AgriculturalDisease_train_annotations_fixed.json")

# 可以单独再准备一个验证 JSON；如果没有，就从训练集中划分
VAL_RATIO = 0.1

OUTPUT_DIR = "./hapnet_sg_outputs"

# 问题一的 best_model 路径（用于迁移学习）
TASK1_CKPT = r"D:\PyCharmproject\2025ShuWeiCup\问题一\outputs\best_model.pth"

# ==========================
# 超参数
# ==========================
NUM_DISEASE_CLASSES = 61        # 与任务一/二一致
NUM_SEVERITY_CLASSES = 4        # healthy / mild / moderate / severe

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_WORKERS = 4

BASE_LR = 3e-4
WEIGHT_DECAY = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2025

# 多任务权重：严重度 + 病害
LAMBDA_DIS = 0.3   # 辅助任务病害分类 loss 系数

# 有序回归阈值数量（num_severity - 1）
NUM_THRESHOLDS = NUM_SEVERITY_CLASSES - 1

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
    if os.path.isfile(path):
        root, name = os.path.split(path)
        if not root:
            root = "."
        path = os.path.join(root, name + "_dir")
    os.makedirs(path, exist_ok=True)
    return path


# ==========================
# 从 disease_class → severity 的映射
# ==========================
def map_disease_to_severity(disease_id: int) -> int:
    """
    !!! 关键函数：根据官方文档/JSON 标注设计 !!!
    当前只是示例规则，你需要按题三的真实标注来改。

    示例占位策略：
    - 这里假设：
        0: 健康
        1-20: 轻度
        21-40: 中度
        41-60: 重度
    请根据实际的 JSON 字段来实现，比如根据“病斑比例/level 字段”等。
    """
    if disease_id == 0:
        sev = 0  # healthy
    elif 1 <= disease_id <= 20:
        sev = 1  # mild
    elif 21 <= disease_id <= 40:
        sev = 2  # moderate
    else:
        sev = 3  # severe
    return sev


# ==========================
# 数据集定义
# ==========================
class SeverityDataset(Dataset):
    def __init__(self, img_dir: str, ann_list: List[dict]):
        super().__init__()
        self.img_dir = img_dir
        self.samples = []
        for ann in ann_list:
            img_name = ann["image_id"]
            disease = int(ann["disease_class"])
            severity = map_disease_to_severity(disease)
            self.samples.append((img_name, disease, severity))

        print(f"[SeverityDataset] {len(self.samples)} samples.")

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
# HAPNet-SG 模型：backbone + disease head + severity ordinal head
# ==========================
class HAPNetSG(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_disease: int,
        num_severity: int,
        num_thresholds: int,
        task1_ckpt: str = None,
    ):
        super().__init__()
        # backbone 输出 global pool 特征
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,      # 先用 ImageNet 预训练
            num_classes=0,        # 输出特征，不做分类
            global_pool="avg"
        )

        if hasattr(self.backbone, "num_features"):
            feat_dim = self.backbone.num_features
        else:
            feat_dim = 1280

        # 病害分类头（辅助任务）
        self.head_disease = nn.Linear(feat_dim, num_disease)

        # 严重度有序回归头：输出 K 个阈值的 logit，K = num_severity - 1
        self.head_severity = nn.Linear(feat_dim, num_thresholds)

        # 尝试从问题一 ckpt 加载 backbone 参数
        if task1_ckpt is not None and os.path.exists(task1_ckpt):
            print(f"[HAPNetSG] Try to load Task1 weights from: {task1_ckpt}")
            try:
                state = torch.load(task1_ckpt, map_location="cpu")
                # 假设问题一的模型也是 timm 模型，其 state_dict 中与 backbone 同名的权重可以利用
                # 这里采用 strict=False，只加载能对上的部分参数
                missing, unexpected = self.backbone.load_state_dict(state, strict=False)
                print(f"Loaded Task1 weights to backbone (strict=False). "
                      f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            except Exception as e:
                print(f"[Warning] Loading Task1 weights failed: {e}")
                print("[HAPNetSG] Will use ImageNet pretrained backbone only.")
        else:
            print("[HAPNetSG] Task1 checkpoint not found or not set, using ImageNet pretrained backbone.")

    def forward(self, x: torch.Tensor):
        """
        返回：
        - logits_disease: [B, num_disease]
        - logits_sev: [B, num_thresholds]（每个元素经 sigmoid 后为 P(y >= k)）
        - feat: [B, D]
        """
        feat = self.backbone(x)          # [B, D]
        logits_dis = self.head_disease(feat)
        logits_sev = self.head_severity(feat)  # [B, K]
        return logits_dis, logits_sev, feat


# ==========================
# 有序回归损失 & 指标
# ==========================
def ordinal_targets(severity: torch.Tensor, num_thresholds: int) -> torch.Tensor:
    """
    将严重度标签 y \in {0,1,2,3} 转为 K 维二值向量：
    y_k = 1(y >= k+1), k = 0..K-1
    """
    # severity: [B]
    # out: [B, K]
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
    # logits -> sigmoid -> p_k = P(y>=k+1)
    loss = F.binary_cross_entropy_with_logits(logits_sev, targets, reduction="mean")
    return loss


def severity_from_logits(logits_sev: torch.Tensor) -> torch.Tensor:
    """
    根据多个阈值的 logit 还原严重度预测：
    \hat{y} = sum( I(p_k > 0.5) )
    """
    probs = torch.sigmoid(logits_sev)   # [B,K]
    preds_bin = (probs > 0.5).long()
    severity_pred = preds_bin.sum(dim=1)  # [B]
    return severity_pred


# ==========================
# 训练 & 验证
# ==========================
def train_one_epoch(
    epoch: int,
    model: HAPNetSG,
    optimizer,
    scheduler,
    train_loader,
    train_tf,
):
    model.train()
    running_loss = 0.0
    running_loss_sev = 0.0
    running_loss_dis = 0.0
    running_acc_sev = 0.0
    n_samples = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", ncols=120)
    for imgs_pil, disease, severity in pbar:
        disease = disease.to(DEVICE, non_blocking=True)
        severity = severity.to(DEVICE, non_blocking=True)
        bsz = severity.size(0)

        imgs = torch.stack([train_tf(img) for img in imgs_pil]).to(DEVICE)

        logits_dis, logits_sev, feat = model(imgs)

        # 严重度有序 loss
        loss_sev = ordinal_loss(logits_sev, severity)

        # 病害辅助 CrossEntropy
        loss_dis = F.cross_entropy(logits_dis, disease)

        loss = loss_sev + LAMBDA_DIS * loss_dis

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # 统计
        running_loss += loss.item() * bsz
        running_loss_sev += loss_sev.item() * bsz
        running_loss_dis += loss_dis.item() * bsz
        n_samples += bsz

        # 训练时给个大概的严重度 acc
        with torch.no_grad():
            sev_pred = severity_from_logits(logits_sev)
            acc_sev = (sev_pred == severity).float().mean().item()
            running_acc_sev += acc_sev * bsz

        avg_loss = running_loss / n_samples
        avg_acc = running_acc_sev / n_samples
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc_sev=f"{avg_acc:.4f}")

    epoch_loss = running_loss / n_samples
    epoch_loss_sev = running_loss_sev / n_samples
    epoch_loss_dis = running_loss_dis / n_samples
    epoch_acc_sev = running_acc_sev / n_samples

    return epoch_loss, epoch_loss_sev, epoch_loss_dis, epoch_acc_sev


@torch.no_grad()
def validate(
    epoch: int,
    model: HAPNetSG,
    val_loader,
    val_tf,
):
    model.eval()

    all_sev_true = []
    all_sev_pred = []

    running_loss = 0.0
    n_samples = 0

    pbar = tqdm(val_loader, desc=f"Val   Epoch {epoch}", ncols=120)
    for imgs_pil, disease, severity in pbar:
        disease = disease.to(DEVICE, non_blocking=True)
        severity = severity.to(DEVICE, non_blocking=True)
        bsz = severity.size(0)

        imgs = torch.stack([val_tf(img) for img in imgs_pil]).to(DEVICE)

        logits_dis, logits_sev, feat = model(imgs)

        loss_sev = ordinal_loss(logits_sev, severity)
        loss_dis = F.cross_entropy(logits_dis, disease)
        loss = loss_sev + LAMBDA_DIS * loss_dis

        running_loss += loss.item() * bsz
        n_samples += bsz

        sev_pred = severity_from_logits(logits_sev)

        all_sev_true.append(severity.cpu().numpy())
        all_sev_pred.append(sev_pred.cpu().numpy())

    epoch_loss = running_loss / n_samples

    all_sev_true = np.concatenate(all_sev_true, axis=0)
    all_sev_pred = np.concatenate(all_sev_pred, axis=0)

    acc = accuracy_score(all_sev_true, all_sev_pred)
    macro_f1 = f1_score(all_sev_true, all_sev_pred, average="macro")

    # 每类 recall
    report = classification_report(
        all_sev_true, all_sev_pred,
        labels=list(range(NUM_SEVERITY_CLASSES)),
        output_dict=True,
        zero_division=0
    )
    per_class_recall = [report[str(c)]["recall"] for c in range(NUM_SEVERITY_CLASSES)]

    print(
        f"Val   Epoch [{epoch}] Loss: {epoch_loss:.4f}, "
        f"Acc: {acc:.4f}, Macro-F1: {macro_f1:.4f}, "
        f"Recall per class: {per_class_recall}"
    )

    return epoch_loss, acc, macro_f1, per_class_recall, all_sev_true, all_sev_pred


# ==========================
# Grad-CAM 实现（针对严重度 logit）
# ==========================
class GradCAM:
    def __init__(self, model: HAPNetSG, target_layer_name: str = "backbone.conv_head"):
        self.model = model
        self.model.eval()

        # 通过名字获取 target layer
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

    def generate(self, img_tensor: torch.Tensor, target_sev_class: int):
        """
        img_tensor: [1,3,H,W]
        target_sev_class: 0/1/2/3，对应严重度
        """
        self.model.zero_grad()
        logits_dis, logits_sev, feat = self.model(img_tensor)

        # 将 ordinal 输出转成对应类的“logit”，这里简单取所有阈值 logit 之和作为 severity 相关 score
        # 或者：只对最高级别的阈值进行 Grad-CAM，这里采用 sum 方式
        sev_score = logits_sev.sum()

        sev_score.backward(retain_graph=True)

        # activations: [1,C,H',W'], gradients: [1,C,H',W']
        act = self.activations  # [1,C,H,W]
        grad = self.gradients   # [1,C,H,W]

        weights = grad.mean(dim=(2, 3), keepdim=True)  # [1,C,1,1]
        cam = (weights * act).sum(dim=1, keepdim=True) # [1,1,H,W]
        cam = F.relu(cam)

        cam = cam.squeeze(0).squeeze(0)  # [H,W]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def visualize_gradcam(
    model: HAPNetSG,
    dataset: SeverityDataset,
    transform,
    save_dir: str,
    num_samples_per_class: int = 3,
):
    os.makedirs(save_dir, exist_ok=True)
    cam_generator = GradCAM(model, target_layer_name="backbone.conv_head")

    # 为每个 severity 类随机找若干样本
    indices_by_sev = {c: [] for c in range(NUM_SEVERITY_CLASSES)}
    for idx, (_, _, sev) in enumerate(dataset.samples):
        if len(indices_by_sev[sev]) < num_samples_per_class:
            indices_by_sev[sev].append(idx)
        if all(len(v) >= num_samples_per_class for v in indices_by_sev.values()):
            break

    for sev, idx_list in indices_by_sev.items():
        for i, idx in enumerate(idx_list):
            img_name, disease, severity = dataset.samples[idx]
            img_path = os.path.join(dataset.img_dir, img_name)
            img_pil = Image.open(img_path).convert("RGB")

            img_t = transform(img_pil).unsqueeze(0).to(DEVICE)
            cam = cam_generator.generate(img_t, target_sev_class=sev)

            # resize cam 到原图大小
            cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(img_pil.size, resample=Image.BILINEAR)
            cam_np = np.array(cam_img) / 255.0

            # 叠加热力图
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(img_pil)
            plt.axis("off")
            plt.title(f"Sev={severity}")

            plt.subplot(1, 2, 2)
            plt.imshow(img_pil)
            plt.imshow(cam_np, cmap="jet", alpha=0.4)
            plt.axis("off")
            plt.title("Grad-CAM")

            fname = f"sev{sev}_idx{idx}_sample{i}.png"
            save_path = os.path.join(save_dir, fname)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"[Grad-CAM] Saved: {save_path}")


# ==========================
# 主函数
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
    train_anns = anns[val_len:]
    val_anns = anns[:val_len]

    train_dataset = SeverityDataset(IMG_DIR, train_anns)
    val_dataset = SeverityDataset(IMG_DIR, val_anns)
    # 为 Grad-CAM 使用
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

    # 2. 构建模型
    model = HAPNetSG(
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
    best_val_f1 = 0.0
    history = []

    csv_path = os.path.join(out_dir, "training_history_task3.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,train_loss_sev,train_loss_dis,train_acc_sev,val_loss,val_acc,val_macro_f1\n")

    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_loss_sev, train_loss_dis, train_acc_sev = train_one_epoch(
            epoch, model, optimizer, scheduler, train_loader, train_tf
        )

        val_loss, val_acc, val_macro_f1, per_class_recall, sev_true, sev_pred = validate(
            epoch, model, val_loader, val_tf
        )

        epoch_time = time.time() - start_time

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_path = os.path.join(out_dir, "best_model_task3.pth")
            torch.save(model.state_dict(), best_path)
            print(f"*** New best model (task3) at epoch {epoch}, Macro-F1={best_val_f1:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_loss_sev": float(train_loss_sev),
            "train_loss_dis": float(train_loss_dis),
            "train_acc_sev": float(train_acc_sev),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
        })

        # 写 CSV
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_loss_sev:.6f},{train_loss_dis:.6f},"
                f"{train_acc_sev:.6f},{val_loss:.6f},{val_acc:.6f},{val_macro_f1:.6f}\n"
            )

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"TrainLoss: {train_loss:.4f}, TrainAccSev: {train_acc_sev:.4f}, "
            f"ValLoss: {val_loss:.4f}, ValAcc: {val_acc:.4f}, MacroF1: {val_macro_f1:.4f}, "
            f"Time: {epoch_time:.1f}s"
        )

    # 保存 history JSON
    with open(os.path.join(out_dir, "training_history_task3.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"Training finished. Best Val Macro-F1 = {best_val_f1:.4f}")

    # 5. 保存最终一次验证的详细指标（包括每类 recall）
    report = classification_report(
        sev_true, sev_pred,
        labels=list(range(NUM_SEVERITY_CLASSES)),
        target_names=[f"severity_{i}" for i in range(NUM_SEVERITY_CLASSES)],
        output_dict=True,
        zero_division=0
    )
    with open(os.path.join(out_dir, "classification_report_task3.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 混淆矩阵
    cm = confusion_matrix(sev_true, sev_pred, labels=list(range(NUM_SEVERITY_CLASSES)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm.astype(np.int32),
        annot=True, fmt="d", cmap="Blues",
        xticklabels=[f"S{i}" for i in range(NUM_SEVERITY_CLASSES)],
        yticklabels=[f"S{i}" for i in range(NUM_SEVERITY_CLASSES)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Severity Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "cm_severity.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # 6. Grad-CAM 可视化（从训练集/验证集中各取若干样本）
    cam_dir = os.path.join(out_dir, "gradcam_samples")
    visualize_gradcam(model, train_dataset, val_tf, cam_dir, num_samples_per_class=3)


if __name__ == "__main__":
    main()
