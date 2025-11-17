# -*- coding: utf-8 -*-
"""
Task 3: HAPNet-SG 评估与科研级可视化脚本

功能：
1. 加载 hapnet_sg_outputs/best_model_task3.pth
2. 在划分出的验证集上计算：
   - 严重度 Top1 Accuracy
   - Macro-F1
   - 每类 Recall
3. 生成以下可视化（保存到 hapnet_sg_outputs/eval/）：
   - confusion matrix（严重度 4 类）
   - Grad-CAM 可视化（每个严重度若干样本）
   - 概率校准曲线 + ECE（reliability diagram）
   - t-SNE 特征分布（按严重度着色）
   - 误差距离直方图（|y_pred - y_true|）
   - 真实 vs 预测严重度分布对比图

注意：
- 需要与训练脚本（问题三那份）使用相同的 HAPNetSG 结构、map_disease_to_severity 规则。
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

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.manifold import TSNE

# ==========================
# 科研配色
# ==========================
COLORS = ['#90D8A6', '#83A1E7', '#E992A9', '#D2CAF8',
          '#F7AF7F', '#B0D9F9', '#E7B6BC', '#B0CDED']
sns.set_palette(COLORS)
plt.rcParams['font.family'] = ['DejaVu Sans']

# ==========================
# 路径 & 配置（与训练脚本保持一致）
# ==========================
BASE_DIR = r"D:\PyCharmproject\2025ShuWeiCup\Data"
IMG_DIR = os.path.join(BASE_DIR, "AgriculturalDisease_trainingset", "images")
ANN_JSON = os.path.join(BASE_DIR, "AgriculturalDisease_trainingset", "AgriculturalDisease_train_annotations_fixed.json")

OUTPUT_ROOT = "./hapnet_sg_outputs"
EVAL_DIR = os.path.join(OUTPUT_ROOT, "eval")

VAL_RATIO = 0.1

NUM_DISEASE_CLASSES = 61
NUM_SEVERITY_CLASSES = 4
NUM_THRESHOLDS = NUM_SEVERITY_CLASSES - 1

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2025
BACKBONE_NAME = "tf_efficientnet_b0_ns"


# ==========================
# 工具函数 & 随机种子
# ==========================
def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ==========================
# disease_class -> severity 映射
# （必须与训练脚本完全一致）
# ==========================
def map_disease_to_severity(disease_id: int) -> int:
    """
    占位示例：
    0: healthy
    1-20: mild
    21-40: moderate
    41-60: severe

    ⚠️ 请根据实际 JSON 标注规则同步修改训练和评估两边的逻辑。
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
# 数据集 & Transform
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
# HAPNet-SG 模型（与训练脚本一致）
# ==========================
class HAPNetSG(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_disease: int,
        num_severity: int,
        num_thresholds: int,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,      # 评估阶段不需要再下预训练
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
        feat = self.backbone(x)          # [B, D]
        logits_dis = self.head_disease(feat)
        logits_sev = self.head_severity(feat)  # [B, K]
        return logits_dis, logits_sev, feat


# ==========================
# 有序回归：logits -> severity_pred & prob 分布
# ==========================
def severity_from_logits(logits_sev: torch.Tensor) -> torch.Tensor:
    """
    根据多个阈值的 logit 还原严重度预测：
    \hat{y} = sum( I(sigmoid(logit_k) > 0.5) )
    """
    probs = torch.sigmoid(logits_sev)   # [B,K]
    preds_bin = (probs > 0.5).long()
    severity_pred = preds_bin.sum(dim=1)  # [B]
    return severity_pred


def severity_probs_from_logits(logits_sev: torch.Tensor) -> torch.Tensor:
    """
    将 ordinal 阈值概率转换为 4 类分类概率：
    p_k = P(y >= k+1)
    P(y=0) = 1 - p1
    P(y=1) = p1 - p2
    P(y=2) = p2 - p3
    P(y=3) = p3
    再做一次 clamp + 归一化，保证数值稳定。
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
# 评估：收集预测结果 & 特征
# ==========================
@torch.no_grad()
def evaluate(model: HAPNetSG, val_loader, val_tf):
    model.eval()
    all_sev_true = []
    all_sev_pred = []
    all_probs = []
    all_feats = []

    pbar = tqdm(val_loader, desc="Eval", ncols=120)
    for imgs_pil, disease, severity in pbar:
        severity = severity.to(DEVICE, non_blocking=True)
        imgs = torch.stack([val_tf(img) for img in imgs_pil]).to(DEVICE)

        logits_dis, logits_sev, feat = model(imgs)

        sev_pred = severity_from_logits(logits_sev)
        sev_probs = severity_probs_from_logits(logits_sev)

        all_sev_true.append(severity.cpu().numpy())
        all_sev_pred.append(sev_pred.cpu().numpy())
        all_probs.append(sev_probs.cpu().numpy())
        all_feats.append(feat.cpu().numpy())

    all_sev_true = np.concatenate(all_sev_true, axis=0)
    all_sev_pred = np.concatenate(all_sev_pred, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_feats = np.concatenate(all_feats, axis=0)

    acc = accuracy_score(all_sev_true, all_sev_pred)
    macro_f1 = f1_score(all_sev_true, all_sev_pred, average="macro")

    print(f"Overall Accuracy: {acc:.4f}, Macro-F1: {macro_f1:.4f}")

    return all_sev_true, all_sev_pred, all_probs, all_feats, acc, macro_f1


# ==========================
# 可视化：混淆矩阵 & 分布 &误差
# ==========================
def plot_confusion_matrix(cm, save_path, normalize=False):
    if normalize:
        cm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)
        fmt = ".2f"
        title = "Normalized Severity Confusion Matrix"
    else:
        fmt = "d"
        title = "Severity Confusion Matrix"

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True, fmt=fmt, cmap="Blues",
        xticklabels=[f"S{i}" for i in range(NUM_SEVERITY_CLASSES)],
        yticklabels=[f"S{i}" for i in range(NUM_SEVERITY_CLASSES)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_severity_distribution(y_true, y_pred, save_path):
    """
    真实 vs 预测严重度的分布对比图
    """
    vals = np.arange(NUM_SEVERITY_CLASSES)
    true_counts = np.array([(y_true == v).sum() for v in vals], dtype=np.float32)
    pred_counts = np.array([(y_pred == v).sum() for v in vals], dtype=np.float32)

    true_ratio = true_counts / (true_counts.sum() + 1e-8)
    pred_ratio = pred_counts / (pred_counts.sum() + 1e-8)

    df = pd.DataFrame({
        "severity": [f"S{i}" for i in vals] * 2,
        "ratio": np.concatenate([true_ratio, pred_ratio]),
        "type": ["True"] * NUM_SEVERITY_CLASSES + ["Pred"] * NUM_SEVERITY_CLASSES,
    })

    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=df,
        x="severity",
        y="ratio",
        hue="type",
    )
    plt.ylabel("Proportion")
    plt.title("True vs Predicted Severity Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved severity distribution plot to {save_path}")


def plot_error_distance_hist(y_true, y_pred, save_path):
    """
    绝对误差 |y_pred - y_true| 的分布：
    0,1,2,3，体现“多数错误是否为邻级错误”
    """
    diff = np.abs(y_pred - y_true)
    plt.figure(figsize=(6, 4))
    sns.histplot(diff, bins=[-0.5, 0.5, 1.5, 2.5, 3.5], stat="probability", discrete=True)
    plt.xticks([0, 1, 2, 3])
    plt.xlabel("|y_pred - y_true|")
    plt.ylabel("Frequency")
    plt.title("Distribution of Severity Prediction Error Distance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved error distance histogram to {save_path}")


# ==========================
# 概率校准：ECE & Reliability Diagram
# ==========================
def compute_ece(probs, labels, n_bins=10):
    """
    multi-class 情况：用 max prob 作为置信度
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    max_conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_acc = []
    bin_conf = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (max_conf >= lo) & (max_conf < hi)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc_in_bin = (preds[in_bin] == labels[in_bin]).mean()
            conf_in_bin = max_conf[in_bin].mean()
            ece += prop_in_bin * abs(acc_in_bin - conf_in_bin)
            bin_acc.append(acc_in_bin)
            bin_conf.append(conf_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_acc.append(0.0)
            bin_conf.append(0.0)
            bin_counts.append(0)
    return ece, bin_edges, np.array(bin_acc), np.array(bin_conf), np.array(bin_counts)


def plot_reliability_diagram(probs, labels, save_path, n_bins=10):
    ece, bin_edges, bin_acc, bin_conf, bin_counts = compute_ece(probs, labels, n_bins=n_bins)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    non_empty = bin_counts > 0
    centers = centers[non_empty]
    bin_acc = bin_acc[non_empty]
    bin_conf = bin_conf[non_empty]

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="#666666", label="Perfect Calibration")
    plt.plot(bin_conf, bin_acc, marker="o", linewidth=2, label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram (ECE={ece:.3f})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved reliability diagram to {save_path}")


# ==========================
# t-SNE 特征可视化
# ==========================
def plot_tsne(feats, labels, save_path, max_samples=2000, random_state=2025):
    n = feats.shape[0]
    if n > max_samples:
        idx = np.random.RandomState(random_state).choice(n, size=max_samples, replace=False)
        feats_sub = feats[idx]
        labels_sub = labels[idx]
    else:
        feats_sub = feats
        labels_sub = labels

    print(f"[t-SNE] using {feats_sub.shape[0]} samples")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        random_state=random_state,
        init="pca",
    )
    emb = tsne.fit_transform(feats_sub)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        emb[:, 0],
        emb[:, 1],
        c=labels_sub,
        cmap="viridis",
        alpha=0.7,
        s=10,
        linewidths=0,
    )
    cbar = plt.colorbar(scatter, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["S0", "S1", "S2", "S3"])
    plt.title("t-SNE of Severity Feature Embeddings")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")


# ==========================
# Grad-CAM
# ==========================
class GradCAM:
    def __init__(self, model: HAPNetSG, target_layer_name: str = "backbone.conv_head"):
        self.model = model
        self.model.eval()

        # 按路径获取层
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

    def generate(self, img_tensor: torch.Tensor):
        """
        img_tensor: [1,3,H,W]
        对所有阈值 logit 求和作为 score，生成 CAM。
        """
        self.model.zero_grad()
        logits_dis, logits_sev, feat = self.model(img_tensor)
        sev_score = logits_sev.sum()
        sev_score.backward(retain_graph=True)

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
    ensure_dir(save_dir)
    cam_generator = GradCAM(model, target_layer_name="backbone.conv_head")

    # 为每个 severity 类随机抽样
    indices_by_sev = {c: [] for c in range(NUM_SEVERITY_CLASSES)}
    for idx, (_, _, sev) in enumerate(dataset.samples):
        sev = int(sev)
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
            cam = cam_generator.generate(img_t)

            cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(img_pil.size, resample=Image.BILINEAR)
            cam_np = np.array(cam_img) / 255.0

            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(img_pil)
            plt.axis("off")
            plt.title(f"Severity={severity}")

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
    ensure_dir(EVAL_DIR)

    print(f"Use device: {DEVICE}")
    print(f"Eval dir: {EVAL_DIR}")

    # 1. 读取 JSON，并按 VAL_RATIO 划分
    with open(ANN_JSON, "r", encoding="utf-8") as f:
        anns = json.load(f)

    random.shuffle(anns)
    total = len(anns)
    val_len = int(total * VAL_RATIO)
    val_anns = anns[:val_len]

    val_dataset = SeverityDataset(IMG_DIR, val_anns)
    val_dataset.img_dir = IMG_DIR

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

    # 2. 构建模型并加载 Task3 最优权重
    model = HAPNetSG(
        backbone_name=BACKBONE_NAME,
        num_disease=NUM_DISEASE_CLASSES,
        num_severity=NUM_SEVERITY_CLASSES,
        num_thresholds=NUM_THRESHOLDS,
    ).to(DEVICE)

    best_model_path = os.path.join(OUTPUT_ROOT, "best_model_task3.pth")
    assert os.path.exists(best_model_path), f"{best_model_path} not found!"
    state = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    print(f"Loaded best Task3 model from {best_model_path}")

    # 3. 评估：拿到 y_true, y_pred, probs, feats
    y_true, y_pred, probs, feats, acc, macro_f1 = evaluate(model, val_loader, val_tf)

    # 4. 混淆矩阵 & 分布 & 误差图
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_SEVERITY_CLASSES)))
    plot_confusion_matrix(cm, os.path.join(EVAL_DIR, "cm_severity_raw.png"), normalize=False)
    plot_confusion_matrix(cm, os.path.join(EVAL_DIR, "cm_severity_norm.png"), normalize=True)
    plot_severity_distribution(y_true, y_pred, os.path.join(EVAL_DIR, "severity_distribution.png"))
    plot_error_distance_hist(y_true, y_pred, os.path.join(EVAL_DIR, "severity_error_distance.png"))

    # 5. 概率校准图
    plot_reliability_diagram(probs, y_true, os.path.join(EVAL_DIR, "reliability_severity.png"), n_bins=10)

    # 6. t-SNE 特征图
    plot_tsne(feats, y_true, os.path.join(EVAL_DIR, "tsne_severity_features.png"), max_samples=2000)

    # 7. Grad-CAM 可视化
    cam_dir = os.path.join(EVAL_DIR, "gradcam_samples")
    visualize_gradcam(model, val_dataset, val_tf, cam_dir, num_samples_per_class=3)

    # 8. 保存详细分类报告（方便写论文/查指标）
    report = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_SEVERITY_CLASSES)),
        target_names=[f"severity_{i}" for i in range(NUM_SEVERITY_CLASSES)],
        output_dict=True,
        zero_division=0
    )
    with open(os.path.join(EVAL_DIR, "classification_report_task3_eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Task3 evaluation & visualization finished.")


if __name__ == "__main__":
    main()
