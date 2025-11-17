# -*- coding: utf-8 -*-
"""
HAPNet 评估与可视化脚本（问题二）

功能：
1. 加载 hapnet_outputs/best_model.pth
2. 在验证集上计算：
   - Top1 / Top5 准确率
   - 每类 Precision / Recall / F1 / Accuracy
3. 绘制并保存：
   - 混淆矩阵（全 61 类，原始 & 归一化）
   - 每类准确率条形图（按准确率排序）
   - 训练过程 Loss / Acc 曲线（从 training_history.csv 读取）
   - t-SNE 特征可视化（高维特征降到 2D）
   - 置信度直方图（正确 vs 错误）
   - 可靠性图（校准曲线 + ECE）
   - 最难类别子矩阵混淆图（Top-K 准确率最低的类）

输出目录：hapnet_outputs/eval
"""

import os
import json
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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE

# --------------------------
# 科研配色
# --------------------------
COLORS = ['#90D8A6', '#83A1E7', '#E992A9', '#D2CAF8',
          '#F7AF7F', '#B0D9F9', '#E7B6BC', '#B0CDED']
sns.set_palette(COLORS)
plt.rcParams['font.family'] = ['DejaVu Sans']  # 如有中文字体需求可改成你本地字体

# --------------------------
# 路径 & 全局配置
# --------------------------
TRAIN_IMG_DIR = r"D:\PyCharmproject\2025ShuWeiCup\Data\AgriculturalDisease_trainingset\images"
TRAIN_ANN_PATH = r"D:\PyCharmproject\2025ShuWeiCup\Data\AgriculturalDisease_trainingset\AgriculturalDisease_train_annotations_fixed.json"

VAL_IMG_DIR = r"D:\PyCharmproject\2025ShuWeiCup\Data\AgriculturalDisease_validationset\images"
VAL_LIST_PATH = r"D:\PyCharmproject\2025ShuWeiCup\Data\AgriculturalDisease_validationset\ttest_list.txt"

OUTPUT_ROOT = "./hapnet_outputs"
EVAL_DIR = os.path.join(OUTPUT_ROOT, "eval")

NUM_CLASSES = 61
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SPECIES = 10
NUM_DISEASE = 27


# --------------------------
# 工具函数
# --------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# --------------------------
# 数据集（验证）
# --------------------------
class LeafValDataset(Dataset):
    def __init__(self, img_dir: str, list_path: str):
        super().__init__()
        self.img_dir = img_dir
        self.samples = []
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                img_name = os.path.basename(parts[0])
                label = int(parts[-1])
                self.samples.append((img_name, label))
        print(f"[ValDataset] {len(self.samples)} samples from {list_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        return img, label


def pil_collate_fn(batch):
    imgs, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(imgs), labels


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


# --------------------------
# HAPNet 模型（与训练脚本一致）
# --------------------------
class HAPNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_species: int,
        num_disease: int,
        backbone_name: str = "tf_efficientnet_b0_ns",
        pretrained: bool = False,     # 评估时不需要再下预训练
        proto_dim: int = 256,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        if hasattr(self.backbone, "num_features"):
            in_dim = self.backbone.num_features
        else:
            in_dim = 1280

        self.feat_proj = nn.Linear(in_dim, proto_dim)

        self.species_proto = nn.Parameter(torch.randn(num_species, proto_dim))
        self.disease_proto = nn.Parameter(torch.randn(num_disease, proto_dim))
        self.class_proto = nn.Parameter(torch.randn(num_classes, proto_dim))

        nn.init.normal_(self.species_proto, std=0.02)
        nn.init.normal_(self.disease_proto, std=0.02)
        nn.init.normal_(self.class_proto, std=0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        z = self.feat_proj(feat)
        z = F.normalize(z, dim=-1)
        return z

    def proto_logits(self, z: torch.Tensor, level: str):
        if level == "species":
            proto = self.species_proto
        elif level == "disease":
            proto = self.disease_proto
        elif level == "fine":
            proto = self.class_proto
        else:
            raise ValueError(f"Unknown level {level}")
        logits = -torch.cdist(z.unsqueeze(1), proto.unsqueeze(0), p=2).squeeze(1)
        return logits

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        logits = self.proto_logits(z, level="fine")
        return logits, z


# --------------------------
# 评估：得到预测 & 特征
# --------------------------
def evaluate_model(model, val_loader, val_tf):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_feats = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Eval", ncols=120)
        for imgs_pil, labels in pbar:
            labels = labels.to(DEVICE, non_blocking=True)
            imgs = torch.stack([val_tf(img) for img in imgs_pil]).to(DEVICE)

            logits, feats = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_feats.append(feats.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_feats = np.concatenate(all_feats, axis=0)

    # Top1 / Top5
    top1_acc = (all_preds == all_labels).mean()

    top5_correct = 0
    batch_size = 256
    for i in range(0, len(all_labels), batch_size):
        probs_batch = torch.from_numpy(all_probs[i:i+batch_size])
        labels_batch = torch.from_numpy(all_labels[i:i+batch_size])
        _, top5 = probs_batch.topk(5, dim=1)
        match = top5.eq(labels_batch.view(-1, 1))
        top5_correct += match.any(dim=1).float().sum().item()
    top5_acc = top5_correct / len(all_labels)

    print(f"Overall Top1 Acc: {top1_acc:.4f}, Top5 Acc: {top5_acc:.4f}")

    return all_labels, all_preds, all_probs, all_feats, top1_acc, top5_acc


# --------------------------
# 指标表 & 基础图
# --------------------------
def plot_confusion_matrix(cm, save_path, normalize=True):
    if normalize:
        cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)
        data = cm_norm
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        data = cm
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        data,
        cmap="Blues",
        xticklabels=False,
        yticklabels=False,
        square=True,
        cbar=True,
        linewidths=0.0,
        linecolor=None,
        fmt=fmt,
    )
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_per_class_accuracy(per_class_acc, save_path):
    cls_ids = np.arange(len(per_class_acc))
    order = np.argsort(per_class_acc)  # 从低到高

    plt.figure(figsize=(14, 6))
    sns.barplot(
        x=np.arange(len(per_class_acc)),
        y=per_class_acc[order],
        palette=COLORS * ((len(per_class_acc) // len(COLORS)) + 1),
    )
    plt.xlabel("Class (sorted by accuracy)")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy (sorted)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved per-class accuracy bar plot to {save_path}")


def plot_training_curves(history_csv_path, save_dir):
    if not os.path.exists(history_csv_path):
        print(f"[Warn] training_history.csv not found at {history_csv_path}, skip curves.")
        return

    df = pd.read_csv(history_csv_path)
    epochs = df["epoch"].values

    # Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_loss"], label="Train Loss")
    plt.plot(epochs, df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "curve_loss.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved loss curve to {save_path}")

    # Acc 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_acc1"], label="Train Top1")
    plt.plot(epochs, df["val_acc1"], label="Val Top1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Top1 Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "curve_acc.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved accuracy curve to {save_path}")


def build_metrics_table(labels, preds, save_dir):
    cm = confusion_matrix(labels, preds, labels=np.arange(NUM_CLASSES))
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)

    report = classification_report(
        labels,
        preds,
        labels=np.arange(NUM_CLASSES),
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for cls in range(NUM_CLASSES):
        cls_report = report[str(cls)]
        rows.append({
            "class_id": cls,
            "support": int(cls_report["support"]),
            "precision": float(cls_report["precision"]),
            "recall": float(cls_report["recall"]),
            "f1": float(cls_report["f1-score"]),
            "accuracy": float(per_class_acc[cls]),
        })
    df = pd.DataFrame(rows)
    metrics_csv = os.path.join(save_dir, "per_class_metrics.csv")
    df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print(f"Saved per-class metrics table to {metrics_csv}")

    overall = {
        "overall_precision_macro": float(report["macro avg"]["precision"]),
        "overall_recall_macro": float(report["macro avg"]["recall"]),
        "overall_f1_macro": float(report["macro avg"]["f1-score"]),
        "overall_precision_weighted": float(report["weighted avg"]["precision"]),
        "overall_recall_weighted": float(report["weighted avg"]["recall"]),
        "overall_f1_weighted": float(report["weighted avg"]["f1-score"]),
    }
    overall_json = os.path.join(save_dir, "overall_metrics.json")
    with open(overall_json, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    print(f"Saved overall metrics to {overall_json}")

    return cm, per_class_acc, df, overall


# --------------------------
# 高级可视化
# --------------------------
def plot_tsne(feats, labels, save_path, max_samples=2000, random_state=2025):
    """
    feats: [N, D]
    labels: [N]
    随机采样 max_samples 个点做 t-SNE 可视化。
    """
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

    plt.figure(figsize=(8, 8))
    # 61 类太多，这里用连续 colormap，点加一点透明度
    scatter = plt.scatter(
        emb[:, 0],
        emb[:, 1],
        c=labels_sub,
        cmap="tab20",
        alpha=0.6,
        s=10,
        linewidths=0,
    )
    plt.title("t-SNE of Feature Embeddings")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")


def plot_confidence_hist(probs, labels, preds, save_path, bins=20):
    """
    probs: [N, C] softmax 输出
    labels, preds: [N]
    绘制正确与错误样本的 max prob 分布
    """
    max_conf = probs.max(axis=1)
    correct = (preds == labels)
    conf_correct = max_conf[correct]
    conf_wrong = max_conf[~correct]

    plt.figure(figsize=(8, 5))
    sns.histplot(conf_correct, bins=bins, stat="density", kde=True, label="Correct", alpha=0.6)
    sns.histplot(conf_wrong, bins=bins, stat="density", kde=True, label="Incorrect", alpha=0.6)
    plt.xlabel("Max Softmax Probability")
    plt.ylabel("Density")
    plt.title("Confidence Distribution (Correct vs Incorrect)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confidence histogram to {save_path}")


def compute_ece(probs, labels, n_bins=10):
    """
    计算 Expected Calibration Error (ECE)
    probs: [N, C]
    labels: [N]
    """
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

    # 只画非空 bin
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    non_empty = bin_counts > 0
    centers = centers[non_empty]
    bin_acc = bin_acc[non_empty]
    bin_conf = bin_conf[non_empty]

    plt.figure(figsize=(6, 6))
    # 参考线
    plt.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Perfect Calibration")

    # 实际点
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


def plot_hard_classes_confusion(cm, per_class_acc, save_path, top_k=15):
    """
    选取 top_k 个最难类别（准确率最低），画一个小的归一化混淆矩阵。
    """
    if top_k > NUM_CLASSES:
        top_k = NUM_CLASSES

    order = np.argsort(per_class_acc)  # 从低到高
    hard_ids = order[:top_k]

    cm_sub = cm[hard_ids][:, hard_ids]
    cm_sub_norm = cm_sub.astype(np.float32) / (cm_sub.sum(axis=1, keepdims=True) + 1e-6)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_sub_norm,
        cmap="Blues",
        annot=False,
        xticklabels=hard_ids,
        yticklabels=hard_ids,
        square=True,
        cbar=True,
        linewidths=0.2,
        linecolor="white",
    )
    plt.xlabel("Predicted Class ID")
    plt.ylabel("True Class ID")
    plt.title(f"Confusion Matrix of {top_k} Hardest Classes")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved hard-classes confusion matrix to {save_path}")


# --------------------------
# 主函数
# --------------------------
def main():
    ensure_dir(EVAL_DIR)
    print(f"Use device: {DEVICE}")
    print(f"Eval dir: {EVAL_DIR}")

    # 1. 构建验证集 loader
    val_dataset = LeafValDataset(VAL_IMG_DIR, VAL_LIST_PATH)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=pil_collate_fn,
    )
    val_tf = get_val_transform(IMG_SIZE)

    # 2. 构建模型并加载 best_model 权重
    model = HAPNet(
        num_classes=NUM_CLASSES,
        num_species=NUM_SPECIES,
        num_disease=NUM_DISEASE,
        backbone_name="tf_efficientnet_b0_ns",
        pretrained=False,   # 评估不需要再加载预训练
        proto_dim=256,
    ).to(DEVICE)

    best_model_path = os.path.join(OUTPUT_ROOT, "best_model.pth")
    assert os.path.exists(best_model_path), f"{best_model_path} not found!"
    state = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    print(f"Loaded best model from {best_model_path}")

    # 3. 在验证集上跑一遍
    labels, preds, probs, feats, top1, top5 = evaluate_model(model, val_loader, val_tf)

    # 4. 指标表 + 混淆矩阵
    cm, per_class_acc, df_metrics, overall = build_metrics_table(labels, preds, EVAL_DIR)

    # 5. 基础图
    plot_confusion_matrix(cm, os.path.join(EVAL_DIR, "cm_norm.png"), normalize=True)
    plot_confusion_matrix(cm, os.path.join(EVAL_DIR, "cm_raw.png"), normalize=False)
    plot_per_class_accuracy(per_class_acc, os.path.join(EVAL_DIR, "per_class_acc.png"))
    history_csv_path = os.path.join(OUTPUT_ROOT, "training_history.csv")
    plot_training_curves(history_csv_path, EVAL_DIR)

    # 6. 高级可视化
    # 6.1 t-SNE 特征可视化
    plot_tsne(feats, labels, os.path.join(EVAL_DIR, "tsne_features.png"),
              max_samples=2000, random_state=2025)

    # 6.2 置信度分布（正确 vs 错误）
    plot_confidence_hist(probs, labels, preds, os.path.join(EVAL_DIR, "conf_hist.png"))

    # 6.3 可靠性图（校准曲线 + ECE）
    plot_reliability_diagram(probs, labels, os.path.join(EVAL_DIR, "reliability.png"), n_bins=10)

    # 6.4 最难类别子矩阵混淆图
    plot_hard_classes_confusion(cm, per_class_acc, os.path.join(EVAL_DIR, "cm_hard_classes.png"), top_k=15)

    print("Eval & visualization finished.")


if __name__ == "__main__":
    main()
