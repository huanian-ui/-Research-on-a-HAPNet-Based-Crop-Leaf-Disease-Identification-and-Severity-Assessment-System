# -*- coding: utf-8 -*-
"""
训练：Acc1 ≈ 0.995（在 10-shot 上属于“几乎记住了训练集”，正常）
验证：Best Val Acc1 = 0.7370（61 类、每类 10 张训练样本，这个水平已经挺不错）
"""

import os
import json
import random
import math
import time
import collections
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

# ==========================
# 路径配置
# ==========================
TRAIN_IMG_DIR = r"D:\PyCharmproject\2025ShuWeiCup\Data\AgriculturalDisease_trainingset\images"
TRAIN_ANN_PATH = r"D:\PyCharmproject\2025ShuWeiCup\Data\AgriculturalDisease_trainingset\AgriculturalDisease_train_annotations_fixed.json"

VAL_IMG_DIR = r"D:\PyCharmproject\2025ShuWeiCup\Data\AgriculturalDisease_validationset\images"
VAL_LIST_PATH = r"D:\PyCharmproject\2025ShuWeiCup\Data\AgriculturalDisease_validationset\ttest_list.txt"

OUTPUT_DIR = "./hapnet_outputs"

# ==========================
# 全局超参数
# ==========================
NUM_CLASSES = 61
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 80
NUM_WORKERS = 4

BASE_LR = 3e-4
WEIGHT_DECAY = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SPECIES = 10
NUM_DISEASE = 27

LAMBDA_SPECIES = 0.5
LAMBDA_DISEASE = 0.7
LAMBDA_FINE = 1.0

LAMBDA_SSL = 0.3
EMA_DECAY = 0.99

SEED = 2025


# ==========================
# 工具函数
# ==========================
def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_dir(base_dir: str) -> str:
    if os.path.isfile(base_dir):
        root, name = os.path.split(base_dir)
        if not root:
            root = "."
        new_dir = os.path.join(root, name + "_dir")
        print(f"[Warning] {base_dir} 是一个文件，改用 {new_dir} 作为输出目录。")
        base_dir = new_dir
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


# ==========================
# 数据集定义
# ==========================
class LeafTrainDataset(Dataset):
    def __init__(self, img_dir: str, ann_path: str):
        super().__init__()
        self.img_dir = img_dir
        with open(ann_path, "r", encoding="utf-8") as f:
            anns = json.load(f)
        self.samples = [
            (ann["image_id"], int(ann["disease_class"]))
            for ann in anns
        ]
        print(f"[TrainDataset] {len(self.samples)} samples from {ann_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        return img, label


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


# ==========================
# 自定义 collate_fn：保留 PIL 列表
# ==========================
def pil_collate_fn(batch):
    """
    batch: list of (PIL.Image, label)
    返回：
        imgs_pil: list[PIL.Image]
        labels: LongTensor
    """
    imgs, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(imgs), labels


# ==========================
# 10-shot 子集构建（关键约束）
# ==========================
def make_10shot_subset(dataset: LeafTrainDataset, num_classes=61, shots=10, seed=2025):
    """
    从完整训练集里，为每个类别随机选择 shots 张样本，
    直接修改 dataset.samples。

    确保严格符合题目：
    - 每类最多 10 张
    - 不引入额外数据
    """
    random.seed(seed)
    by_class = collections.defaultdict(list)

    for img_name, label in dataset.samples:
        by_class[label].append((img_name, label))

    new_samples = []
    for c in range(num_classes):
        imgs = by_class[c]
        if len(imgs) == 0:
            continue
        random.shuffle(imgs)
        new_samples.extend(imgs[:shots])

    print(f"[10-shot] Use {len(new_samples)} samples ({shots} per class).")
    dataset.samples = new_samples


# ==========================
# 层级映射（占位）
# ==========================
def build_hierarchy_mapping(num_classes: int, num_species: int, num_disease: int):
    species_ids = []
    disease_ids = []
    for c in range(num_classes):
        species_ids.append(c % num_species)
        disease_ids.append(c % num_disease)
    species_ids = np.array(species_ids, dtype=np.int64)
    disease_ids = np.array(disease_ids, dtype=np.int64)
    return species_ids, disease_ids


SPECIES_IDS, DISEASE_IDS = build_hierarchy_mapping(NUM_CLASSES, NUM_SPECIES, NUM_DISEASE)


# ==========================
# 数据增强 & LesionMix
# ==========================
def get_transforms(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    weak_tf = T.Compose([
        T.Resize(int(img_size * 1.1)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    strong_tf = T.Compose([
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

    return weak_tf, strong_tf, val_tf


def generate_lesion_mask(x: torch.Tensor, thresh_ratio: float = 0.6):
    with torch.no_grad():
        gray = x.mean(dim=1, keepdim=True)   # [B,1,H,W]
        b, _, h, w = gray.shape
        flat = gray.view(b, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True)
        std = torch.clamp(std, min=1e-6)

        thr = mean + thresh_ratio * std
        thr = thr.view(b, 1, 1, 1)
        mask = (gray > thr).float()

        for i in range(b):
            if mask[i].sum() < (h * w * 0.01):
                rh = int(h * 0.3)
                rw = int(w * 0.3)
                sy = random.randint(0, h - rh)
                sx = random.randint(0, w - rw)
                m = torch.zeros((1, h, w), device=x.device)
                m[:, sy:sy + rh, sx:sx + rw] = 1.0
                mask[i] = m
    return mask


def lesion_mix(x: torch.Tensor, y: torch.Tensor, p: float = 0.5):
    if random.random() > p:
        return x, y

    b, c, h, w = x.shape
    device = x.device
    mask = generate_lesion_mask(x)
    x_new = x.clone()

    unique_labels = y.unique()
    for cls in unique_labels:
        idxs = (y == cls).nonzero(as_tuple=False).view(-1)
        if idxs.numel() < 2:
            continue
        perm = idxs[torch.randperm(idxs.numel(), device=device)]
        for i in range(len(idxs)):
            src = idxs[i]
            tgt = perm[i]
            if src == tgt:
                continue
            m = mask[src]
            x_new[src] = x[src] * (1 - m) + x[tgt] * m

    return x_new, y


# ==========================
# HAPNet 模型
# ==========================
class HAPNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_species: int,
        num_disease: int,
        backbone_name: str = "tf_efficientnet_b0_ns",
        pretrained: bool = True,
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


# ==========================
# EMA Teacher
# ==========================
def create_ema_model(student: nn.Module):
    import copy
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, ema_decay: float = EMA_DECAY):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(ema_decay).add_(s_param.data, alpha=(1.0 - ema_decay))


# ==========================
# 损失 & 度量
# ==========================
def prototype_ce_loss(logits: torch.Tensor, targets: torch.Tensor):
    return F.cross_entropy(logits, targets)


def ssl_cosine_loss(z_s: torch.Tensor, z_t: torch.Tensor):
    z_s = F.normalize(z_s, dim=-1)
    z_t = F.normalize(z_t, dim=-1)
    return 1.0 - (z_s * z_t).sum(dim=-1).mean()


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


# ==========================
# 训练 & 验证
# ==========================
def train_one_epoch(
    epoch: int,
    model: HAPNet,
    teacher: HAPNet,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loader: DataLoader,
    weak_tf,
    strong_tf,
):
    model.train()
    teacher.eval()

    running_loss = 0.0
    running_proto_loss = 0.0
    running_ssl_loss = 0.0
    running_acc1 = 0.0
    running_acc5 = 0.0
    n_samples = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", ncols=120)
    for step, (imgs_pil, labels) in enumerate(pbar, start=1):
        labels = labels.to(DEVICE, non_blocking=True)
        bsz = labels.size(0)

        imgs_s = torch.stack([strong_tf(img) for img in imgs_pil]).to(DEVICE)
        imgs_s, labels_mix = lesion_mix(imgs_s, labels, p=0.7)

        imgs_t = torch.stack([weak_tf(img) for img in imgs_pil]).to(DEVICE)

        logits_fine, z_s = model(imgs_s)
        with torch.no_grad():
            _, z_t = teacher(imgs_t)

        species_targets = torch.from_numpy(SPECIES_IDS[labels.cpu().numpy()]).to(DEVICE)
        disease_targets = torch.from_numpy(DISEASE_IDS[labels.cpu().numpy()]).to(DEVICE)

        logits_species = model.proto_logits(z_s, "species")
        logits_disease = model.proto_logits(z_s, "disease")

        loss_species = prototype_ce_loss(logits_species, species_targets)
        loss_disease = prototype_ce_loss(logits_disease, disease_targets)
        loss_fine = prototype_ce_loss(logits_fine, labels_mix)

        proto_loss = (
            LAMBDA_SPECIES * loss_species
            + LAMBDA_DISEASE * loss_disease
            + LAMBDA_FINE * loss_fine
        )

        ssl_loss = ssl_cosine_loss(z_s, z_t)
        loss = proto_loss + LAMBDA_SSL * ssl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        update_ema(model, teacher, ema_decay=EMA_DECAY)

        acc1, acc5 = accuracy(logits_fine, labels, topk=(1, 5))

        running_loss += loss.item() * bsz
        running_proto_loss += proto_loss.item() * bsz
        running_ssl_loss += ssl_loss.item() * bsz
        running_acc1 += acc1.item() * bsz
        running_acc5 += acc5.item() * bsz
        n_samples += bsz

        avg_loss = running_loss / n_samples
        avg_acc1 = running_acc1 / n_samples
        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc1=f"{avg_acc1:.4f}",
        )

    epoch_loss = running_loss / n_samples
    epoch_proto_loss = running_proto_loss / n_samples
    epoch_ssl_loss = running_ssl_loss / n_samples
    epoch_acc1 = running_acc1 / n_samples
    epoch_acc5 = running_acc5 / n_samples

    return epoch_loss, epoch_proto_loss, epoch_ssl_loss, epoch_acc1, epoch_acc5


@torch.no_grad()
def validate(
    epoch: int,
    model: HAPNet,
    val_loader: DataLoader,
    val_tf,
):
    model.eval()

    running_loss = 0.0
    running_acc1 = 0.0
    running_acc5 = 0.0
    n_samples = 0

    pbar = tqdm(val_loader, desc=f"Val   Epoch {epoch}", ncols=120)
    for imgs_pil, labels in pbar:
        labels = labels.to(DEVICE, non_blocking=True)
        bsz = labels.size(0)

        imgs = torch.stack([val_tf(img) for img in imgs_pil]).to(DEVICE)

        logits_fine, _ = model(imgs)
        loss = F.cross_entropy(logits_fine, labels)

        acc1, acc5 = accuracy(logits_fine, labels, topk=(1, 5))

        running_loss += loss.item() * bsz
        running_acc1 += acc1.item() * bsz
        running_acc5 += acc5.item() * bsz
        n_samples += bsz

        avg_loss = running_loss / n_samples
        avg_acc1 = running_acc1 / n_samples
        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc1=f"{avg_acc1:.4f}",
        )

    epoch_loss = running_loss / n_samples
    epoch_acc1 = running_acc1 / n_samples
    epoch_acc5 = running_acc5 / n_samples

    return epoch_loss, epoch_acc1, epoch_acc5


# ==========================
# 主训练入口
# ==========================
def main():
    set_seed(SEED)

    out_dir = prepare_output_dir(OUTPUT_DIR)
    print(f"Use device: {DEVICE}")
    print(f"Output dir: {out_dir}")

    weak_tf, strong_tf, val_tf = get_transforms(IMG_SIZE)

    train_dataset = LeafTrainDataset(TRAIN_IMG_DIR, TRAIN_ANN_PATH)
    # ⭐ 严格变成 10-shot 训练集
    make_10shot_subset(train_dataset, num_classes=NUM_CLASSES, shots=10, seed=SEED)

    val_dataset = LeafValDataset(VAL_IMG_DIR, VAL_LIST_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=pil_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=pil_collate_fn,
    )

    model = HAPNet(
        num_classes=NUM_CLASSES,
        num_species=NUM_SPECIES,
        num_disease=NUM_DISEASE,
        backbone_name="tf_efficientnet_b0_ns",
        pretrained=True,
        proto_dim=256,
    ).to(DEVICE)

    teacher = create_ema_model(model).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = len(train_loader) * 5

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    history = []

    csv_path = os.path.join(out_dir, "training_history.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,train_proto_loss,train_ssl_loss,train_acc1,train_acc5,val_loss,val_acc1,val_acc5,epoch_time_sec\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_proto_loss, train_ssl_loss, train_acc1, train_acc5 = train_one_epoch(
            epoch, model, teacher, optimizer, scheduler, train_loader,
            weak_tf, strong_tf
        )

        val_loss, val_acc1, val_acc5 = validate(
            epoch, model, val_loader, val_tf
        )

        epoch_time = time.time() - start_time

        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            best_model_path = os.path.join(out_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"*** New best model at epoch {epoch}, val_acc1={best_val_acc:.4f}")

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_proto_loss": float(train_proto_loss),
            "train_ssl_loss": float(train_ssl_loss),
            "train_acc1": float(train_acc1),
            "train_acc5": float(train_acc5),
            "val_loss": float(val_loss),
            "val_acc1": float(val_acc1),
            "val_acc5": float(val_acc5),
            "epoch_time_sec": float(epoch_time),
        }
        history.append(record)

        json_path = os.path.join(out_dir, "training_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_proto_loss:.6f},{train_ssl_loss:.6f},"
                f"{train_acc1:.6f},{train_acc5:.6f},"
                f"{val_loss:.6f},{val_acc1:.6f},{val_acc5:.6f},{epoch_time:.3f}\n"
            )

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"TrainLoss: {train_loss:.4f}, TrainAcc1: {train_acc1:.4f}, "
            f"ValLoss: {val_loss:.4f}, ValAcc1: {val_acc1:.4f}, "
            f"Time: {epoch_time:.1f}s"
        )

    print(f"Training finished. Best Val Acc1 = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
