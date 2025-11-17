import os
import json

# ====== 按你当前的实际路径设置 ======
BASE_DIR = r"D:\PyCharmproject\2025ShuWeiCup\Data\AgriculturalDisease_trainingset"

IMG_DIR = os.path.join(BASE_DIR, "images")
TRAIN_LIST_PATH = os.path.join(BASE_DIR, "train_list.txt")

OUT_JSON_PATH = os.path.join(
    BASE_DIR, "AgriculturalDisease_train_annotations_fixed.json"
)


def build_name_map(img_dir):
    """
    扫描 images 目录，建立 “规范化名字 → 实际文件名” 的映射。
    规范化规则：
    - 去掉扩展名，例如 '1_0.jpg' -> '1_0'
    - 再按 ' - ' 切分，取前半部分，例如 '1_0 - 副本' -> '1_0'
    """
    files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    name_map = {}
    for fname in files:
        base = os.path.splitext(fname)[0]
        norm = base.split(' - ')[0]
        if norm not in name_map:
            name_map[norm] = fname

    print(f"[build_name_map] Found {len(files)} image files, {len(name_map)} unique normalized keys.")
    return name_map


def build_fixed_annotations(train_list_path, name_map, out_json_path):
    """
    使用 train_list.txt 中的 (相对路径, 标签)，结合 name_map，
    生成一份新的 JSON 标注文件，image_id 使用实际存在的文件名。
    train_list.txt 每行示例：
    AgriculturalDisease_trainingset/images\\1_0.jpg 1
    """
    fixed_anns = []
    total_lines = 0
    matched = 0
    missing = 0

    with open(train_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1

            try:
                path_str, label_str = line.split()
            except ValueError:
                print(f"[WARN] skip malformed line: {line}")
                continue

            label = int(label_str)

            # 取出文件名部分：1_0.jpg
            fname = os.path.basename(path_str)
            base = os.path.splitext(fname)[0]
            norm = base.split(' - ')[0]

            if norm in name_map:
                real_fname = name_map[norm]  # 比如 '1_0 - 副本.jpg' 或 '1_0.jpg'
                fixed_anns.append({
                    "disease_class": label,
                    "image_id": real_fname
                })
                matched += 1
            else:
                missing += 1

    print(f"[build_fixed_annotations] Total lines in train_list: {total_lines}")
    print(f"[build_fixed_annotations] Matched: {matched}")
    print(f"[build_fixed_annotations] Missing: {missing}")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(fixed_anns, f, ensure_ascii=False, indent=2)

    print(f"[build_fixed_annotations] Saved fixed annotations to: {out_json_path}")


def main():
    if not os.path.exists(IMG_DIR):
        print(f"[ERROR] IMG_DIR not found: {IMG_DIR}")
        return
    if not os.path.exists(TRAIN_LIST_PATH):
        print(f"[ERROR] TRAIN_LIST_PATH not found: {TRAIN_LIST_PATH}")
        return

    name_map = build_name_map(IMG_DIR)
    build_fixed_annotations(TRAIN_LIST_PATH, name_map, OUT_JSON_PATH)


if __name__ == "__main__":
    main()
