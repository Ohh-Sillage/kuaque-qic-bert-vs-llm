"""
从 PaddleNLP 镜像下载 KUAKE-QIC 医疗意图分类数据集并保存为本地 JSON 文件

教学重点：
  1. 数据集的获取与本地化存储方式（tar.gz 下载 + 解压）
  2. 类别映射（label_id ↔ 中文类别名）的建立
  3. 数据集划分结构（train / val / test）

使用方式：
  python download_data.py

依赖：
  仅 Python 标准库（urllib / tarfile），无需额外安装
"""

import json
import tarfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_URL = "https://paddlenlp.bj.bcebos.com/datasets/KUAKE_QIC.tar.gz"

def build_label_map(label_names: list) -> dict:
    """构建类别映射：label_id(int) ↔ 中文类别名(str)，标签顺序取自官方 label.txt"""
    id2name = {i: name for i, name in enumerate(label_names)}
    return {
        "id2name": id2name,
        "name2id": {v: k for k, v in id2name.items()},
        "num_labels": len(label_names),
    }


def download_dataset(url: str, dest_dir: Path) -> Path:
    """下载 tar.gz 并解压，返回解压后的目录"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "KUAKE_QIC.tar.gz"
    print(f"正在下载 {url} ...")
    urllib.request.urlretrieve(url, archive_path)
    print(f"下载完成 → {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir, filter="data")
    archive_path.unlink()  # 解压后删除压缩包
    return dest_dir


def find_file(extract_dir: Path, filename: str) -> Path:
    """在解压目录中递归定位指定文件"""
    matches = list(extract_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"解压目录中未找到 {filename}")
    return matches[0]


def load_txt_records(path: Path, label_map: dict):
    """读取官方 txt（每行 query\tlabel），转换为统一记录格式"""
    records = []
    for i, line in enumerate(path.read_text(encoding="utf-8").strip().splitlines()):
        parts = line.split("\t")
        query = parts[0].strip()
        label_name = parts[1].strip() if len(parts) > 1 else None
        label_id = label_map["name2id"].get(label_name, -1)  # -1 表示无标签
        records.append({
            "idx": i,
            "sentence": query,
            "label": label_id,  # int
        })
    return records


def save_split(source_path: Path, path: Path, split_name: str, label_map: dict):
    records = load_txt_records(source_path, label_map)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  {split_name}: {len(records)} 条 → {path}")


def main():
    print("\n正在下载 KUAKE-QIC ...")
    extract_dir = download_dataset(DATA_URL, DATA_DIR)

    # 标签列表从官方 label.txt 读取，保证 id 与数据一致
    label_path = find_file(extract_dir, "label.txt")
    label_names = label_path.read_text(encoding="utf-8").strip().splitlines()
    label_map = build_label_map(label_names)
    label_map_path = DATA_DIR / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"label_map 已保存 → {label_map_path}")
    print(f"类别数：{label_map['num_labels']}")
    for i, name in label_map["id2name"].items():
        print(f"  {i:2d} | {name}")

    print("\n正在保存各数据集分割 ...")
    save_split(find_file(extract_dir, "train.txt"), DATA_DIR / "train.json", "train", label_map)
    save_split(find_file(extract_dir, "dev.txt"), DATA_DIR / "val.json", "val", label_map)
    save_split(find_file(extract_dir, "data.txt"), DATA_DIR / "test.json", "test", label_map)
    print("  （注：PaddleNLP 包未含完整 test 集，test.json 仅有演示样本）")

    print("\n下载完成。")


if __name__ == "__main__":
    main()
