"""
用本地 Qwen2.5-0.5B-Instruct 做 zero-shot 分类，与 BERT fine-tune 对比（KUAKE-QIC 医疗意图 11 分类）

教学重点：
  1. LLM zero-shot 分类 vs fine-tuning：不需要训练，但需要精心设计 prompt
  2. 结构化输出：通过 prompt 约束模型只输出类别名，简化解析逻辑
  3. 对比视角：LLM 在哪些意图类别上表现好/差？为什么？

使用方式：
  python classify_llm.py                    # 随机采样 200 条评估（zero-shot）
  python classify_llm.py --num_samples 500  # 采样 500 条
  python classify_llm.py --few_shot 2       # few-shot：每类给 2 条示例
  python classify_llm.py --demo             # 只跑 5 条示例，快速演示

依赖：
  pip install torch transformers
"""
import os
import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
# MODEL_PATH = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
MODEL_NAME = "Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "")) / MODEL_NAME


# 11 个意图类别名（顺序与 label_map.json 的 id2name 一致），用于 prompt 和解析
LABEL_NAMES = [
    "病情诊断", "治疗方案", "病因分析", "指标解读", "就医建议",
    "疾病表述", "后果表述", "注意事项", "功效作用", "医疗费用",
    "其他",
]

SYSTEM_PROMPT = (
    "你是一个医疗搜索意图分类助手。请将给定的医疗问题分类到以下 11 个意图类别之一。\n"
    "规则：只输出类别名称本身（例如：治疗方案），"
    "不要输出序号、标点、解释或任何其他文字。\n"
    "可选类别：" + "、".join(LABEL_NAMES)
)


def build_prompt(text: str, few_shot_examples: list = None) -> str:
    """构造 user prompt。

    few_shot_examples 为 [(类别名, 问题), ...] 示例对，非空时拼在问题前。
    """
    if few_shot_examples:
        examples = "\n".join(
            f"医疗问题：{sent}\n意图类别：{label}"
            for label, sent in few_shot_examples
        )
        return f"以下是几个示例：\n{examples}\n\n医疗问题：{text}\n意图类别："
    return f"医疗问题：{text}\n意图类别："


def load_model(model_path: str, device: torch.device):
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # transformers 5.x 将 torch_dtype 参数重命名为 dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    print("模型加载完成")
    return model, tokenizer


def classify_one(text: str, model, tokenizer, device: torch.device,
                 few_shot_examples: list = None,
                 max_new_tokens: int = 16) -> str:
    """返回模型生成的类别字符串（原始输出，未做解析）。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(text, few_shot_examples)},
    ]
    # transformers 5.x apply_chat_template 返回 BatchEncoding，需取 ["input_ids"]
    encoding  = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len     = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,   # 留足余量，避免类别名被截断
            do_sample=False,   # greedy decoding，保证结果可复现
            pad_token_id=tokenizer.eos_token_id,
        )
    # 只取新生成的部分
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_prediction(raw_output: str) -> str | None:
    """
    从模型输出中提取类别名。
    模型可能输出 "科技" 或 "科技新闻" 或 "科技类"，做模糊匹配。
    """
    for name in LABEL_NAMES:
        if name in raw_output:
            return name
    return None  # 无法解析


def main():
    parser = argparse.ArgumentParser(description="LLM Zero-Shot 分类对比")
    parser.add_argument("--model_path",  default=str(MODEL_PATH))
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--num_samples", default=200, type=int,
                        help="从验证集随机采样的样本数")
    parser.add_argument("--few_shot",    default=0, type=int,
                        help="每类 few-shot 示例数，0=zero-shot（默认）")
    parser.add_argument("--seed",        default=42,  type=int)
    parser.add_argument("--demo",        action="store_true",
                        help="只跑 5 条示例（快速演示）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(data_dir / "val.json", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    id2name = {int(k): v for k, v in label_map["id2name"].items()}
    name2id = {v: k for k, v in id2name.items()}

    random.seed(args.seed)

    # ── few-shot 示例（可选）─────────────────────────────────────────────────
    # 用独立随机源采样：不消耗 val 采样的随机序列，保证与 zero-shot 对比同一批样本
    few_shot_examples = []
    if args.few_shot > 0:
        with open(data_dir / "train.json", encoding="utf-8") as f:
            train_data = json.load(f)
        by_label = {}
        for item in train_data:
            by_label.setdefault(item["label"], []).append(item["sentence"])
        rng = random.Random(args.seed)
        for label_id in range(len(LABEL_NAMES)):
            pool = by_label.get(label_id, [])
            picked = rng.sample(pool, min(args.few_shot, len(pool)))
            few_shot_examples += [(id2name[label_id], s[:50]) for s in picked]
        print(f"Few-shot 示例: 每类 {args.few_shot} 条 x {len(LABEL_NAMES)} 类 "
              f"= {len(few_shot_examples)} 条")

    n = 5 if args.demo else args.num_samples
    samples = random.sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model_path, device)

    # ── 推理 ──────────────────────────────────────────────────────────────────
    correct, total, unparseable = 0, 0, 0
    results = []
    t0 = time.time()

    for i, item in enumerate(samples):
        text     = item["sentence"]
        true_id  = item["label"]
        true_name = id2name[true_id]

        raw_output = classify_one(text, model, tokenizer, device,
                                  few_shot_examples=few_shot_examples)
        pred_name  = parse_prediction(raw_output)

        is_correct = (pred_name == true_name)
        if pred_name is None:
            unparseable += 1
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "text":        text,
            "true_label":  true_name,
            "pred_label":  pred_name,
            "raw_output":  raw_output,
            "correct":     is_correct,
        })

        # 打印进度
        # GBK 终端无法编码 ✓/✗，用 ASCII 符号替代
        status = "OK" if is_correct else ("?" if pred_name is None else "X")
        print(f"[{i+1:3d}/{len(samples)}] {status} "
              f"真实:{true_name:4s} 预测:{str(pred_name):4s} | {text[:35]}")

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0
    mode_name = "Zero-Shot" if not few_shot_examples else f"Few-Shot({args.few_shot})"
    print(f"\n{'='*50}")
    print(f"{mode_name} LLM 分类结果（{args.model_path.split('/')[-1]}）")
    print(f"{'='*50}")
    print(f"  样本数   : {total}")
    print(f"  准确率   : {correct}/{total} = {acc:.4f}")
    print(f"  无法解析 : {unparseable} 条 ({unparseable/total*100:.1f}%)")
    print(f"  总耗时   : {elapsed:.1f}s, 均值 {elapsed/total:.2f}s/条")

    # ── 与 BERT fine-tune 的对比说明 ─────────────────────────────────────────
    print(f"""
对比参考（本周 BERT fine-tune 实际结果）：
  BERT fine-tune (3 epochs, cls)   val accuracy ≈ 0.82, macro-F1 ≈ 0.78
  Qwen2.5-0.5B {mode_name}           val accuracy ≈ {acc:.2f}

思考题：
  1. LLM zero-shot 在哪些意图类别上表现好？为什么？（如"治疗方案" vs "指标解读"）
  2. fine-tuning 数据量越多效果越好吗？试试只用 1000 条训练数据。
  3. 用 --few_shot 2 对比 few-shot：示例越多准确率越高吗？
""")

    # 保存结果（按模式区分文件名，zero-shot 与 few-shot 结果互不覆盖）
    mode_tag = "zero_shot" if not few_shot_examples else f"few_shot{args.few_shot}"
    out_path = ROOT / "outputs" / f"llm_{mode_tag}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode_tag, "accuracy": acc, "total": total,
            "correct": correct, "unparseable": unparseable,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
