# KUAKE-QIC 中文医疗意图分类：BERT fine-tune vs LLM 三路对比

> 同一份数据、同一套评估协议，对比三类中文意图分类方案：
> **BERT 判别式微调** vs **LLM zero-shot / few-shot** vs **LLM 指令微调（LoRA）**
>
> 数据集：[KUAKE-QIC](https://tianchi.aliyun.com/dataset/96483)（CBLUE 中文医疗 benchmark，11 类医疗搜索意图）

## 实验结果（验证集）

| 方法 | 准确率 | 样本量 | 备注 |
|---|---:|---:|---|
| **BERT fine-tune（cls 池化，3 epochs）** | **81.7%** | 1955（全量） | macro-F1 77.7% |
| Qwen2.5-0.5B SFT（LoRA r=8） | 46.0% | 200 | 可训练参数仅 0.22%（1.08M / 495M） |
| Qwen2.5-0.5B zero-shot | 29.0% | 200 | 13% 输出无法解析 |
| Qwen2.5-0.5B few-shot（每类 3 条） | 25.5% | 200 | 示例越多反而更差 |

**核心结论：小数据 + 类别不平衡场景下，判别式 fine-tune 完胜生成式路线（81.7% > 46.0% > 29.0% > 25.5%）。**
完整错误分析见 [Result.md](Result.md)。

## 项目亮点

- **三路方案同台对比**：判别式微调 / 免训练 zero-shot / LoRA 指令微调，同一数据、同一评估
- **池化策略消融**：cls / mean / max 三种句向量提取方式（cls 81.7% ≈ mean 80.9% > max 77.8%）
- **完整工程细节**：分层学习率、Linear Warmup、梯度累积、加权 CrossEntropyLoss（24x 类别不均衡）
- **SFT 关键机制**：chat 格式转换、loss masking（仅类别 token 计算损失）、LoRA 参数高效微调
- **CPU 可完整复现**：无 GPU 也能跑完全流程（BERT 每 epoch 约 15 分钟）

## 项目结构

```
kuaque-qic-bert-vs-llm/
├── src/                    # BERT fine-tuning 实现
│   ├── download_data.py    #   数据下载 → 本地 JSON
│   ├── explore_data.py     #   数据探索 → 5 张分析图表
│   ├── dataset.py          #   Dataset / DataLoader
│   ├── model.py            #   BertModel + 三种池化分类头
│   ├── train.py            #   训练循环（分层 lr / warmup / 加权 loss）
│   ├── evaluate.py         #   评估 + 混淆矩阵
│   ├── predict.py          #   单条 / 批量推理
│   └── compare_class_weight.py  # 加权 loss 对比实验
├── src_llm/                # LLM 对比实现
│   ├── classify_llm.py     #   zero-shot / few-shot 分类
│   ├── train_sft.py        #   LoRA 指令微调（chat 格式 + loss masking）
│   └── evaluate_sft.py     #   SFT 评估 + 多路准确率对比
├── data/                   # KUAKE-QIC 数据集（JSON 格式）
├── outputs/                # 训练日志 / 图表 / LoRA adapter / 评估明细
└── requirements.txt
```

## 快速开始

> 在项目根目录执行；预训练模型（bert-base-chinese / Qwen2.5-0.5B-Instruct）的下载与路径配置见 [USAGE_GUIDE.md](USAGE_GUIDE.md)。

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 数据探索（生成 outputs/figures/ 下 5 张图表）
python src/explore_data.py

# 3. BERT fine-tune（三种池化策略消融）
python src/train.py --pool cls                      # 默认，val_acc 81.7%
python src/train.py --pool mean
python src/train.py --pool max --use_class_weight

# 4. 评估与推理
python src/evaluate.py --pool cls
python src/predict.py

# 5. LLM 对比（zero-shot / few-shot / SFT 评估）
python src_llm/classify_llm.py
python src_llm/classify_llm.py --few_shot 3
python src_llm/evaluate_sft.py
```

## 文档索引

| 文档 | 内容 |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 技术方案：选型理由、流水线、消融矩阵、踩坑记录 |
| [Result.md](Result.md) | 实验报告：全部结果、逐 epoch 曲线、错误分析 |
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | 使用指南：环境、数据下载、训练 / 评估命令、参数说明 |
| [RESUME_GUIDE.md](RESUME_GUIDE.md) | 简历呈现：可量化数据、岗位写法、面试问答 |
| [GIT_UPLOAD_GUIDE.md](GIT_UPLOAD_GUIDE.md) | 仓库维护：git 历史清理与推送新仓库完整流程 |

## 环境要求

- Python 3.12，`torch >= 2.6`，`transformers >= 5.5`，`peft`（LoRA），`scikit-learn`
- 硬件：CPU 即可复现 BERT 全流程；GPU（如 RTX 4060 8GB）可加速
