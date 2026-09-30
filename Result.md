# KUAKE-QIC 医疗意图分类：BERT fine-tune vs LLM 对比实验报告

> 任务：医疗搜索意图 11 分类（病情诊断 / 治疗方案 / 病因分析 / 指标解读 / 就医建议 / 疾病表述 / 后果表述 / 注意事项 / 功效作用 / 医疗费用 / 其他）
> 数据：KUAKE-QIC（CBLUE），train 6931 条 / val 1955 条
> 硬件：CPU 12 逻辑核（Windows，无 CUDA，未启用 DirectML）
> 模型：BERT-base-Chinese（fine-tune）、Qwen2.5-0.5B-Instruct（zero-shot / few-shot / SFT-LoRA）

---

## 1. 结果总览

| 方法 | 准确率 | macro-F1 | 无法解析 | 样本量 | 推理耗时 |
|---|---|---|---|---|---|
| **BERT fine-tune（cls 池化，3 epochs）** | **81.7%** | **77.7%** | 无（判别式） | val 全量 1955 | — |
| Qwen2.5-0.5B zero-shot | 29.0% | — | 13.0% | val 随机 200 | 127s（0.63s/条） |
| Qwen2.5-0.5B few-shot（每类 3 条） | 25.5% | — | 11.0% | val 随机 200 | 485s（2.43s/条） |
| Qwen2.5-0.5B SFT（LoRA） | 46.0% | — | 2.0% | val 随机 200 | — |

**核心结论：判别式 fine-tune（BERT 81.7%）远胜生成式路线（SFT 46.0% > zero-shot 29.0% > few-shot 25.5%）。**

---

## 2. BERT fine-tune 详细结果

### 2.1 三种池化策略对比（各 3 epochs）

| 池化策略 | val_acc（E3） | val_macro_f1（E3） | 单 epoch 耗时 |
|---|---|---|---|
| **cls**（默认） | **81.7%** | **77.7%** | ~15 分钟 |
| mean | 80.9% | 77.4% | ~15 分钟 |
| max（加权） | 77.8% | 74.3% | ~15 分钟 |

- cls 与 mean 接近（差距 <1 个点），max 池化明显更差（-4 个点）：max 逐维取最大会丢失位置/分布信息，对细粒度意图区分不利
- 三类均在 45 分钟（CPU）内完成训练

### 2.2 cls 池化逐 epoch 曲线（train_log_cls.json）

| Epoch | train_loss | train_acc | val_acc | val_macro_f1 | 耗时 |
|---|---|---|---|---|---|
| 1 | 0.978 | 66.8% | 81.0% | 78.1% | 918s |
| 2 | 0.378 | 87.1% | 81.5% | 78.0% | 908s |
| 3 | 0.239 | 92.1% | 81.7% | 77.7% | 896s |

**过拟合信号明显**：train_acc 66.8% → 92.1% 快速上升，而 val_acc 每 epoch 仅 +0.3~0.5 个百分点（81.0 → 81.7），val_macro_f1 反而从 78.1% 微降到 77.7%。继续训练收益趋近于零，3 epochs 是合适停止点。

### 2.3 acc 与 macro-F1 的差距（81.7% vs 77.7%）

数据类别极不平衡（最多"治疗方案"1750 条 vs 最少"指标解读"137 条，24.1 倍）：

- **accuracy 被多数类主导**，虚高
- **macro-F1 平等对待每类**，少数类（病因分析 / 指标解读 / 功效作用）分错直接拉低
- 模型在"牺牲少数类保多数类"，加权 loss（`compare_class_weight.py`）有提升空间（待运行）

---

## 3. LLM 对比详细结果

### 3.1 Zero-shot（29.0%，58/200）

- 无法解析 26 条（13.0%）：模型输出自由文本（如"感染治疗""症状解读"）而非类别名，子串匹配失败
- **"治疗方案"类最强**：带"怎么治/怎么办/治疗方法"的问题几乎全对（强关键词信号）
- **系统性偏向"就医建议"**：大量"真实=治疗方案"被判成"就医建议"（孩子高烧不退怎么办、怀孕失眠怎么办），模型把"求助语气"误解为"咨询就医"
- **"其他"类几乎全灭**：模型不输出"其他"这种抽象兜底类

### 3.2 Few-shot 3（25.5%，51/200）——示例越多反而更差

| 指标 | zero-shot | few-shot 3 |
|---|---|---|
| 准确率 | 29.0% | **25.5%**（-3.5pp） |
| 无法解析 | 13.0% | 11.0% |
| 推理耗时 | 0.63s/条 | 2.43s/条（prompt 长 4 倍） |

**反直觉但真实的发现：few-shot 不是免费的**：

1. 0.5B 模型容量有限：33 条示例（每类 3 条）把 prompt 拉长 4 倍，注意力被大量示例稀释，抓不住当前问题的关键信息
2. 示例随机采样、与查询分布不匹配：如"注意事项"类示例都是"能吃 XX 吗"，可能带偏模型
3. 推理成本显著上升（2.43s/条 vs 0.63s/条）

### 3.3 SFT（LoRA，200 条完整评估 46.0%，92/200）

- SFT 训练日志（train_log_sft.json）：3 epochs，单 epoch **约 65 分钟**（CPU），train_loss 0.667 → 0.212
- **val_loss 先降后升**（0.363 → 0.320 → 0.389），E3 已过拟合——训练 2 epochs 可能效果更好
- **结构化输出问题基本被训练解决**：无法解析率 13%（zero-shot）→ 11%（few-shot）→ **2.0%**（SFT，仅 4 条）
- **但出现新问题——过度使用"其他"类**：大量真实=后果表述/疾病表述/注意事项的样本被判成"其他"（孕妇高烧有影响吗→其他、甲亢的症状→其他、蜂蜜一天喝多少→其他），与 zero-shot 恰好相反（zero-shot 从不输出"其他"，SFT 把不确定的都归"其他"）
- 偶发错别字仍导致解析失败：如"治病方案"（漏字）匹配不到"治疗方案"——生成式分类的固有风险，SFT 后依然存在
- **demo 5 条 80% 严重高估**：前 5 条恰好多是"治疗方案"类（模型强项），200 条完整评估回落到 46.0%

---

## 4. 综合结论

1. **小数据 + 强不平衡任务，判别式 fine-tune 完胜生成式路线**：BERT 81.7% > SFT 46.0% > zero-shot 29.0% > few-shot 25.5%。6931 条标注数据足以让 BERT-base 学到意图边界，而 0.5B 通用模型即使 SFT 后仍有 35.7 个百分点的差距。
2. **few-shot 对 0.5B 小模型不升反降**：提示词技巧的收益依赖模型容量与示例质量，不能默认"示例越多越好"。
3. **SFT 是生成式路线里唯一有效的方案**：比 zero-shot 提升 17 个百分点，且把无法解析率从 13% 压到 2%；但训练成本高（3.2 小时/3 epochs）、E3 已过拟合、且过度使用"其他"类（意图边界学得不如 BERT 细）。
4. **生成式分类的工程代价**：即使 SFT 后仍有 2% 无法解析（错别字如"治病方案"），需要解析兜底（模糊匹配/重试），判别式分类没有这个问题。
5. **类别不平衡影响所有方法**：macro-F1 低于 acc 约 4 个点，加权 loss 与少数类增强是后续重点（`compare_class_weight.py` 待运行，需 `pip install seaborn`）。

---

## 5. 复现命令

```bash
# 1. 数据探索（生成 figures/ 下 5 张图）
python "杨建国\week06\src\explore_data.py"

# 2. BERT fine-tune（cls / mean / max+weighted）
python "杨建国\week06\src\train.py" --pool cls
python "杨建国\week06\src\train.py" --pool mean
python "杨建国\week06\src\train.py" --pool max --use_class_weight

# 3. BERT 评估 / 预测
python "杨建国\week06\src\evaluate.py"
python "杨建国\week06\src\predict.py"

# 4. LLM zero-shot / few-shot 评估（结果写入 outputs/llm_*_results.json）
python "杨建国\week06\src_llm\classify_llm.py"
python "杨建国\week06\src_llm\classify_llm.py" --few_shot 3

# 5. LLM SFT 评估（四方对比，自动读取上述结果文件）
python "杨建国\week06\src_llm\evaluate_sft.py"
```

## 6. 结果文件清单

| 文件 | 内容 |
|---|---|
| `outputs/train_log_cls.json` | BERT cls 池化训练日志（3 epochs） |
| `outputs/train_log_mean.json` | BERT mean 池化训练日志 |
| `outputs/train_log_max_weighted.json` | BERT max+加权训练日志 |
| `outputs/train_log_sft.json` | SFT LoRA 训练日志（3 epochs） |
| `outputs/llm_zero_shot_results.json` | zero-shot 200 条评估明细 |
| `outputs/llm_few_shot3_results.json` | few-shot(3) 200 条评估明细 |
| `outputs/llm_sft_results.json` | SFT 200 条评估明细（accuracy 0.46） |
| `outputs/figures/` | 数据探索图（类别分布 / 长度分布 / token 分布） |
