# Qwen3.5-27B + ms-swift

使用 ms-swift 框架训练 Qwen3.5-27B 完成 ORDINARY-BENCH 空间推理任务。

## 为什么用 ms-swift 而不是 verl？

- verl 当前版本**不支持 Qwen3.5** 架构
- ms-swift v4.0+ 已原生支持 Qwen3.5 Dense/MoE 模型的 SFT 和 GRPO
- Qwen3.5-27B 是原生多模态模型（早期融合），需要 `transformers>=5.2.0`

## 快速开始

```bash
# 1. 安装环境
bash training/swift/setup.sh

# 2a. 准备 GRPO 数据（默认写入 prepared_data/swift/grpo）
python training/swift/prepare_swift_data.py --data-dir data-gen/output

# 2b. 准备 SFT 数据（默认写入 prepared_data/swift/sft）
python training/swift/prepare_swift_data.py --mode sft --data-dir data-gen/output

# 3. GRPO 训练（推荐）
bash training/swift/run_grpo.sh --gpus 8

# 4. 或 SFT 训练（需要先执行步骤 2b）
bash training/swift/run_sft.sh --gpus 8
```

> **注意**: 导出的 JSONL 数据中图片路径为绝对路径，迁移到服务器后需重新运行数据准备步骤。
> 默认输出目录:
> - `prepared_data/swift/grpo`
> - `prepared_data/swift/sft`

## 数据格式

ms-swift 使用 JSONL 格式，每行一个样本：

```json
{
  "images": ["/path/to/scene.png"],
  "messages": [
    {"role": "system", "content": "You are a spatial reasoning assistant..."},
    {"role": "user", "content": "<image>\nObjects in the image:\n  - obj_0: ...\n\nQuestion:\nCompare the distance..."}
  ],
  "solution": "{\"type\": \"qrr\", \"qid\": \"qrr_0001\", \"answer\": \"<\", \"ratio\": 0.62}"
}
```

- `images`: 图片路径列表
- `messages`: 对话格式的 prompt
- `solution`: ground truth JSON（reward 函数解析用）

## 奖励函数

自定义奖励插件位于 `plugins/ordinary_bench_reward.py`，注册了两个 ORM：

| 名称 | 说明 |
|------|------|
| `ordinary_bench_accuracy` | 软评分：QRR 基于距离比值部分给分，TRR 多级评分 |
| `ordinary_bench_format` | 格式检查：输出是否为有效答案格式 |

评分逻辑与 `VLM-test/verl_reward.py` 完全一致。

## 配置说明

## 服务器最小步骤

如果服务器上只有代码、还没有 `prepared_data/swift`，按默认路径直接执行:

```bash
git checkout train/swift-qwen35-27b
bash training/swift/setup.sh
python training/swift/prepare_swift_data.py --data-dir data-gen/output
python training/swift/prepare_swift_data.py --mode sft --data-dir data-gen/output
bash training/swift/run_sft.sh --gpus 8
bash training/swift/run_grpo.sh --gpus 8
```

如果你已经把本地的 `prepared_data/swift` 一并同步到服务器，则可以跳过数据准备:

```bash
git checkout train/swift-qwen35-27b
bash training/swift/setup.sh
bash training/swift/run_sft.sh --gpus 8
bash training/swift/run_grpo.sh --gpus 8
```

## 三组对比实验

### 1. 纯 RL（前期格式奖励高，后期正确性奖励高）

推荐直接拆成两段 GRPO：

```bash
# Stage 1: 先把输出格式训稳定
bash training/swift/run_grpo.sh \
  --gpus 8 \
  --epochs 1 \
  --accuracy-weight 0.4 \
  --format-weight 0.6 \
  --output-dir output/swift_grpo_qwen35_27b_stage1

# Stage 2: 再把重心切到正确性
bash training/swift/run_grpo.sh \
  --gpus 8 \
  --epochs 4 \
  --accuracy-weight 0.95 \
  --format-weight 0.05 \
  --resume-from-checkpoint output/swift_grpo_qwen35_27b_stage1 \
  --output-dir output/swift_grpo_qwen35_27b_stage2
```

`--resume-from-checkpoint` 可以直接传某个 `checkpoint-*` 目录，也可以直接传整个输出目录；脚本会自动选择最新 checkpoint。

### 2. 先 SFT 后 RL

```bash
# Stage 1: SFT 固化 <answer> 输出格式
bash training/swift/run_sft.sh \
  --gpus 8 \
  --output-dir output/swift_sft_qwen35_27b

# Stage 2: 从 SFT adapter 继续做 GRPO
bash training/swift/run_grpo.sh \
  --gpus 8 \
  --sft-adapter output/swift_sft_qwen35_27b \
  --accuracy-weight 0.95 \
  --format-weight 0.05 \
  --output-dir output/swift_grpo_from_sft_qwen35_27b
```

`--sft-adapter` 也可以传 `checkpoint-*` 目录，或整个 SFT 输出目录；脚本会自动选择最新 checkpoint，并同时设置 `adapters` 和 `ref_adapters`。

### 3. 全部 SFT

```bash
bash training/swift/run_sft.sh \
  --gpus 8 \
  --output-dir output/swift_sft_only_qwen35_27b
```

### GRPO 默认参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_generations` | 4 | 每 prompt 生成的回答数 |
| `learning_rate` | 1e-6 | RL 学习率 |
| `epsilon` | 0.2 | PPO/GRPO 裁剪比 |
| `temperature` | 0.7 | 采样温度 |
| `max_completion_length` | 128 | 最大回答长度 |
| `max_prompt_length` | 1536 | 最大 prompt 长度 |
| `lora_rank` | 32 | LoRA 秩 |
| `max_grad_norm` | 0.5 | 梯度裁剪（防止训练不稳定） |

`run_grpo.sh` 额外支持：

- `--accuracy-weight` / `--format-weight`: 控制两个 reward 的相对比重
- `--reward-weights a b`: 等价写法
- `--resume-from-checkpoint`: 从上一个 GRPO checkpoint 继续
- `--sft-adapter`: 从 SFT 输出目录或 checkpoint 接入 LoRA adapter
- `--adapters` / `--ref-adapters`: 手工指定 adapter
- `--output-dir`: 自定义输出目录
- `--epochs`: 覆盖默认训练轮数

`run_sft.sh` 额外支持：

- `--output-dir`: 自定义 SFT 输出目录
- `--epochs`: 覆盖默认训练轮数

### 全参数训练

```bash
bash training/swift/run_grpo.sh --full --gpus 8
```

## 硬件需求

| 模式 | GPU 显存 | 8×80GB |
|------|---------|--------|
| LoRA GRPO (n=4) | ~35GB/卡 | 充裕 |
| 全参数 GRPO (n=4) | ~55GB/卡 | 可行 |
| LoRA SFT | ~10GB/卡 | 充裕 |

## 对比实验设计

本分支与 `train/verl-qwen3vl-32b` 分支配合使用：

| 维度 | verl 分支 | swift 分支 |
|------|----------|-----------|
| 模型 | Qwen3-VL-32B | Qwen3.5-27B |
| 框架 | verl | ms-swift |
| 训练 | GRPO + LoRA | GRPO + LoRA / SFT |
| 数据 | 相同的 700 场景 / 327K 问题 | 相同 |
| 评分 | 相同的软评分 reward | 相同 |

通过在相同数据上训练两个不同模型，可以分析：
1. **数据量是否充足** — 学习曲线是否饱和
2. **架构差异** — Qwen3-VL vs Qwen3.5 原生多模态在空间推理上的表现差异
3. **训练方法** — GRPO vs SFT 对空间推理能力的提升效果
