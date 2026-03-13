# ORDINARY-BENCH ms-swift 训练

使用 ms-swift 框架训练 VLM 完成 ORDINARY-BENCH 空间推理任务。

支持模型：
- **Qwen3.5-27B** — 原生 thinking 模式，全参数/LoRA
- **Qwen2.5-VL-7B** — 无原生 thinking，通过 SFT warmup 学习 `<think>` 格式

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

## Qwen2.5-VL-7B CoT 训练

Qwen2.5-VL 无原生 thinking 模式，`<think>...</think>` 作为纯文本结构由 SFT warmup 教会。

### 快速开始

```bash
# 1. 准备 CoT 数据（两个模型共用）
python training/swift/prepare_swift_data.py --mode cot-sft --data-dir data-gen/output
python training/swift/prepare_swift_data.py --mode cot-grpo --data-dir data-gen/output

# 2. CoT SFT warmup（300 步，学习 <think> 格式）
bash training/swift/run_sft_cot_warmup_qwen25vl.sh --gpus 8

# 3. CoT GRPO（从 warmup checkpoint 继续）
bash training/swift/run_grpo_cot_qwen25vl.sh \
  --cot-sft-checkpoint output/swift_cot_sft_warmup_qwen25vl \
  --gpus 8
```

### CoT 策略：有/无原生 Thinking 模式

| 维度 | Qwen3.5-27B | Qwen2.5-VL-7B |
|------|-------------|----------------|
| `enable_thinking` | `true` | `false` |
| `<think>` 来源 | 模型内置 special token | SFT 学到的纯文本标签 |
| SFT warmup 步数 | 200 | 300（需额外学格式） |
| GRPO format_weight | 0.05 | 0.10（防格式退化） |
| DeepSpeed | zero3 | zero2（7B 无需 zero3） |
| GRPO beta（KL 惩罚） | 0.001 | 0.005（防 collapse） |
| GRPO learning_rate | 5e-7 | 1e-6 |
| vllm_gpu_memory_util | 0.20 | 0.45（7B headroom 大） |

> **监控信号**：通过 rollout 日志观察 SFT warmup 后的输出。如果 `<think>` 出现率 < 80%，增加 warmup 步数或加第二轮 SFT epoch。

## Rollout 日志

CoT GRPO 脚本自动记录模型输出采样到 `{output_dir}/rollout_log.jsonl`，便于离线分析。

每 50 次奖励计算采样 4 条，记录：completions、三项奖励分数、think keys、parse path。

```bash
# 实时监控
tail -f output/swift_cot_grpo_qwen25vl/rollout_log.jsonl | python3 -m json.tool

# 统计 <think> 出现率
python3 -c "
import json
lines = open('output/swift_cot_grpo_qwen25vl/rollout_log.jsonl').readlines()
total = sum(len(json.loads(l)['samples']) for l in lines)
has_think = sum(1 for l in lines for s in json.loads(l)['samples'] if '<think>' in s['completion'])
print(f'think 出现率: {has_think}/{total} = {has_think/total*100:.1f}%')
"
```

通过环境变量 `ROLLOUT_LOG_DIR` 控制日志路径，GRPO 脚本默认设为 `$OUTPUT_DIR`。

## 硬件需求

| 模式 | 模型 | GPU 显存 | 8×80GB |
|------|------|---------|--------|
| LoRA GRPO (n=4) | Qwen3.5-27B | ~35GB/卡 | 充裕 |
| 全参数 GRPO (n=4) | Qwen3.5-27B | ~55GB/卡 | 可行 |
| 全参数 CoT SFT | Qwen3.5-27B | ~45GB/卡 | 可行 |
| 全参数 CoT GRPO (n=8) | Qwen3.5-27B | ~60GB/卡 | 可行 |
| 全参数 CoT SFT | Qwen2.5-VL-7B | ~20GB/卡 | 充裕 |
| 全参数 CoT GRPO (n=8) | Qwen2.5-VL-7B | ~35GB/卡 | 充裕 |
| LoRA SFT | Qwen3.5-27B | ~10GB/卡 | 充裕 |

## 对比实验设计

本分支与 `train/verl-qwen3vl-32b` 分支配合使用：

| 维度 | verl 分支 | swift 分支 (Qwen3.5) | swift 分支 (Qwen2.5-VL) |
|------|----------|-----------|-----------|
| 模型 | Qwen3-VL-32B | Qwen3.5-27B | Qwen2.5-VL-7B |
| 框架 | verl | ms-swift | ms-swift |
| 训练 | GRPO + LoRA | GRPO + LoRA / SFT | CoT SFT → GRPO |
| Thinking | N/A | 原生 thinking | SFT 学习 `<think>` |
| 数据 | 相同的 700 场景 / 327K 问题 | 相同 | 相同 |
| 评分 | 相同的软评分 reward | 相同 | 相同（+CoT reward） |

通过在相同数据上训练多个模型，可以分析：
1. **数据量是否充足** — 学习曲线是否饱和
2. **架构差异** — Qwen3-VL vs Qwen3.5 vs Qwen2.5-VL 在空间推理上的表现
3. **训练方法** — GRPO vs SFT vs CoT-GRPO 对空间推理能力的提升效果
4. **CoT 策略** — 原生 thinking vs SFT 学习的 `<think>` 对推理质量的影响
5. **模型规模** — 27B vs 7B 在结构化空间推理任务上的能力差距

## 多视角训练

每个场景有 4 个不同视角的渲染图（`data-gen/output/images/multi_view/`），多视角训练让模型同时看到所有视角来推理 3D 空间关系。

### 数据准备

```bash
# CoT SFT 多视角数据
python training/swift/prepare_swift_data.py \
  --mode cot-sft --multi-view --n-views 4 \
  --data-dir data-gen/output \
  --output-dir prepared_data/swift/cot_sft_mv

# CoT GRPO 多视角数据
python training/swift/prepare_swift_data.py \
  --mode cot-grpo --multi-view --n-views 4 \
  --data-dir data-gen/output \
  --output-dir prepared_data/swift/cot_grpo_mv
```

### 训练

```bash
# SFT warmup
bash training/swift/run_sft_cot_warmup_qwen25vl.sh --multi-view --gpus 8

# GRPO（从 warmup checkpoint 继续）
bash training/swift/run_grpo_cot_qwen25vl.sh \
  --multi-view \
  --cot-sft-checkpoint output/swift_cot_sft_warmup_qwen25vl_mv \
  --gpus 8
```

### 单视角 vs 多视角参数差异

| 参数 | 单视角 | 多视角（4 views） | 原因 |
|------|--------|-------------------|------|
| `MAX_PROMPT_LENGTH` | 2048 | 4096 | 4 图 ~1200 视觉 token |
| `MAX_LENGTH`（SFT） | 4096 | 8192 | 同上 |
| `VLLM_MAX_NUM_SEQS` | 32 | 16 | 每序列显存翻倍 |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.45 | 0.35 | 训练侧需更多显存 |
| `GRAD_ACCUM_STEPS` | 8 | 16 | 补偿 batch 缩小 |
