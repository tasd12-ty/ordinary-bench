# Qwen3-VL-32B + verl v0.7.0

本目录当前整理后的推荐目标是:

- 模型: `Qwen/Qwen3-VL-32B`
- 框架: `verl v0.7.0`（固定到 commit `80eb57ea`）
- 训练方式: `GRPO + LoRA`
- 环境管理: `uv` 虚拟环境

## 结论先说

在这个仓库里，当前推荐且已经接线的路径是:

1. 用 `training/setup_uv.sh` 创建 `.venv`
2. 用 `training/prepare_data.sh` 导出 `verl` 可直接读取的多模态 parquet
3. 用 `training/run_grpo.sh` 启动 `Qwen3-VL-32B` 的 GRPO-LoRA

`run_sft.sh` 对 `Qwen3-VL-32B` 不再作为推荐入口。
原因是截至 `verl v0.7.0`，本仓库内没有稳定验证通过的 `Qwen3-VL` 多模态 SFT 接线；继续保留默认可执行入口只会误导使用。

训练目标环境默认按 `Linux x86_64 + NVIDIA GPU + CUDA` 设计，不以 macOS 本机训练为目标。

## 版本固定

版本统一放在 [versions.env](/Users/tsyq/code/ordinary-bench/training/versions.env)。
这里刻意用 `versions.env + setup_uv.sh` 作为迁移载体，而不是直接依赖单一 `uv.lock`，因为 GPU/CUDA 轮子会随集群环境变化。

当前默认值:

- Python `3.11`
- verl `80eb57ea` (`v0.7.0`)
- torch `2.7.1`
- torchvision `0.22.1`
- vllm `0.11.0`
- transformers `4.57.0`
- ray `2.49.2`

如果迁移到另一台机器，优先复制:

- 整个仓库
- [versions.env](/Users/tsyq/code/ordinary-bench/training/versions.env)

然后重新执行安装脚本即可。

> **注意**: 导出的 parquet 数据中图片路径为绝对路径，迁移后需要在目标机器上重新运行 `training/prepare_data.sh` 生成数据。

## 环境安装

```bash
bash training/setup_uv.sh
source .venv/bin/activate
```

如果你的集群不是 CUDA 12.8，先改 [versions.env](/Users/tsyq/code/ordinary-bench/training/versions.env) 里的 `TORCH_INDEX_URL`、`TORCH_VERSION`、`TORCHVISION_VERSION`。

## 数据准备

单视角:

```bash
bash training/prepare_data.sh --data-dir ./data-gen/output
```

多视角:

```bash
bash training/prepare_data.sh --data-dir ./data-gen/output --multi-view --n-views 4
```

输出目录默认是:

- [VLM-test/verl_data_qwen3vl32b](/Users/tsyq/code/ordinary-bench/VLM-test/verl_data_qwen3vl32b)

当前导出的 RL 样本有三个关键点:

- `prompt` 是结构化 chat messages，不再是 JSON 字符串
- `images` 是 `[{\"path\": ...}]` 列表
- user message 里会按图片数自动补 `<image>` 占位符

这三点是为了对齐 `verl` 当前的多模态 RL loader。

## 启动训练

```bash
bash training/run_grpo.sh --gpus 8
```

如果模型已经下到本地:

```bash
bash training/run_grpo.sh --model /path/to/Qwen3-VL-32B --gpus 8
```

当前默认配置文件是 [grpo_qwen3vl32b_lora.yaml](/Users/tsyq/code/ordinary-bench/training/configs/grpo_qwen3vl32b_lora.yaml)。

配置重点:

- `use_remove_padding: false`
- `target_modules: all-linear`
- `exclude_modules: ".*visual.*"`
- `rollout.name: vllm`

这是为了更稳妥地匹配 `Qwen3-VL` 的当前 `verl`/`vLLM` 路径。

## 硬件建议

保守起见，当前配置按 `8 x A100/H100 80GB` 级别准备。

如果显存不足，优先调小:

1. `data.train_batch_size`
2. `actor_rollout_ref.rollout.n`
3. `actor_rollout_ref.rollout.gpu_memory_utilization`
4. `actor_rollout_ref.actor.ppo_mini_batch_size`

## 已知限制

- `Qwen3-VL-32B` 在这个仓库里当前只整理了推荐的 `GRPO-LoRA` 路径
- `run_sft.sh` 对该模型默认不开放
- 全参数 RL 对 32B-VL 成本很高，不作为当前默认目标

## 官方参考

- verl `v0.7.0`: https://github.com/volcengine/verl/releases/tag/v0.7.0
- Reward function docs: https://verl.readthedocs.io/en/latest/preparation/reward_function.html
- RL dataset docs: https://verl.readthedocs.io/en/latest/_modules/verl/utils/dataset/rl_dataset.html
- Config docs: https://verl.readthedocs.io/en/latest/examples/config.html
