# VRAMScout

[English](README.md) | **中文**

**这个模型能不能塞进我的 GPU？显存都花到哪里了？最多能跑多长 context？**

VRAMScout 是一个面向现代开源权重模型的 **LLM 部署显存预检工具**。它读取 Hugging Face 模型元数据，不需要下载完整权重文件，就可以结合当前 GPU 的可用显存，估算每张卡上的：

- 模型权重；
- KV / MLA / Sliding KV / recurrent state；
- sparse indexer / compressed cache；
- CUDA / runtime workspace；
- chunked prefill scratch；
- safety headroom；
- **在当前硬件和部署配置下最多能支持多长 context。**

> **v0.7 alpha：** cache/state 显存已经重构为可复用的 memory components，而不是“一个模型写一套公式”。同时加入了公开 vLLM / SGLang 部署日志的 `Predicted vs Actual` 校验，以及显式、可审计的 engine-layout calibration。

## 快速开始

```bash
pip install -e .

# 当前 GPU 上估算
vramscout Qwen/Qwen3.8-Flash-Next

# 按 vLLM 的 gpu_memory_utilization 逻辑估算
vramscout Qwen/Qwen3.8-Flash-Next \
  --engine vllm \
  --gpu-memory-utilization 0.92

# Kimi K3，8 卡 + DCP
vramscout moonshotai/Kimi-K3 \
  --tp 8 \
  --dcp 8 \
  --context 262144

# 假设有 8 张 192 GiB GPU
vramscout nvidia/MiniMax-M3-NVFP4 \
  --vram-gib 192 \
  --tp 8 \
  --context 1048576

# 查看当前覆盖的模型架构
vramscout --list-supported
```

`--vram-gib` 表示**单张 GPU 的显存预算**。例如 `--vram-gib 96 --tp 8` 表示 8 张 96 GiB GPU，而不是一张虚构的 768 GiB GPU。

## v0.7 的显存模型

现代模型已经不能统一套用传统的

```text
2 × layers × kv_heads × head_dim × tokens
```

来计算 KV Cache。VRAMScout 现在把模型解析成多个可以复用的显存组件：

```text
Hugging Face config
        ↓
architecture/profile parser
        ↓
Memory components
 ├─ KVCache
 ├─ LatentCache / MLA
 ├─ SlidingKV
 ├─ SparseIndexer
 ├─ GatedDeltaNet / KDA state
 ├─ Mamba-2 recurrent state
 └─ DeepSeek compressed K=V state
        ↓
TP / DCP placement
        ↓
engine-layout calibration（可选）
        ↓
每卡显存 + 最大 context
```

这样以后新模型如果只是复用了已有的 inference-state primitive，通常不需要再单独写一套显存公式。只有模型引入新的 persistent state、cache 结构或新的并行放置规则时，才需要增加新的基础组件。

## 当前架构覆盖

| 模型家族 | 已建模的 inference state |
|---|---|
| Standard decoder | MHA / GQA / MQA K+V |
| **Qwen3.8 / qwen3_5** | full-attention KV + GatedDeltaNet state |
| **Qwen3.8-Flash-Next / qwen4_exp** | QSA KV + compressed indexer + GatedDeltaNet |
| **DeepSeek-V4 Flash/Pro** | SWA + compressed CSA/HCA / MLA-index state |
| **GLM-5.2** | MLA latent + DSA IndexShare |
| **GLM-5.3-Flash** | sparse MLA/indexer + KDA recurrent state |
| **Kimi-K2.5/K2.6** | MLA latent；TP replicated、DCP sequence-sharded |
| **Kimi K3** | Gated MLA + KDA recurrent state |
| **Tencent Hy4-preview** | gated MLA/DSA + IndexShare |
| **MiniMax-M2.7** | 标准 GQA + TP KV-head placement |
| **MiniMax-M3** | paged KV + sparse-attention index state |
| **Gemma 4** | global KV + bounded sliding-window KV |
| **Nemotron-3.5-Lightning** | attention KV + Mamba-2 recurrent state |
| **Mistral Medium 3.5** | 标准 GQA |
| **Mistral Small 4** | MLA latent cache |
| **LongCat-2.0** | MLA latent + sparse index state |
| **MiMo-V2.5 / Pro** | full KV + bounded chunk/SWA KV，支持非对称 K/V 维度 |

完整 checkpoint registry 见 [`validation/current_models_v06.json`](validation/current_models_v06.json)。

## 最大 context 是怎么估算的？

VRAMScout 会寻找最大的 context `C`，使得：

```text
模型权重
+ architecture cache/state(C)
+ CUDA/runtime
+ prefill scratch(C)
+ safety reserve
<= 当前每卡可用显存 / engine budget
```

然后再和模型 config 自己声明的最大 context 比较：

```text
usable max context = min(VRAM-derived max, model config limit)
```

因此它回答的是：

> **在当前模型、dtype、TP/DCP、engine 和 GPU 显存预算下，理论上最多能容纳多长上下文。**

而不是只把模型配置里的 `max_position_embeddings` 原样打印出来。

## context 长度估计到底准不准？

目前结论是：**对 cache/state 本身已经比较准；对“最终 OOM 边界对应的最大 context”仍然属于部署预估，而不是硬保证。**

原因是最大 context 由两部分共同决定：

```text
固定项：weights + runtime + CUDA graph/workspace + allocator overhead
变量项：随 context 增长的 KV / MLA / index cache
```

VRAMScout 对变量项已经做了 architecture-aware 建模。v0.7 新加入的 6 组公开 hard cache receipts 中：

```text
纯架构公式 MAPE            2.850%
公开样本上的校准后误差       0.237%
```

其中部分公开结果：

| 模型 | Raw error | 显式 calibration 后 |
|---|---:|---:|
| Qwen3.8-Flash-Next · vLLM BF16 | 2.813% | **0.005%** |
| Kimi K3 · vLLM BF16 | **0.493%** | 0.493% |
| GLM-5.3-Flash · vLLM BF16 | 6.500% | **0.889%** |
| Nemotron-3.5 Lightning · vLLM FP8 | 7.293% | **0.031%** |
| Mistral Small 4 · SGLang BF16 | **0.002%** | 0.002% |
| MiniMax M3 · vLLM BF16, TP8 | **0.001%** | 0.001% |

但需要特别注意：

**cache/token 的误差不等于 max-context 的误差。**

例如 CUDA graph、allocator fragmentation、backend workspace、paged-cache rounding、EP/PP routing buffer、multimodal activation 等固定或半固定开销如果估错了，即使 KV slope 很准，也可能让最终的 OOM 边界移动很多 token。

所以当前推荐把输出理解为：

```text
显存结构分析：        较可靠
cache/token 斜率：     已有公开日志验证，通常较准
最大 context：         很有用的 deployment preflight
精确 OOM 临界点：      仍以真实 vLLM/SGLang profiler 为准
```

一个典型反例是 **Gemma 4**：某个公开 vLLM 版本中，真实 hybrid allocator 占用和简单 tensor layout 相差约 **36.7%**。VRAMScout 没有强行塞一个常数把它“校准漂亮”，而是明确标成 version-sensitive unresolved case。

换句话说：

> **现在 VRAMScout 已经适合回答“我大概能不能跑 128K / 256K / 1M context，以及瓶颈在哪里”，但还不应该承诺“预测 263,421 tokens，真实就一定在 263,421 附近 OOM”。**

## Predicted vs Actual

完整公开校验总表：

[`validation/PREDICTED_VS_ACTUAL.md`](validation/PREDICTED_VS_ACTUAL.md)

里面严格区分：

- `Raw`：只由模型 config 和架构公式推导；
- `Actual`：公开 vLLM / SGLang 启动日志里的真实 allocator 数据；
- `Calibrated`：仅在完全匹配的 `(engine, memory family, cache dtype)` 上应用公开 layout factor。

目前不会把 engine-specific factor 偷偷写进 architecture constant。

如果想看完全未经经验修正的结果：

```bash
vramscout MODEL --calibration none
```

## vLLM / SGLang 日志解析

可以直接把自己的服务启动日志交给 VRAMScout：

```bash
vramscout --parse-log vllm.log
vramscout --parse-log sglang.log --json
```

目前会尝试抽取：

- model loading memory；
- available / active KV memory；
- logical cache tokens；
- maximum concurrency；
- CUDA graph memory；
- peak activation memory；
- consumed memory。

这些数据可以继续扩充 `Predicted vs Actual` corpus，用来校准不同 engine / version / hardware 的真实 layout overhead。

## vLLM memory-budget 模式

```bash
vramscout MiniMaxAI/MiniMax-M3 \
  --engine vllm \
  --gpu-memory-utilization 0.92 \
  --tp 8
```

vLLM 实际会在

```text
total VRAM × gpu_memory_utilization
```

这个预算里规划实例，并在启动阶段自行 profile 非 KV 开销。VRAMScout 会模拟这套预算逻辑。

v0.7 中，`--engine vllm` 默认不再额外减一个 5% reserve，因为 `gpu_memory_utilization` 本身已经在 engine budget 外保留了 headroom；否则会重复保守。

如果确实希望在 vLLM budget 内继续留额外显存，可以显式设置：

```bash
--reserve-gib N
```

## 多 GPU 语义

```bash
# 本机 GPU 0..7
vramscout moonshotai/Kimi-K3 --tp 8 --dcp 8

# 假设 4 张 96 GiB GPU
vramscout mistralai/Mistral-Small-4-119B-2603-NVFP4 \
  --vram-gib 96 \
  --tp 4
```

VRAMScout 不会简单地把总显存相加。

- 对本机多卡，使用所选 GPU 中 **free VRAM 最少的 rank** 作为瓶颈；
- standard KV/index heads 按明确的 TP sharding / replication 规则计算；
- MLA 不会因为 `TP=N` 就无脑除以 N；
- DCP 只在已有公开证据支持其 sequence-sharding 行为的架构上启用，否则 fail closed。

## 原生 checkpoint 字节

现代 FP8 / NVFP4 / INT4 checkpoint 往往是混合精度，不能简单用：

```text
parameters × bits-per-parameter
```

来估算权重显存。

VRAMScout 会优先读取 Hugging Face 上真实 `.safetensors` shard size，并避免把同一个仓库里的 `model-*` 与 `consolidated-*` 两套等价导出重复计算。

## 多模态范围

对 GLM-5.3-Flash、Kimi K3、MiniMax M3、Gemma 4、MiMo 等多模态模型：

- checkpoint 中的 vision/audio 权重驻留会被计入；
- text-generation cache/state 会按架构计算；
- 具体一张图、一段视频、一段音频带来的 token expansion 和 encoder activation peak 目前还不能精确预测。

所以这些模型的**纯文本部署预检**比完整多模态 peak 更可靠。

## Validation 与开发

```bash
git clone https://github.com/S3vryn/vramscout.git
cd vramscout
pip install -e . pytest

pytest -q
python validation/run.py
python validation/predicted_vs_actual.py
python validation/current_models_smoke.py
```

VRAMScout 的验证原则是：

> config 推出来的是 **estimate**；公开 vLLM/SGLang 日志是 **evidence**；engine/version-specific 的修正是 **calibration profile**，不能冒充新的架构常数。

## Roadmap

- [x] 可组合 cache/state component graph
- [x] 当前 2026 主流 open-weight 架构覆盖
- [x] TP / DCP placement semantics
- [x] vLLM memory-budget preflight
- [x] vLLM / SGLang startup-log parser
- [x] Predicted vs Actual 公开数据集
- [x] 显式 engine-layout calibration
- [ ] 多 engine version / 多 hardware 的 held-out calibration split
- [ ] Gemma4 hybrid allocator/version model
- [ ] EP / PP-aware MoE placement 与 routing workspace
- [ ] 精确的 request-dependent multimodal activation peak

## License

MIT
