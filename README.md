# zjh-vllm

一个从零实现的轻量级 LLM 推理引擎，参考 [vLLM](https://github.com/vllm-project/vllm) 的架构思想，用 PyTorch + Triton 手写核心算子。目标是理解并复现工业级推理引擎的关键技术。

> 支持 Qwen3（dense）和 Qwen3-MoE（稀疏专家）架构，通过 `models/registry.py` 按 `model_type` 分发，可扩展其他模型家族。

## 特性

| 能力 | 说明 | 位置 |
|---|---|---|
| 连续批处理 (Continuous Batching) | chunked prefill + 抢占式调度，多请求共享一次前向 | `engine/scheduler.py` |
| Paged Attention | 分块 KV cache，block table 索引，前缀缓存（hash 复用） | `engine/block_manager.py` |
| 张量并行 (TP) | NCCL + 子进程，支持 tp=1..8 | `layers/linear.py` |
| CUDA Graph 解码 | 按 batch 分桶捕获解码图，降低 launch 开销 | `engine/model_runner.py` |
| 手写 Triton 采样器 | temperature/top_k/top_p/repetition_penalty + Gumbel-max + Philox 可复现 | `layers/sampler.py` |
| Flash Attention | varlen（prefill）+ kvcache（decode） | `layers/attention.py` |
| FP8 KV cache | e4m3 量化 + per-token scale，KV 显存减半 | `layers/attention.py` |
| MoE（稀疏专家） | router + top-k 专家选择 + 稀疏激活 FFN | `layers/moe.py` |
| 专家并行（EP） | 专家按卡分布 + all-to-all 路由（router replicated） | `layers/ep_moe.py` |

## 环境

```bash
conda create -n lumina python=3.12
conda activate lumina
# 服务器不通外网时，先走端口转发代理：
export http_proxy=http://127.0.0.1:17988 https_proxy=http://127.0.0.1:17988 all_proxy=http://127.0.0.1:17988 no_proxy=localhost,127.0.0.1
pip install torch==2.8.0 triton transformers flash-attn xxhash
pip install pytest fastapi uvicorn   # 测试 + 服务
```

依赖版本：`torch 2.8.0(+cu128)`、`triton 3.4.0`、`transformers`、`flash-attn 2.8.3.post1`（cxx11abiTRUE）、`xxhash`。

> 注意：torch 2.8.0 用的是 `_GLIBCXX_USE_CXX11_ABI=True`，flash-attn 需选 `cxx11abiTRUE` 的 wheel。

## 快速开始

```bash
# 示例（Qwen3-0.6B）
python example.py

# 压测（Qwen3-32B，需要 TP）
python bench.py
```

模型路径：`~/huggingface/Qwen3-0.6B`（小模型，单卡即可）。`bench.py` 用 32B，需 `tensor_parallel_size=4`（单卡 KV cache 容不下 256 条并发序列，见 `OPTIMIZATION_ROADMAP.md`）。

### FP8 KV cache

```python
from zjhvllm import LLM
llm = LLM(model_path, kv_cache_dtype="fp8")  # 默认 "auto"=bf16
```

KV 用 e4m3 fp8 + per-token scale 存储，`flash-attn 2.x` 不支持 fp8 KV，因此 decode/prefill 读缓存时用 PyTorch 原生反量化 gather 成连续 bf16 再走 flash_attn。实测相同显存预算下 KV 容量翻倍（533 → 1062 blocks）。

### MoE（稀疏专家）

支持 Qwen3-MoE 架构（`model_type="qwen3_moe"`，经 `models/registry.py` 分发）：

- 单卡版：`layers/moe.py` 的 `MoE`（router + top-k 专家选择 + 稀疏激活）。
- 专家并行（EP）：`layers/moe.py` 的 `MoE` 在 `expert_parallel_size>1` 时走 all-to-all 路由（专家按卡分布、router replicated、权重按 rank 过滤加载）。
- 已用真实模型 **Qwen3-30B-A3B**（128 专家 top-8，30B 总参/3B 激活）跑通：TP=4 和 **EP=8**（每卡 16 专家）输出一致。

```python
llm = LLM("path/to/Qwen3-30B-A3B", tensor_parallel_size=4)      # 纯 TP
llm = LLM("path/to/Qwen3-30B-A3B", expert_parallel_size=8)      # 专家并行（num_key_value_heads=4 限制 TP≤4，上 8 卡需 EP）
```

EP 的设计见 [MOE_DESIGN.md](MOE_DESIGN.md)。

## 正确性测试

21 个测试覆盖：logits 与 HuggingFace 逐元素对齐、greedy/批处理/前缀缓存/chunked prefill/确定性、Triton 采样器与 numpy 参考对齐、fp8 KV cache、MoE 层与专家并行（EP）。

```bash
python -m pytest tests/ -v
```

测试设计要点（见 `tests/test_logits.py`、`tests/test_generate.py`）：
- **logits 对齐**：zjhvllm 用 flash-attn、HF 用 SDPA，两者有 ~0.1~0.4 的数值噪声，用 `max_abs_diff < 1.0` 区分「噪声」和「真 bug」。
- **greedy 近邻 tie**：当某步 top1-top2 gap 小于噪声时 argmax 会翻转、级联发散，这是预期行为。测试在分歧点验证 HF 的 top1-top2 gap 来区分近邻 tie 与真 bug。

## OpenAI 兼容服务

```bash
python server.py --model ~/huggingface/Qwen3-0.6B --enforce-eager
# 或 uvicorn server:app --host 0.0.0.0 --port 8000
```

端点：

| 端点 | 说明 |
|---|---|
| `GET /health` | 健康检查 |
| `GET /v1/models` | 模型列表 |
| `POST /v1/completions` | 文本补全（支持 `stream`） |
| `POST /v1/chat/completions` | 对话补全（chat template + `stream`） |

```bash
# 非流式
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":32,"temperature":0}'
# 流式
curl -sN http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"Hi","max_tokens":16,"temperature":0,"stream":true}'
```

架构：`zjhvllm/engine/async_llm_engine.py` 的异步引擎在后台线程跑 continuous batching 的 `step()` 循环，所有引擎访问集中在单线程（避免多线程竞争调度器/KV cache），HTTP 侧通过 thread-safe 队列提交请求、`loop.call_soon_threadsafe` 投递 token 流。

## 目录结构

```
zjhvllm/
  llm.py                 # LLM 入口（LLMEngine 子类）
  config.py              # 配置
  sampling_params.py     # 采样参数
  engine/                # 调度 / 执行
    scheduler.py         # 连续批处理调度
    block_manager.py     # paged KV cache + 前缀缓存
    model_runner.py      # 模型执行 + TP + CUDA graph + KV cache 分配
    llm_engine.py        # 同步引擎（add_request / step / generate）
    async_llm_engine.py  # 异步引擎（后台 step 循环 + 流式）
    sequence.py
  layers/                # attention / 线性 / norm / rope / sampler / 采样
  models/qwen3.py        # Qwen3 模型
tests/                   # 正确性测试
server.py                # OpenAI 兼容服务
bench.py / example.py    # 压测 / 示例
benchmarks/              # 与官方 vLLM 的吞吐对比（见 benchmarks/README.md）
```


