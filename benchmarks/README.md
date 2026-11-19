# Benchmark：zjh-vllm vs 官方 vLLM 吞吐对比

> 本文下方为早期 Qwen3/旧 vLLM 实验与脚本记录，不能作为当前 Qwen3.5 的对照结论。当前方法见 [统一基准说明](qwen35_baseline_20261112/基准与Profiling实施说明.md)，结果见 [Qwen3.5 矩阵](qwen35_auto_20261112/实验汇总.md) 和 [profiling 汇总](qwen35_profile_retry_20261114/实验汇总.md)。

**新增 Qwen3.5-27B 八卡实验**：见 [service_8gpu/REPORT.md](service_8gpu/REPORT.md)。该实验测试独立单卡副本容量和精确逐请求延迟，不是 TP/EP 扩展，也不是与 vLLM 的对照。现有 vLLM 0.11.0 未注册 Qwen3.5。

方法提醒：下面历史 trace 脚本在 Lumina 使用 open-loop、在 vLLM 使用 closed-loop，二者的排队时序不同，不能直接用这组 TTFT p99 判断引擎优劣；闭环吞吐和并发 sweep 应单独按对应条件解读。

两个脚本用**完全相同的 workload**（`seed(0)`，256 条序列，input/output 各 100~1024 随机 token），保证对比公平。总生成 token 一致（133966），吞吐 = 总 token / 总时间。

## 用法

zjh-vllm（在 `lumina` 环境）：

```bash
conda activate lumina
python benchmarks/bench_zjh.py ~/huggingface/Qwen3-0.6B 1        # 0.6B 单卡
python benchmarks/bench_zjh.py /path/to/Qwen3-32B 4              # 32B TP=4
```

官方 vLLM（在 `verl` 环境，vLLM 0.11.0）：

```bash
conda activate verl
export CUDA_HOME=/usr/local/cuda-13.2   # 否则默认 /usr/bin/nvcc(CUDA 11.5) 编译 extension 报 math.h 找不到
export PATH=/usr/local/cuda-13.2/bin:$PATH
python benchmarks/bench_vllm.py ~/huggingface/Qwen3-0.6B 1
python benchmarks/bench_vllm.py /path/to/Qwen3-32B 4
```

> 注意：vLLM 需要 `CUDA_HOME` 指向 CUDA 13.2（系统默认 `/usr/bin/nvcc` 是 CUDA 11.5，与 gcc 11.4 不兼容）。

## 实测结果（8×A100 80GB）

| 配置 | zjh-vllm | vLLM 0.11.0 | 对比 |
|---|---|---|---|
| Qwen3-0.6B，单卡 | 10368 tok/s | 8159 tok/s | +27% |
| Qwen3-32B，4×A100（TP=4） | 2997 tok/s | 3069 tok/s | -2.4%（达 97.6%） |

- **32B 大模型（compute-bound）**：zjh-vllm 达到 vLLM 吞吐的 ~98%，贴近工业级水平。
- **0.6B 小模型（调度 overhead 占比高）**：zjh-vllm 反而快 27%（简化调度器开销更低）。

## 延迟指标（TTFT / TPOT / p99）

除吞吐外，另有两个脚本测量延迟，workload 与吞吐脚本完全一致（256 条并发、随机 token、input/output 各 100~1024）：

```bash
# zjh-vllm（lumina 环境）
python benchmarks/bench_latency_zjh.py ~/huggingface/Qwen3-0.6B 1
python benchmarks/bench_latency_zjh.py /path/to/Qwen3-32B 4

# vLLM（verl 环境，延迟时间戳来自 RequestMetrics）
python benchmarks/bench_latency_vllm.py ~/huggingface/Qwen3-0.6B 1
python benchmarks/bench_latency_vllm.py /path/to/Qwen3-32B 4
```

指标定义：

| 指标 | 含义 | 计算 |
|---|---|---|
| TTFT | Time To First Token | 请求提交 → 首个 completion token |
| ITL | Inter-Token Latency | 相邻 completion token 的生成间隔 |
| p50 / p90 / p99 | 分位数（尾延迟） | TTFT、ITL 各自的分位 |

zjh 侧的时间戳由引擎记录（`Sequence.arrival_time/first_token_time/finished_time/token_times`，见 `engine/llm_engine.py::latency_stats`），可复用于 OpenAI 服务层；vLLM 侧读 Prometheus histogram（`vllm:time_to_first_token_seconds` / `vllm:inter_token_latency_seconds`）。

### 延迟实测（Qwen3-0.6B，单卡，256 条并发）

| 指标 | zjh-vllm | vLLM 0.11.0 |
|---|---|---|
| TTFT | mean 501ms / p50 519ms / **p99 916ms** | mean 690ms / p50 672ms / **p99 2429ms** |
| ITL | mean 14.5ms / p50 15.0ms / **p99 16.5ms** | mean 18.7ms / p50 17.3ms / **p99 61.7ms** |

- **TTFT / ITL 均更快**，尤其 p99 尾延迟（zjh TTFT p99 低 62%、ITL p99 低 73%）。
- zjh 的 ITL p99（16.5ms）贴近 mean（14.5ms），说明 token 生成间隔非常均匀；vLLM 的 ITL p99（61.7ms）相对 mean（18.7ms）明显偏高，反映更复杂的调度/preemption 交互带来的尾延迟。
- 这与吞吐结论一致：0.6B 小模型下 zjh 的简化调度器 overhead 更低（吞吐 +27%）。

> 诚实提醒：这是固定随机负载的 offline batch，256 条同时到达。TTFT 的绝对量级由「所有请求同时排队 prefill」决定；真实在线负载下请求逐步到达，TTFT 会低得多。ITL 则更能反映 steady-state 的 decode 间隔。

## 真实负载 trace（重尾长度 + 共享前缀 + Poisson 到达）

针对旧 bench 的三大不足——**输入长度均匀随机**（非重尾）、**无共享前缀**（前缀缓存收益没被测到）、**同时到达**（TTFT 虚高）——新增共享 workload 生成器 [workload.py](workload.py)，引入三个真实负载要素：

| 要素 | 旧 bench | 新 workload |
|---|---|---|
| 输入/输出长度 | 100~1024 均匀随机 | log-normal 重尾（短 prompt 为主、少数超长） |
| 前缀 | 无 | 共享 512-token system prompt（触发 prefix caching） |
| 到达模式 | 256 条同时涌入 | Poisson 到达（open-loop，陆续到达） |

```bash
# zjh-vllm：open-loop，默认 2000 条（样本量 256 → 2000+，p99 更稳）
python benchmarks/bench_trace_zjh.py ~/huggingface/Qwen3-0.6B 1 --num-seqs 2000 --rate 50
python benchmarks/bench_trace_zjh.py /path/to/Qwen3-32B 4 --num-seqs 2000 --rate 50

# 关掉前缀做 abl 对比（量化 prefix caching 收益）
python benchmarks/bench_trace_zjh.py ~/huggingface/Qwen3-0.6B 1 --num-seqs 2000 --no-prefix

# vLLM 基线：closed-loop（离线 LLM 无 add_request/step，无法模拟到达）
python benchmarks/bench_trace_vllm.py ~/huggingface/Qwen3-0.6B 1 --num-seqs 256 --prefix-len 512
```

说明：
- zjh 侧用引擎原生 `add_request`/`step` 循环按到达时刻**陆续提交**（open-loop），消除「同时到达」的 TTFT 虚高，是引擎本身的能力（复用 `Sequence.arrival_time` 等时间戳）。
- vLLM 离线 `LLM` 只有 `generate()`（closed-loop），所以对比的是「重尾 + 前缀」两个 workload 要素；到达模式只在 zjh 侧体现。
- open-loop 下吞吐是 **arrival-limited**（到达速率决定），关注点是 TTFT/ITL 的 p99 尾延迟，而非吞吐绝对量。

### 真实负载实测（Qwen3-0.6B，单卡，2000 条，Poisson rate=50/s）

| 配置 | TTFT p99 | ITL p99 | arrival-limited 吞吐 |
|---|---|---|---|
| 共享前缀 512 tok | 39.1ms | 39.3ms | 5063 tok/s |
| 无前缀（abl） | 35.7ms | 35.8ms | 5118 tok/s |

- **TTFT p99 从旧「256 条同时到达」的 916ms 降到 39ms**：证明旧数字是「同时排队 prefill」的 artifact，真实到达模式下尾延迟低一个数量级。
- 2000 条样本让 p99 更稳定（对比旧 256 条，p99 噪声大）。
- 前缀 abl 在 rate=50/s（arrival-limited、引擎未饱和）下收益≈0：低负载时引擎有余量，前缀缓存省下的 prefill 算力体现不出来。**前缀缓存收益要在饱和负载下才显现**，见下方「前缀缓存收益」的公平 abl（+53% 吞吐、p99 降 71%）。

## 并发扫描（sweep）：吞吐 / TTFT vs 并发数

每个并发档位 closed-loop 提交 C 条请求，测吞吐 + TTFT p50/p99，画吞吐与尾延迟随并发的变化趋势：

```bash
python benchmarks/bench_sweep_zjh.py ~/huggingface/Qwen3-0.6B 1 --levels 8,16,32,64,128,256,512
python benchmarks/bench_sweep_vllm.py ~/huggingface/Qwen3-0.6B 1 --levels 8,16,32,64,128,256,512
```

- zjh 侧延迟用 `latency_stats()` 直接读；vLLM 侧 histogram 是累计的，脚本做「bucket 快照差」得到每档位增量分位数。
- 小并发档位总 token 少、运行时间短，吞吐噪声大，看趋势即可。
- 每个档位独立 seed，避免跨档位命中前缀缓存。

### 32B 并发扫描实测（Qwen3-32B，4×A100 TP=4）

| 并发 | zjh 吞吐 | vLLM 吞吐 | zjh TTFT p50 | vLLM TTFT p50 | zjh TTFT p99 | vLLM TTFT p99 |
|---|---|---|---|---|---|---|
| 8 | 178 | 194 | 274ms | 357ms | 274ms | 497ms |
| 16 | 277 | 315 | 309ms | 158ms | 309ms | 248ms |
| 32 | 317 | 358 | 579ms | 205ms | 579ms | 492ms |
| 64 | 557 | 640 | 1113ms | 222ms | 1114ms | 994ms |
| 128 | 1102 | 1267 | 1283ms | 605ms | 2042ms | 2454ms |
| 256 | 1781 | 2038 | 1288ms | 690ms | **4108ms** | **4861ms** |

- 重尾短请求 workload 下 vLLM 吞吐略高（zjh ≈ 87%），但**高并发时 zjh 的 p99 尾延迟更低**（256 并发 4108ms vs 4861ms），调度器的简化 overhead 在尾延迟上反而占优。
- 注意：这里的绝对吞吐（1781 tok/s）比旧 bench（2997 tok/s）低，是因为 workload 输入:输出比不同（重尾短输出 → prefill 占比高，而「吞吐」分母只算 output token）。**吞吐数字强依赖 input:output 比例**，跨 workload 不能直接比。

### 前缀缓存收益（公平 abl：等长 1024-token 前缀，共享 vs 各不同）

| 配置 | 吞吐 | TTFT p50 | TTFT p99 |
|---|---|---|---|
| 共享前缀（缓存命中） | 6665 tok/s | 495ms | 681ms |
| 唯一前缀（无缓存，同长度） | 4348 tok/s | 1198ms | 2357ms |

- 前缀缓存带来 **+53% 吞吐、TTFT p99 降 71%**（等长对比，隔离了「请求变长」的干扰）。
- 关键：前缀缓存省的是「重算共享前缀 KV 的 prefill 算力」，在**饱和 + prefill 占比高**的负载下收益最大；decode 阶段仍要 attend 前缀 token，所以对纯 decode 密集负载收益有限。

## 复现要点

1. 两个脚本的 workload 生成代码必须逐字一致（`seed(0)` + 相同的 `randint` 调用顺序）。
2. 都用 CUDA graph（`enforce_eager=False`）。
3. 相同 TP。
4. 先 warmup 一次再计时（排除 CUDA graph 捕获、首次 kernel 编译的开销）。

## 自研 paged decode attention 性能

自研 Triton decode kernel（`zjhvllm/layers/paged_attention.py`）替代 `flash_attn_with_kvcache`，正确性与性能分别由测试和 benchmark 验证：

```bash
# 正确性：与 flash_attn_with_kvcache 逐元素对齐（bf16 噪声阈值内），覆盖 GQA 2 头/KV 和 8 头/KV
pytest tests/test_paged_attention.py -q

# 性能：32B GQA 配置（64 Q 头 / 8 KV 头 / head_dim 128），扫 batch × ctx
python benchmarks/bench_paged_attention.py
```

ratio = 自研 / flash，越小越好（<1 表示反超）。当前（v2）数字是 `bench_paged_attention.py` 可复现的：

| case | v1（逐 Q 头 naive） | v2（GQA shared-KV + tensor core） |
|---|---|---|
| batch 256 / ctx 4096 | 8.84x | **1.05x** |
| batch 128 / ctx 1024 | 4.87x | **1.12x** |
| batch 32 / ctx 4096 | 3.42x | **0.85x** |

- batch ≥ 32（真实连续批处理区间）全程 0.85–1.12x，追平/反超 flash-attn。
- 剩余差距在小 batch 长 context（batch=1 / ctx=4096 约 3–6x）：只有 `num_kv_heads=8` 个 program 喂 108 SM，是**并行度问题**，不是算法问题。
- 曾尝试 split-K 并回退（对 memory-bound decode 不划算），负结果记录在 `paged_attention.py` 的 module docstring。

> v1 → v2 的演进是一次「诊断驱动的性能优化」：定位到「GQA 没复用 KV 读取（8 个 Q 头各读一遍同一 KV）+ 整页 256 tile 撑爆寄存器」，然后对症改 grid 和 tile。详细拆解见 [TUTORIAL.md](../TUTORIAL.md) §3.13。

## 独立单卡副本压测复现

```bash
conda activate lumina
python -m benchmarks.bench_replica_service \
  --gpus 0 1 2 3 4 5 6 7 --replicas 1 2 4 8 \
  --concurrencies 8 --extra-concurrencies 16 \
  --requests-per-replica 128 --input-tokens 128 --output-tokens 64 \
  --rates 8 --output-dir benchmarks/service_8gpu
python -m benchmarks.report_replica_service benchmarks/service_8gpu
```

脚本需从 Lumina 根目录运行。各 worker 使用独立 CUDA_VISIBLE_DEVICES 与 distributed_port；每卡 max_num_seqs=32，token budget=1024。请先检查 GPU 占用；8 个 27B 副本会占用全部八卡的大部分显存。计时结束退出后释放显存。

吞吐按共同 makespan 计算；open-loop TTFT 从计划到达时刻起算，保留负载生成滞后。汇总包含 TTFT、TPOT、ITL、SLO goodput 和样本量标记，JSONL 保留逐请求 token 时间戳。
