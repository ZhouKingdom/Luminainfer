# Qwen3.5-27B 多副本容量与尾延迟实测

日期：2026-11-01。A100-SXM4-80GB；每 GPU 独立单卡文本模型，eager、BF16 权重与 FP32 recurrent state。该实验不新增模型并行。

固定输入 128 tokens、输出 64 tokens；每副本 128 条请求；greedy、ignore_eos；token budget=1024，max_num_seqs=32。
引擎原生接口计时，未包含 HTTP/SSE、外部网络、tokenization、模型加载和预热。prompt 来自固定句式的自然语言合成文本，不是真实生产 trace。

## 已完成场景

| 场景 | GPU 副本 | 总并发 | 请求数 | output tok/s | TTFT p99 | TPOT p99 | ITL p99 | SLO 达标率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| closed_r1_c8 | 1 | 8 | 128 | 82.7 | 880.6ms | 85.9ms | 87.3ms | 100.0% |
| closed_r2_c8 | 2 | 16 | 256 | 163.9 | 1079.5ms | 93.4ms | 87.8ms | 100.0% |
| closed_r4_c8 | 4 | 32 | 512 | 324.0 | 1087.7ms | 93.9ms | 90.7ms | 100.0% |
| closed_r8_c8 | 8 | 64 | 1024 | 649.6 | 1097.3ms | 95.7ms | 117.4ms | 100.0% |
| closed_r8_c16 | 8 | 128 | 1024 | 1157.5 | 2166.8ms | 102.8ms | 879.3ms | 93.8% |
| open_r8_rate8 | 8 | open-loop | 1024 | 468.4 | 216.1ms | 102.1ms | 219.3ms | 93.5% |

## 扩展性口径

closed-loop 每副本保持 8 个在途请求，完成后补充新请求；每副本请求数、输入/输出长度相同，这是副本容量的弱扩展测试。按共同起点到最后请求完成的总 makespan 计算聚合吞吐，不相加各副本局部吞吐。它不是单请求 8 卡加速，也不是 TP/EP scaling。

- 1 GPU：1.000× 单卡容量，效率 100.0%。
- 2 GPU：1.981× 单卡容量，效率 99.1%。
- 4 GPU：3.917× 单卡容量，效率 97.9%。
- 8 GPU：7.852× 单卡容量，效率 98.2%。

## 延迟与 SLO 定义

固定每卡并发 8 的扩展实验中，重复请求输出 hash：896 次相同、0 次不同；不代表 HuggingFace 全模型精度验收。
聚合 open-loop 到达序列为 Poisson，按请求序号轮询分发到副本；不把各副本的到达序列当成独立 Poisson。
- TTFT：首 token 时刻减计划到达/提交时刻。open-loop 使用预先生成的 Poisson 计划到达时间，包含引擎忙碌造成的实际提交滞后；该滞后另行记录，不能从延迟中扣掉。
- TPOT：每请求 (最后 token 时刻−首 token 时刻)/(输出 token 数−1)。ITL 为逐 token 间隔，p99 对整个 phase 的间隔池取分位数，间隔并非相互独立样本。
- SLO 是本实验预设门槛 TTFT≤2s 且 TPOT≤100ms，不是已验证的生产 SLA。SLO goodput=同时满足两项门槛的请求数/总 makespan。
- 每副本/小规模档位少于 1000 条请求，p99 标为 exploratory。8 副本有 1024 条请求，仍只有单次实验，没有重复试验置信区间。
- open-loop 采用固定总到达速率，吞吐可能受 arrival rate 限制；不能当作饱和吞吐峰值。
- 各场景分别报告指标；不能把高并发峰值吞吐与低并发低尾延迟拼成同一个场景。

## 基线限制

现有 verl 环境为 vLLM 0.11.0，其模型注册表未支持 Qwen3.5，因此本轮没有同模型 vLLM 对照；不使用 Qwen3 的数据代替它，也不宣称相对 vLLM 提升。

## 原始证据

- summary.json：运行参数、GPU/状态槽/缓存容量与各场景汇总。
- manifest.json：确定性请求清单；*_r*_*.jsonl：每请求计划/实际提交、完成时刻、全部 token 时间戳与输出 hash。
- results.csv 与 *_latencies.csv：可复核的聚合/逐请求结果。
- source_snapshot/sha256.json 与 environment.json：本次实现快照与依赖版本。

复现命令见 summary.json 的 config；使用 benchmarks.bench_replica_service。生成本报告：python -m benchmarks.report_replica_service benchmarks/service_8gpu.
