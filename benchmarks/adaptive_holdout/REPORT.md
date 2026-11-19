# 自适应调度：混合长度与突发流量对照

8×A100，两个四副本策略组。seed=29；输入 64/128/512/1024、输出 32/64/128 token；每场景每策略两轮各 256 请求，第二轮交换 GPU0–3/GPU4–7。计划到达序列、请求和轮询路由完全相同。
闭环每卡并发16；Poisson 聚合2 req/s；突发每2s同时到达8请求（聚合4 req/s）。引擎原生接口，无 HTTP/网络。

| 场景 | 策略 | tok/s | goodput req/s | SLO达标率 | TTFT p99 ms | TPOT p99 ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| burst | adaptive | 249.1 | 1.425 | 40.82% | 1814.6 | 117.6 | 536.1 |
| burst | alternating | 249.2 | 1.480 | 42.38% | 848.6 | 116.2 | 534.5 |
| closed | adaptive | 380.7 | 0.886 | 16.60% | 3851.4 | 167.6 | 581.9 |
| closed | alternating | 399.4 | 0.634 | 11.33% | 3842.1 | 168.0 | 541.6 |
| open | adaptive | 120.3 | 1.583 | 93.95% | 508.2 | 106.9 | 315.9 |
| open | alternating | 120.3 | 1.570 | 93.16% | 508.2 | 106.7 | 315.3 |

## 每轮收益与硬件交叉

| 场景 | 轮次 | baseline GPU | adaptive GPU | goodput变化 | 吞吐变化 | 输出hash一致 |
|---|---:|---|---|---:|---:|---:|
| burst | 0 | [0, 1, 2, 3] | [4, 5, 6, 7] | -1.9% | -0.0% | 202/256 |
| burst | 1 | [4, 5, 6, 7] | [0, 1, 2, 3] | -5.5% | -0.1% | 209/256 |
| closed | 0 | [0, 1, 2, 3] | [4, 5, 6, 7] | +37.6% | -5.0% | 142/256 |
| closed | 1 | [4, 5, 6, 7] | [0, 1, 2, 3] | +41.8% | -4.3% | 147/256 |
| open | 0 | [0, 1, 2, 3] | [4, 5, 6, 7] | +0.5% | +0.0% | 215/256 |
| open | 1 | [4, 5, 6, 7] | [0, 1, 2, 3] | +1.2% | -0.1% | 221/256 |

## 口径与限制

- 预设 SLO：每请求 TTFT≤2s 且平均TPOT≤100ms。没有按结果调整阈值。
- 汇总吞吐/goodput为总token/总达标请求数除以两轮各自makespan之和；p99从合并的逐请求/逐token数据重新计算，不平均各轮p99。
- 每策略每场景合计512请求，p99仍为探索性数据；只有两轮交叉，不提供统计显著性结论或生产SLA。重复同一manifest，不是独立语料样本。
- CPU 元数据缓存对两种策略都开启。adaptive只改变执行顺序和prefill分块，不修改模型算术；batch/chunk形状变化仍会改变BF16输出，不能宣称跨策略逐位一致。
- 执行预算是不可抢占前向的耗时预测，不是硬实时保证。负载超过目标SLO可承受能力时，decode余量与TTFT保护可能冲突。
- 默认保持alternating；adaptive显式启用。需要结合所有场景、吞吐代价和HF审计判断，不能只挑goodput提升项。

原始请求、逐步调度决策、manifest、源码快照与SHA256均保存在本目录。完整分层指标见summary.json；策略分歧见divergence_cases.json。

## 策略作用与采用结论

决策计数：{'ordinary_step': 19655, 'fill_decode_batch': 789, 'ttft_guard': 245, 'earn_decode_credit': 1900, 'credit_fit': 1}。本轮真正按credit选择128-token配额仅1次；其余主要是延后prefill直到decode有余量或TTFT保护触发。收益主要来自时序变化，不能把它包装成预算预测全面奏效。
饱和闭环goodput两轮分别+37.6%/+41.8%，汇总约+39.7%，但吞吐约−4.7%，SLO达标率仅11.3%→16.6%，TTFT/ITL p99没有改善。突发goodput两轮均下降，汇总约−3.7%，TTFT p99约0.85s→1.81s。Poisson收益不足1%，不能据此宣称显著改善。
采用结论：默认alternating与已验证的CPU元数据优化保持不变；adaptive为未通过完整验收的实验选项，不能当作已验收生产策略或写成全面性能提升。

## 数值审计（未通过验收）

共有400处跨策略输出分歧。按输入长度分层抽查16处；相同历史的HF单条完整前向参考中，candidate regret比baseline高出1以上的有7处，反方向有2处。这些分歧不能全部用近邻tie解释；既有baseline也存在大偏差，不据此断言调度是唯一根因。
审计只覆盖抽样首次分歧，不能替代整模型raw logits验收。具体case、prefix与候选分数见hf_divergence_audit.json和divergence_cases.json。
HF自身batch1完整前向、batch2完整前向、batch1缓存增量前向的参考top1在11/16个抽样历史上不一致；所有参考均认为candidate相对baseline regret高出1以上的仍有3处。参考路径敏感性不能用于豁免剩余误差，详见hf_self_reference_audit.json。

## 回归

最终回归44 passed、1 failed。保留的失败为Qwen3-0.6B test_batch_equals_individual：首个偏差已定位到第一次decode首层QKV GEMM的batch形状变化，输入相同而输出有0.001953125差异，最终在同分候选处分叉；新增同batch形状的请求重排隔离测试通过。原断言未放宽。详细日志/JUnit在../adaptive_regression.log与../adaptive_regression.xml。
