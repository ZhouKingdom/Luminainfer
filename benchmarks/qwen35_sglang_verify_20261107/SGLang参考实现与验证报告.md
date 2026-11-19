# SGLang参考实现与验证报告

参考本地SGLang提交 `e2982985110fb432fd8ddba5f0147a5004fd03ab`，选定文件无本地修改；SHA256见参考源码记录.json。没有运行SGLang服务性能对比。

## 参考了什么

- SGLang的TritonGDNKernel.target_verify：禁止直接更新持久S，窗口递推在一个kernel中完成，缓存各步中间状态，再按接受边界提交。
- GDN后端的causal_conv1d_update验证分支：批量处理短窗口、保存中间conv窗口。
- Qwen3.5模型还合并qkv/z及b/a投影，草稿worker有独立Graph。Lumina本轮只实现短窗口卷积与递推融合，保留默认GEMM/norm及NCCL算术，并未完整复刻这些优化。

源文件：`/home/jhzhou/sft_prj/sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`、`kernels/ops/attention/linear/fused_sigmoid_gating_recurrent.py`、`srt/layers/attention/linear/gdn_backend.py`、`srt/models/qwen3_5.py`。后三条相对于同一python/sglang目录。

## Lumina改动

新增可选 `verification_fused_gdn=True`，用于1/2行验证窗口。自实现Triton卷积窗口与递推kernel，读取请求私有初始状态，递推时不重读上一时刻S，所有中间S/conv写入scratch，接受后沿用原提交与故障回滚。保持BF16卷积输出舍入边界、FP32状态及原递推算术顺序；FP32/其他卷积宽度保留原卷积准备路径，更宽验证窗口回退原实现。默认关闭。

Graph持有独立cu_seqlens缓冲，重放更新动态元数据；融合与原路径分别缓存，避免切换开关后误用旧Graph。

## 严格逐行算术对照

真实27B、TP2、BF16、默认decode_equivalent；每配置8条输入，C1每条32输出，C4每条64输出。完整预热、两轮反序、相同prefill，同引擎内开关融合。单位为prefill后output tok/s。

| 并发 | 同算术普通Graph | 原批量草稿MTP | 融合后MTP | MTP路径提升 | 原免费草稿 | 融合免费草稿 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32.944 | 25.187 | 26.367 | +4.68% | 31.923 | 33.500 |
| 4 | 35.066 | 30.904 | 33.625 | +8.80% | 34.754 | 38.047 |

本节全部模式输出逐token一致。免费草稿来自离线参考、全部接受、没有草稿模型成本，只用于诊断，不可当作真实MTP成果。这里的普通Graph刻意固定C1算术，与关闭MTP的普通批量引擎并非同一路径。

## 实际普通引擎与批量算术实验

关闭MTP、保持实际普通批量Graph，同一C4输入/64输出、预热后两轮：解码 **123.081 tok/s**，E2E **101.271 tok/s**。完整输出相对逐行参考为4/8一致，跨算术路径差异保留。这个结果说明战胜逐行参考不能等价于战胜实际普通推理。

| batched实验模式（C4） | 解码 tok/s | E2E tok/s | 相对本后端普通Graph的完整序列一致数 |
|---|---:|---:|---:|
| 普通Graph | 120.685 | 100.052 | 8/8 |
| 原无草稿验证 | 79.802 | 70.555 | 8/8 |
| 原批量草稿MTP | 93.055 | 80.568 | 4/8 |
| 融合无草稿验证 | 86.371 | 75.582 | 8/8 |
| 融合MTP | 104.209 | 88.688 | 4/8 |

融合相对原批量草稿MTP：+11.99%；相对本后端普通Graph：-13.65%。融合前后MTP完整输出与接受计数相同，但batched后端自身相对普通解码已有输出差异，不能作为默认无损加速交付。

## 正确性、局部计时与证据

- 完整融合相关CUDA回归142项通过，额外1项Graph开关切换测试通过；覆盖BF16卷积、每步FP32 S、中间conv、混合长度、动态槽位/页表、Graph及旧数值路径。TP2/4/8分布式接受/回滚测试全部通过并正常退出。
- 最新完整融合/批量后端/实际普通控制正式共320次生成、16896个输出token。严格逐行后端输出全部匹配；实验batched差异逐项保留。递推首版试验独立保存在final_tp2_c*，不混入这些统计。
- 递推kernel隔离CUDA Graph计时：C1 9.50→4.91微秒（1.93×）；C4 31.82→10.68微秒（2.98×）。仅准备好的q/k/v/g/beta递推，不含投影、conv、通信或调度，不能当端到端收益。
- 两轮速率用累计token/累计时间合并；同引擎各模式顺序反转，加载/JIT/Graph捕获不计时，全rank有限值检查计时。各实验使用不重叠GPU组，CPU共享，非隔离环境；实际普通控制是单独引擎，避免拿微小差异作强结论。
- 原始结果、逐轮输出与各rank计数：full_tp2_c*/result.json、batched_tp2_c4/result.json、ordinary_tp2_c4/result.json。每组有冻结源码与SHA256。

## 后续重点

继续参考SGLang的合并投影、批量attention和草稿Graph，减少默认逐行路径的成本；批量算术路径需单独完成输出差异与质量验收。仍以实际普通批量Graph为性能目标，以相同后端的普通解码为正确性控制。没有因为局部内核快了就宣布投机已经有效，也未宣称HF严格logits验收通过。

## 复现命令

使用现有lumina解释器，从Lumina根目录运行；先检查GPU空闲，输出目录必须是新的，端口不得与其他引擎冲突。

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. /home/data/jhzhou/conda_envs/lumina/bin/python benchmarks/bench_mtp_net_gain.py --tp 2 --tokens 64 --repeats 2 --oracle --batch-draft --compare-fused-gdn --port 2542 --output benchmarks/融合复测_新运行
```

测试batched后端时去掉--oracle并增加--backend batched；C1增加--capacity 1 --tokens 32。实际普通引擎控制用benchmarks/bench_qwen35_ordinary_control.py的--workload指向对应result.json，并指定新的--output及独立--port。
