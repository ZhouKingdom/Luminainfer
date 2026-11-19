# Qwen3.5 进度

2026-11-19 补充：真实 TP1/2/4/8 启动及普通性能矩阵已执行；原队列 33 配置通过、28 因 vLLM 重复输出变化未通过，四组 profiling 补跑完成。后续应分析 trace 与输出重复性，不再将这些工作写成尚未启动。37 项基准/监测 CPU 回归通过，不等价于完整模型质量验收。公开汇总见 [README](README.md)。

更新：2026-11-12。范围是Qwen3.5-27B文本推理。使用方式见 [使用指南](使用指南.md)，所有数字与证据统一见 [测试与性能](测试与性能.md)。

当前方向（2026-11-12）：聚焦普通 BF16 推理，以官方 vLLM 为基线，补全可复现 benchmark、profiling 和优化前后证据。MTP、递归 MTP-k 与量化暂停研发，以下 MTP 能力均为历史已实现功能，不是当前优先工作。量化只有实验 W8A16 MLP 代码及 CPU 检查，尚无 CUDA/真实模型验收，不纳入性能成果。

新增共享 workload/双引擎离线 runner、请求级精确统计与结果配对检查、Nsight SQLite 内核汇总。独立 `/home/data/jhzhou/conda_envs/vllm` 已安装官方 0.26.0，PyTorch 2.11.0/CUDA 13.0、Transformers 5.22.2、Triton 3.6.0；依赖检查、编译扩展及 Qwen3.5 注册验证通过。正式 GPU 对照尚未开始，当前所有卡有其他计算进程。旧 verl/vLLM 0.11.0 未改动。方法与运行步骤见 [实施说明](benchmarks/qwen35_baseline_20261112/基准与Profiling实施说明.md)。

## 已完成

| 部分 | 当前状态 |
|---|---|
| 模型结构 | 64层：48层Gated DeltaNet、16层full attention；BF16权重/激活、FP32递归状态；适配norm、partial RoPE和query/gate布局 |
| 混合状态 | 按请求管理S/conv，分块prefill保留状态，结束释放槽，抢占后完整前缀重算 |
| 内核 | 长prefill使用chunk和Triton scan，decode原地更新递归状态；full attention使用paged KV |
| 真正TP | 普通推理和MTP均支持TP1/2/4/8、EP=1；不是多个独立单卡副本 |
| 普通Graph | 混合decode与NCCL捕获/重放，实际batch分桶，未捕获大小回退eager |
| 真实MTP-1 | 加载15组checkpoint参数，共享目标embedding/head，草稿KV独立且按局部KV头分配 |
| 递归MTP-k（可选） | `mtp_num_steps=1–8`，默认1；单层checkpoint递归传递草稿hidden，临时KV不跨轮保留；真实TP2已测K1/2/4，TP2/4/8的K4回滚通过 |
| 批量草稿（可选） | `mtp_batch_draft=True` 合并投影/head/预测通信，保留私有KV与变长输入；默认关闭，真实TP2 C1/C4已做严格输出对照 |
| 独立接受 | 各请求分别决定接受长度，不取全batch最小值；按各自边界提交S/conv、target hidden和KV元数据 |
| 多卡验证Graph | 支持多请求及混合1/2行窗口，动态更新token、position、状态槽、页表和有效长度 |
| 短窗口融合（可选） | `verification_fused_gdn=True` 合并最多9行卷积窗口与递推，保存各步S/conv；默认数值路径逐位对照通过，TP2/4/8回滚已测，真实吞吐只复测TP2 |
| 连续补位 | 批量接口默认启用，在线入口step_speculative；新请求无需等待整个cohort结束 |
| 回滚与退出 | 准备/提交阶段可恢复错误跨rank回滚；页不足回退普通decode；补齐异常引用回收与NCCL退出同步 |

GDN按key/value头切分，MLP和attention使用列/行并行。27B的4个KV头在TP8时按rank组复制。Embedding 与 LM head 已做词表并行切分（逐位一致），MTP fc 和必要 norm 仍复制。

## 已解决的关键问题

- **数值路径不一致：** 默认验证固定逐token的投影、norm、GDN递归和attention算术；多卡还固定BF16 NCCL行归约形状，避免仅固定GEMM仍出现输出分歧。
- **拒绝后的状态恢复：** 保存每个验证输入步的中间S/conv，按接受长度提交；常规MTP-1拒绝后不必整模型重放。
- **大状态池地址溢出：** slot乘stride前改为64位，测试覆盖超过2^31元素的地址跨度。
- **补位状态污染：** 复用槽时清零目标状态、重置草稿逻辑KV长度、重新计算有效hidden前缀；Graph不绑定最初请求。
- **低容量死锁：** 唯一decoder跨页时，先回收部分prefill持有的页，避免抢占自己后等待队首无法分配。

## 当前限制

- 只独立验收Qwen3.5-27B文本；视觉、随机拒绝采样、取消API和HTTP自动MTP尚未完成；K8仅做小模型窗口检查，真实模型只测K1/2/4。
- MTP显式并发上限1–8；真实模型已测C1/C2/C4，最新连续补位负载最大并发4，不宣称C8或更大容量已验收。
- 验证Graph默认缓存16种布局，可配置到64；FP8 KV、超过9行的窗口或布局超限回退eager。草稿、调度和提交仍在Graph外。
- 混合模型禁用仅KV的前缀复用；复用KV不足以恢复S/conv。未完成混合状态跨引擎迁移。
- 相同TP/后端/对应调度配置的严格行为对照已通过；HF严格raw logits尚未通过，不保证跨任意batch、TP或prefill调度逐位一致。
- `batch_invariant`是可选实验后端，不是默认；与旧cuBLAS有输出差异，多卡MTP/HF质量尚未独立验收。

## 下一步

1. 独立官方 vLLM 0.26.0 环境已安装并保存依赖清单；在空闲卡上完成 TP2 C1 的启动、长度与输出行为检查，确认实际状态精度和 Graph，再测 C32 性能锚点。
2. 普通 BF16 的短输入/长输入/长输出负载，逐步扩展 TP1/2/4/8 和 C1/8/32/128；报告 TTFT、TPOT、ITL、端到端吞吐、实际容量与显存，容量不足如实记录。
3. 独立采集代表性 prefill/decode 的 Nsight 时间线，分析 GDN、attention、GEMM、NCCL 与主机空隙；随后对热点使用 Nsight Compute，不用 profiling 吞吐代替正式性能。
4. 选择实测主要瓶颈，逐项做正确性与性能前后对照；保留回退、失败、版本和源码证据。
5. 扩大自然文本、长上下文、HF 与任务质量验收；跨引擎输出差异单独报告，不放宽旧失败阈值。

## 代码入口

| 位置 | 内容 |
|---|---|
| `models/qwen3_5.py`、`qwen3_5_mtp.py` | 混合模型、TP、数值路径和真实草稿层 |
| `layers/gated_delta.py` | Triton scan、decode和中间状态写入 |
| `engine/model_runner.py`、`verification_graph.py` | 分布式前向、缓存、Graph、验证和回滚 |
| `engine/llm_engine.py`、`scheduler.py` | 生成API、连续补位、分块prefill和抢占 |

以上代码路径均相对于 `zjhvllm/`。历史逐阶段说明已归档；本次用户要求的接手入口见 [接手续做说明](接手续做说明.md)。
