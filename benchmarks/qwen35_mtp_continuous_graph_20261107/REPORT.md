# 多卡验证 Graph 与连续补位验收

162项相关CUDA回归通过，无skip。TP2/4/8分布式小模型均验证Graph动态请求、混合接受[1,0]、单rank准备/提交/草稿/Graph缓冲分配错误后的回滚和继续运行。

真实Qwen3.5-27B、8×A100-80GB、TP2/4/8、默认decode_equivalent、BF16、eager普通decode、显式验证Graph。8条输入长度36/25/106/128/64/96/255/256，输出预算8/48/16/32/12/24/9/17，总输出166 token/阶段，最大并发4。
共160次正式生成、3320个输出token，所有请求回收页和状态槽。96次连续MTP（eager、Graph冷启动、Graph预热后两轮）全部与相同TP普通解码逐token一致。另24次固定cohort使用对应普通控制。所有rank每次有效logits输出检查有限。

| TP | 普通eager E2E | 连续MTP eager E2E | Graph冷启动E2E | 连续MTP Graph预热后E2E | Graph布局数 |
|---:|---:|---:|---:|---:|---:|
| 2 | 10.047 | 9.464 | 10.129 | 23.048 | 8 |
| 4 | 10.110 | 8.628 | 13.008 | 26.685 | 5 |
| 8 | 8.886 | 8.183 | 9.503 | 26.263 | 8 |

单位output tok/s。Graph预热后合并两轮wall time计算；捕获成本单列在冷启动阶段，不计入预热后数字。E2E含prefill和全rank有限值检查。TP2/4分组运行、TP8随后测量，早期小模型验证在退出时留有空闲CUDA上下文（后已修复并清理）；共享主机非隔离环境，这是小集功能与延迟负载，不是C128容量或投机算法独立收益评测。

每个连续阶段均记录第5个请求在最初4个请求全部完成前开始生成；固定cohort对应控制为False。分布式各rank的Graph布局/重放计数完全一致，包含混合1/2行尾部窗口；精确动态slots/pages/positions和页255–257由CUDA单测覆盖。

TP4初始固定cohort与连续普通基线有2条跨调度输出差异，保留final_tp4失败记录。新增同cohort普通解码控制后严格比较通过；未放宽连续Graph/eager逐token验收。见INITIAL_NOTES.md。

每组源码快照、SHA256、完整输出、逐轮接受、调度轨迹和资源/rank元数据随result.json保存；报告逐文件核对冻结源码SHA256；当前计算路径的AST与正式快照一致。计时后只加强了exit里的gc回收/通信同步，并改了两处说明文字；exit在E2E计时外，已用实际生产exit重跑TP2/4/8故障恢复测试并确认全部子进程退出。

使用和边界见 [MTP_CONTINUOUS_GRAPH.md](../../MTP_CONTINUOUS_GRAPH.md)。仍greedy、MTP-1、显式1–8请求；FP8/更宽窗口/超过16种布局回退eager。本轮真实容量4，未宣称C8/C128验收或HTTP服务自动MTP。
