# Qwen3.5 官方 vLLM 基线与 Profiling

2026-11-12。当前阶段是基准基础设施与方法验收，尚未得到新的 GPU 对照数据或瓶颈占比。项目聚焦 Qwen3.5-27B 文本普通 BF16；MTP、量化暂停，历史代码和报告保留。

## 当前现场与完成状态

- 已新增统一脚本 `benchmarks/qwen35_baseline.py`：prepare/preflight/run/compare。两引擎独立环境和进程读取相同 workload，不修改官方 vLLM 源码。
- 已新增 `benchmarks/qwen35_profile_summary.py`：读取 Nsight SQLite 的 kernel、StringIds 表，按设备计算区间并集、kernel 时间线空隙、名称分类和热点。
- 已准备 `workload_128_128_C32.json`，32 条、每条 128 输入/128 输出，固定 token IDs；是重复/截断自然文本的合成性能负载，不代替自然语料质量测试。
- CPU 验证见 `CPU验证.log`、`基准最终验证.log`。模拟测试不证明 CUDA、NCCL 或 vLLM adapter 已真实可用。
- `Lumina环境.json`、`现有vLLM环境.json` 为只读现场证据。现有 verl：vLLM 0.11.0、Transformers 4.56.1，模型目录/注册表没有 Qwen3.5；lumina 没有 vLLM。Nsight Systems 2025.6.3 可用。
- 当前 8 张卡都有其他计算进程。没有启动 GPU 基准或干扰这些进程。脚本在初始化 CUDA 前检查所选卡；有计算进程就拒绝运行。
- 用户新建的 `/home/data/jhzhou/conda_envs/vllm` 已安装官方 vLLM 0.26.0；Python 3.12.16、Torch 2.11.0/CUDA 13.0、Transformers 5.22.2、Triton 3.6.0。`pip check` 通过，编译扩展 `_C_stable_libtorch`、Qwen3.5 模型模块/注册和基准参数导入验证通过；见 `vllm环境验收.json`、`vllm依赖锁定.txt`、`vllm依赖检查.log`、`新版vLLM环境_完整.json`。旧环境保持不变。
- 安装先遇到代理失效与下载超时，失败日志均保留；最终使用 uv 0.13.7、阿里云 PyPI 镜像，命令级清空失效代理后成功。新缓存位于新环境 `.uv-cache`，临时文件位于 `.install-tmp`，未挤占工作分区。
- 旧验证脚本曾检查已不存在的 `vllm._C` 名称；检查已改为本版实际 `_C_stable_libtorch` 并通过，失败记录保留。不是通过忽略编译扩展错误完成验收。
- 当前新增基准/汇总器回归见 `vllm安装后基准回归.log`；模型实例化、真实 GPU 吞吐和多卡退出依旧未验收，不能把命令模板写成已跑通完整模型。

## 对照契约

| 项目 | 约定 |
|---|---|
| 模型 | 同一路径的 Qwen3.5-27B、相同 tokenizer/config/index 哈希；记录权重文件大小和 mtime，不冒称全量权重 SHA256 |
| 精度 | BF16 权重/激活；Lumina GDN S 为 FP32；vLLM 显式 mamba_cache_dtype=bfloat16、mamba_ssm_cache_dtype=float32，实际状态精度仍需启动后确认并记录 |
| 输入 | 一次生成 token-ID JSON，两侧复用；无聊天模板或再次分词 |
| 采样 | greedy、ignore EOS、固定 max_tokens、repetition penalty=1；不继承模型 generation_config 的采样默认值 |
| 功能 | 无量化、无投机、无前缀复用；vLLM 文本专用模式；允许其原生融合/编译/调度优化 |
| 资源 | 相同物理卡、TP、max_num_seqs、max_model_len、token budget、显存利用配置；相同百分比不代表相同有效容量，另记录容量 |
| Graph | 默认分别使用引擎原生 Graph；启动配置和实际覆盖要确认，不强迫 vLLM 使用 Lumina 算术路径；eager 作独立消融 |
| 到达 | 所有请求在测量起点同时到达，依次入队；配置 capacity 是调度上限，不保证实际活跃数量恒定 |
| 计时 | 模型加载/JIT/Graph 初始化不计；完整同负载预热；测量包含入队、prefill、decode、输出观察，排除分词/HTTP |
| 输出 | 保存每轮完整 token IDs；同引擎重复一致才进入比较；跨引擎完整一致数与公共前缀单列 |
| 结果 | 聚合吞吐为总输出 token / 总 wall time，不平均各轮 tok/s；每轮指标和请求原始观察均保留 |

现阶段 adapter 使用官方 `LLM.llm_engine` 的 add_request/step 累积输出接口；这是版本敏感接口，必须在选定兼容版本做启动与多请求测试。源文件存在检查只是必要条件，不代表运行兼容或权重加载正确。实际 vLLM 配置写入 result.json，环境版本差异单列。

TTFT 从共同 burst 起点到首次可观察的输出；TPOT 按每请求 `(最后 token 时间－首 token 时间)/(输出数－1)`，ITL 是相邻 token 间隔，两者分别统计。输出只有 1 token 的请求不参与 TPOT。如果一次观察得到多个 token，不插值制造时间戳，TPOT/ITL 留空，报告 `single_token_observations=false`。P99 在不足 1000 请求时标为探索值，不据 32 个样本宣传稳定尾延迟。

不同内核/归约造成的输出分歧需独立调查。相同固定输出预算仍可比较性能，但不能称为已通过无损或 HF 质量验收；历史 HF raw logits 失败仍保留。

## 分阶段矩阵

1. 启动检查：TP2、C1、128/32，双方长度正确、无 NaN、能退出；确认 BF16/FP32 state、禁用功能与真实 Graph 路径。此阶段不发布正式数字。
2. 性能锚点：TP2、C32、128/128，完整预热、至少 3 轮；按独立进程交错顺序 L/V、V/L 复测，记录温度/频率/占用，避免两侧并行抢卡。
3. 扩展矩阵：TP1/2/4/8 × C1/8/32/128；输入/输出取 128/128、2048/128、128/512。逐步运行，有容量不足就记录拒绝/OOM/实际活跃，不假称满载；不要一次启动整个矩阵。
4. Prefill/decode：当前入口报告含 prefill 的端到端与请求延迟，不声称分离稳态解码吞吐；后续在双方建立同定义的阶段事件与活跃窗口，再报告 input tok/s、decode output tok/s。
5. Profiling：选 TP2 C1、TP2 C32 的短/长 prefill 和 decode；先 Nsight Systems，后对主要热点使用 Nsight Compute，避免盲目采全量。
6. 服务层：离线流程验收后再加共享 HTTP 客户端、固定到达率/Poisson 负载、SLO/goodput；本轮未实现，不混用旧的不对称 trace。

## 环境与复现

不要升级 lumina 或 verl。用户新建的独立环境已安装 0.26.0，以下使用其真实路径；安装成功不等于模型启动成功。后续复现实验使用 `vllm依赖锁定.txt` 对照精确版本，记录来源和配置，不随意滚动升级。

在 Lumina 根目录：

```bash
# 本轮已执行的共享输入生成方式；复测使用新的文件名。
PYTHONPATH=. /home/data/jhzhou/conda_envs/lumina/bin/python benchmarks/qwen35_baseline.py prepare \
  --model /home/data/jhzhou/huggingface/Qwen3.5-27B \
  --requests 32 --input-tokens 128 --output-tokens 128 --seed 0 \
  --output benchmarks/新workload.json

# 新环境已安装，所选卡空闲后才执行；两条命令顺序运行。
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. /home/data/jhzhou/conda_envs/lumina/bin/python \
  benchmarks/qwen35_baseline.py run --engine lumina \
  --model /home/data/jhzhou/huggingface/Qwen3.5-27B \
  --workload benchmarks/qwen35_baseline_20261112/workload_128_128_C32.json \
  --tp 2 --capacity 32 --max-model-len 4096 --token-budget 2048 --memory .85 \
  --warmups 1 --repeats 3 --port 2571 --output benchmarks/普通BF16_Lumina_新运行

CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH=. \
  /home/data/jhzhou/conda_envs/vllm/bin/python benchmarks/qwen35_baseline.py run --engine vllm \
  --model /home/data/jhzhou/huggingface/Qwen3.5-27B \
  --workload benchmarks/qwen35_baseline_20261112/workload_128_128_C32.json \
  --tp 2 --capacity 32 --max-model-len 4096 --token-budget 2048 --memory .85 \
  --warmups 1 --repeats 3 --output benchmarks/普通BF16_vLLM_新运行

/home/data/jhzhou/conda_envs/lumina/bin/python benchmarks/qwen35_baseline.py compare \
  --lumina benchmarks/普通BF16_Lumina_新运行/result.json \
  --vllm benchmarks/普通BF16_vLLM_新运行/result.json --output benchmarks/普通BF16_新比较.json
```

不能覆盖已有目录/结果。compare 会拒绝不完整、重复不稳定、workload/模型不同、关键参数或 GPU placement 不同的结果，以及任何 profile 运行；跨引擎差异不隐去。启动失败时 result.json 保留 complete=false 和错误，不能拿失败结果生成加速百分比。

## Profiling 流程

官方文档建议 vLLM 在 nsys 下用 spawn，并启用 fork/Graph node 跟踪。以下为待真实验收的命令模板；必须检查每个 TP rank/device 是否真的出现在 trace 中，不能用主进程空 trace 或单卡 trace 解释多卡瓶颈。

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. nsys profile \
  --trace=cuda,nvtx,osrt --trace-fork-before-exec=true --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=benchmarks/普通BF16_Lumina_新profile \
  /home/data/jhzhou/conda_envs/lumina/bin/python benchmarks/qwen35_baseline.py run \
  --engine lumina --model /home/data/jhzhou/huggingface/Qwen3.5-27B \
  --workload benchmarks/qwen35_baseline_20261112/workload_128_128_C32.json \
  --tp 2 --capacity 32 --token-budget 2048 --max-model-len 4096 --memory .85 \
  --warmups 1 --repeats 1 --profile --port 2572 --output benchmarks/普通BF16_Lumina_新profile结果

nsys export --type sqlite --output benchmarks/普通BF16_Lumina_新profile.sqlite \
  benchmarks/普通BF16_Lumina_新profile.nsys-rep
/home/data/jhzhou/conda_envs/lumina/bin/python benchmarks/qwen35_profile_summary.py \
  benchmarks/普通BF16_Lumina_新profile.sqlite --output benchmarks/普通BF16_Lumina_新profile汇总.json
```

vLLM 替换解释器/engine 并设置 spawn；如果 CUDA Profiler API 只捕获主进程而漏掉工作进程，须改用该版本官方 worker profiler 配置或逐 rank 采集，不宣布完成。已有 Lumina NVTX 覆盖 step、schedule、prepare、run_model、sampler、postprocess；层级 GDN/attention/MLP 的归属、实际 Graph replay 和跨 rank NCCL 等待仍需进一步时间线定位。

汇总器按设备合并 kernel 区间，通信和计算可以重叠；分类时长之和不是 wall time。`kernel_span_coverage` 仅是首末 kernel 区间内的 kernel 覆盖率，不包含 memcpy、首尾主机空隙，也不是 GPU utilization。不能仅凭 kernel 名将所有 GEMM 归为 MLP。后续报告应保留原始 trace、分段证据、热点形状、NCU 带宽/计算指标及优化前后配对数据。

参考：[官方 Qwen3.5 部署指南](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md)、[官方 Profiling 文档](https://docs.vllm.ai/en/stable/contributing/profiling/)。这两份在线文档会变化；正式运行应额外冻结对应版本页面/提交。

## 完成标准

基准工具 CPU 测试通过只是第一步。最终需要兼容官方环境与双方启动验收、无 profiler 的成对性能矩阵、可验证的多卡 trace、热点解释、至少一项有正确性与性能证据的优化，以及清晰的质量和容量边界。当前不能宣称以上全部完成。
