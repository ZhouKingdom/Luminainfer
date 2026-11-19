# LuminaInfer

基于 PyTorch、Triton 和 NCCL 的轻量 LLM 推理引擎，Python 包名为 `zjhvllm`。当前聚焦 **Qwen3.5-27B 文本 BF16 多卡推理、可复现 benchmark 和 profiling**，以官方 vLLM 为外部基线。

## 核心能力

| 方向 | 实现 |
|---|---|
| 混合注意力 | 48 层 Gated DeltaNet + 16 层全注意力；按请求维护 FP32 递归状态与 BF16 卷积状态 |
| 多卡推理 | TP1/2/4/8；GDN 头切分、词表并行、KV 头不足时复制 |
| 执行与调度 | 分页 KV、分块 prefill、连续批处理、抢占重算、decode CUDA Graph |
| 内核 | Triton GDN 扫描/递推、分页注意力及采样算子 |
| 性能评估 | 统一 token-ID workload、逐请求 TTFT/TPOT/ITL/E2E、独立 vLLM 对照、Nsight 汇总 |

已有 Qwen3 Dense、Qwen3-MoE 和 MTP 实验路径。**当前暂停 MTP 与量化研发**，不宣称它们已带来普通推理净加速。

## 快速使用

已验证的 Lumina 环境：Python 3.12、PyTorch 2.8.0+cu128、Triton 3.4.0、Transformers 5.20.1、FlashAttention 2.8.3.post1。FlashAttention wheel 须匹配 PyTorch/CUDA/ABI；不同版本组合需自行验证。

准备环境和本地 Qwen3.5-27B 权重后，从项目根目录运行：

```python
from zjhvllm import LLM, SamplingParams


def main():
    llm = LLM(
        '/path/to/Qwen3.5-27B',
        tensor_parallel_size=2,
        enable_mtp=False,
        max_num_seqs=32,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=.85,
        enforce_eager=False,
    )
    try:
        outputs = llm.generate(
            ['解释混合注意力模型中的递归状态。'],
            SamplingParams(greedy=True, max_tokens=128),
            use_tqdm=False,
        )
        print(outputs)
    finally:
        llm.exit()


if __name__ == '__main__':
    main()
```

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python your_script.py
```

## Benchmark 与 profiling

官方对照环境独立安装 vLLM 0.26.0 / PyTorch 2.11.0，避免改变 Lumina 的依赖。以 BF16、相同输入、输出预算、TP、并发和物理 GPU 配对，预热与测量分离；profiling 不参与吞吐排名。

截至 2026-11-19：原 65 组队列中 33 组通过配对验收、28 组因 vLLM 重复输出不一致被排除；4 组 profiling 已在修复后的独立队列完成。通过验收的配置只是子集，跨引擎输出也存在差异，不能据此宣称全面质量对齐或超过 vLLM。Profiling 采集成功不等于热点优化已完成。

- [性能数据与验收边界](测试与性能.md)
- [原性能矩阵汇总](benchmarks/qwen35_auto_20261112/实验汇总.md)
- [Profiling 补跑汇总](benchmarks/qwen35_profile_retry_20261114/实验汇总.md)
- [统一基准与复现说明](benchmarks/qwen35_baseline_20261112/基准与Profiling实施说明.md)

例如先生成输入，再在各自环境执行 `run --engine lumina` / `run --engine vllm`，最后 `compare`：

```bash
PYTHONPATH=. python benchmarks/qwen35_baseline.py prepare \
  --model /path/to/Qwen3.5-27B --requests 32 \
  --input-tokens 128 --output-tokens 128 --output /tmp/workload.json
```

脚本与文档中的 `/home/...` 路径是原实验机器配置，复现时替换为本机路径。后台监测脚本不会随克隆或导入自动启动。

## 测试与代码

无需加载模型的基准/监测 CPU 回归：

```bash
PYTHONPATH=. python -m pytest \
  tests/test_qwen35_baseline.py tests/test_qwen35_profile_summary.py \
  tests/test_watch_qwen35.py -q
```

GPU/真实模型测试需要相应权重、空闲 GPU 和兼容依赖，不应把上面的 CPU 子集视为全项目验收。HF 严格 raw logits 对齐仍未通过，历史 next-token argmax 一致不等价于完整质量保证。

```text
zjhvllm/engine/   调度、分页 KV、混合状态、分布式执行与 Graph
zjhvllm/models/   Qwen3、Qwen3-MoE、Qwen3.5 及 MTP
zjhvllm/layers/   注意力、GDN、并行算子与 Triton 内核
tests/           CPU、CUDA、数值与分布式测试
benchmarks/      基准、诊断脚本和精简报告
```

进一步阅读：[项目说明](项目说明.md)、[使用指南](使用指南.md)、[Qwen3.5 进度](Qwen3.5进度.md)。

仓库提交源码、测试、脚本、精简报告与部分汇总；权重、缓存、完整源码快照、运行日志、完整 token 结果和大型 Nsight trace 保留在实验机器，不随 Git 上传。部分历史文档中的原始产物链接因此只在原机器可用。

参考实现：[vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)。本项目为独立实现，并非上述项目的官方版本。
