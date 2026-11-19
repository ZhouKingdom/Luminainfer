"""共享 workload 生成器：zjh-vllm 与 vLLM 两侧逐字一致，保证对比公平。

对比旧的「256 条同时到达 + input/output 各 100~1024 均匀随机」，这里引入
真实负载的三个要素：

  1. 输入/输出长度重尾（log-normal）：短 prompt 为主、少数超长（ShareGPT 形态），
     而非均匀随机。
  2. 共享系统前缀：所有请求共享同一段 system prompt，触发 prefix caching。
  3. Poisson 到达：请求陆续到达（open-loop），而非同时涌入；closed-loop 传
     arrival_rate=None。

所有随机数都用独立的 Random(seed+offset) 流，保证：
  (a) 跨引擎逐字可复现；
  (b) 关掉前缀（prefix_len=0）时 body 的长度/内容不变，便于做「前缀缓存 abl 对比」。
"""

from dataclasses import dataclass
import random


@dataclass
class Request:
    prompt_token_ids: list[int]
    max_tokens: int
    arrival: float = 0.0   # 相对首个请求的到达秒数（closed-loop 恒 0）


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(v))))


def make_requests(
    num_seqs: int,
    seed: int = 0,
    *,
    prefix_len: int = 512,
    input_mu: float = 5.2,      # 中位输入 ≈ exp(5.2) ≈ 181 token
    input_sigma: float = 1.0,   # 重尾
    output_mu: float = 4.3,     # 中位输出 ≈ exp(4.3) ≈ 74 token
    output_sigma: float = 0.9,
    min_len: int = 8,
    max_input: int = 2048,
    max_output: int = 512,
    vocab: int = 32000,
    arrival_rate: float | None = None,   # None=closed-loop；否则 Poisson 速率（请求/秒）
    shared_prefix: bool = True,          # True=所有请求共享同一段前缀（触发缓存）；False=每请求唯一前缀（同长度、不共享，作 abl 基线）
) -> list[Request]:
    len_rng = random.Random(seed)
    prefix_rng = random.Random(seed + 0x10000)
    body_rng = random.Random(seed + 0x20000)
    arr_rng = random.Random(seed + 0x30000)

    arrivals = [0.0] * num_seqs
    if arrival_rate is not None:
        t = 0.0
        for i in range(num_seqs):
            t += arr_rng.expovariate(arrival_rate)
            arrivals[i] = t

    shared = [prefix_rng.randint(0, vocab - 1) for _ in range(prefix_len)] if (prefix_len and shared_prefix) else None

    reqs: list[Request] = []
    for i in range(num_seqs):
        in_len = _clamp_int(len_rng.lognormvariate(input_mu, input_sigma), min_len, max_input)
        out_len = _clamp_int(len_rng.lognormvariate(output_mu, output_sigma), min_len, max_output)
        if prefix_len:
            prefix = shared if shared is not None else [prefix_rng.randint(0, vocab - 1) for _ in range(prefix_len)]
        else:
            prefix = []
        body = [body_rng.randint(0, vocab - 1) for _ in range(in_len)]
        reqs.append(Request(prompt_token_ids=prefix + body, max_tokens=out_len, arrival=arrivals[i]))
    return reqs


def percentile(sorted_vals: list[float], p: float) -> float:
    """线性插值分位数（numpy.percentile 的线性插值语义）。"""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def report_latency(name: str, vals: list[float]):
    vals = sorted(vals)
    if not vals:
        print(f"{name}: (empty)")
        return
    mean = sum(vals) / len(vals)
    print(
        f"{name}: mean {mean*1000:8.2f}ms   "
        f"p50 {percentile(vals, 0.50)*1000:8.2f}ms   "
        f"p90 {percentile(vals, 0.90)*1000:8.2f}ms   "
        f"p99 {percentile(vals, 0.99)*1000:8.2f}ms"
    )
