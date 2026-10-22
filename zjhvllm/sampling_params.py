from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    """解码采样参数。

    greedy=True 时强制取 argmax（忽略温度，等价于 T→0）；
    否则按 temperature 做核采样。top_k / top_p / repetition_penalty 可叠加。
    """

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    greedy: bool = True
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 0, "temperature must be > 0 (set greedy=True for argmax decoding)"
        assert self.top_k == -1 or self.top_k > 0, "top_k must be -1 (disabled) or > 0"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
        assert self.repetition_penalty > 0, "repetition_penalty must be > 0"
