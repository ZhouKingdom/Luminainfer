"""异步推理引擎：在后台线程跑 continuous batching 的 step 循环，支持逐 token 流式输出。

设计要点：
- 所有对同步引擎（LLMEngine）的访问都集中在后台线程里，避免多线程竞争调度器/KV cache。
- HTTP 侧（asyncio）只通过 thread-safe 的 queue.Queue 提交请求，
  通过 asyncio.Queue（经 loop.call_soon_threadsafe 投递）接收 token 流。
"""

import asyncio
import queue
import threading
import time

from zjhvllm.llm import LLM
from zjhvllm.sampling_params import SamplingParams


class AsyncLLMEngine:
    def __init__(self, model: str, **kwargs):
        self.engine = LLM(model, **kwargs)
        self._input: "queue.Queue[tuple[list[int], SamplingParams, asyncio.Queue]]" = queue.Queue()
        self._pending: dict[int, asyncio.Queue] = {}  # seq_id -> 该请求的结果队列
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    def _run_loop(self):
        while True:
            # 1. 把异步侧提交的请求灌进同步引擎
            while True:
                try:
                    prompt_ids, sampling_params, result_q = self._input.get_nowait()
                except queue.Empty:
                    break
                seq_id = self.engine.add_request(prompt_ids, sampling_params)
                self._pending[seq_id] = result_q

            # 2. 执行一步 continuous batching（只有有待处理序列时才调用，避免空跑）
            if self.engine.is_finished():
                time.sleep(0.001)
                continue

            outputs, _num_tokens, stream = self.engine.step()

            loop = self._loop
            if loop is not None:
                for seq_id, token_id in stream:
                    q = self._pending.get(seq_id)
                    if q is not None:
                        loop.call_soon_threadsafe(q.put_nowait, ("token", token_id))
                for seq_id, completion in outputs:
                    q = self._pending.pop(seq_id, None)
                    if q is not None:
                        loop.call_soon_threadsafe(q.put_nowait, ("done", completion))

    async def _submit(self, prompt_ids: list[int], sampling_params: SamplingParams) -> asyncio.Queue:
        self._loop = asyncio.get_running_loop()
        result_q: asyncio.Queue = asyncio.Queue()
        self._input.put((prompt_ids, sampling_params, result_q))
        return result_q

    async def stream_generate(self, prompt_ids: list[int], sampling_params: SamplingParams):
        """异步生成器：逐 token yield（int token id）。遇到 "done" 事件即结束。"""
        result_q = await self._submit(prompt_ids, sampling_params)
        while True:
            kind, payload = await result_q.get()
            if kind == "done":
                return
            yield payload

    async def generate(self, prompt_ids: list[int], sampling_params: SamplingParams) -> list[int]:
        """非流式：收集全部 token 后返回。"""
        tokens = []
        async for tok in self.stream_generate(prompt_ids, sampling_params):
            tokens.append(tok)
        return tokens
