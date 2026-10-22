"""OpenAI 兼容的推理服务（FastAPI）。

启动：
    python server.py --model ~/huggingface/Qwen3-0.6B --port 8000
或
    uvicorn server:app --host 0.0.0.0 --port 8000

端点：
    POST /v1/completions      文本补全（支持 stream）
    POST /v1/chat/completions 对话补全（支持 stream）
    GET  /v1/models           模型列表
    GET  /health              健康检查
"""

import argparse
import json
import os
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from zjhvllm.engine.async_llm_engine import AsyncLLMEngine
from zjhvllm.sampling_params import SamplingParams


# --------------------------------------------------------------------------- #
# 请求 / 响应模型（OpenAI 兼容字段）
# --------------------------------------------------------------------------- #
class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str | list[int]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    stream: bool = False


# --------------------------------------------------------------------------- #
# 全局引擎（启动时加载）
# --------------------------------------------------------------------------- #
app = FastAPI(title="zjhvllm OpenAI-compatible server")
_engine: AsyncLLMEngine | None = None
_MODEL_NAME = "default"


def _to_sampling_params(*, temperature, top_p, top_k, repetition_penalty, seed, max_tokens) -> SamplingParams:
    greedy = temperature == 0.0
    return SamplingParams(
        temperature=1.0 if greedy else max(temperature, 1e-6),
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=seed,
        greedy=greedy,
        max_tokens=max_tokens,
    )


def _chat_to_prompt_ids(engine: AsyncLLMEngine, messages: list[ChatMessage]) -> list[int]:
    text = engine.tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    return engine.tokenizer.encode(text, add_special_tokens=False)


def _finish_reason(tokens: list[int], eos_id: int) -> str:
    if tokens and tokens[-1] == eos_id:
        return "stop"
    return "length"


async def _stream_deltas(engine: AsyncLLMEngine, prompt_ids: list[int], sp: SamplingParams) -> AsyncGenerator[tuple[str, str], None]:
    """逐 token 产出 (delta_text, finish_reason)，finish_reason 仅最后一个 chunk 非 None。"""
    eos_id = engine.tokenizer.eos_token_id
    all_tokens: list[int] = []
    prev_text = ""

    async for token in engine.stream_generate(prompt_ids, sp):
        if token == eos_id:
            yield "", "stop"
            return
        all_tokens.append(token)
        text = engine.tokenizer.decode(all_tokens, skip_special_tokens=False)
        delta = text[len(prev_text):]
        prev_text = text
        yield delta, None

    yield "", _finish_reason(all_tokens, eos_id)


# --------------------------------------------------------------------------- #
# 端点
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": _MODEL_NAME, "object": "model"}]}


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    assert _engine is not None
    prompt_ids = req.prompt if isinstance(req.prompt, list) else _engine.tokenizer.encode(req.prompt)
    sp = _to_sampling_params(**req.model_dump(exclude={"model", "prompt", "stream"}))

    if req.stream:
        return StreamingResponse(_completions_sse(_engine, prompt_ids, sp, req.model), media_type="text/event-stream")

    tokens = await _engine.generate(prompt_ids, sp)
    text = _engine.tokenizer.decode(tokens, skip_special_tokens=True)
    eos_id = _engine.tokenizer.eos_token_id
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": None,
            "finish_reason": _finish_reason(tokens, eos_id),
        }],
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(tokens),
                  "total_tokens": len(prompt_ids) + len(tokens)},
    }


async def _completions_sse(engine, prompt_ids, sp, model):
    cmpl_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    async for delta, finish in _stream_deltas(engine, prompt_ids, sp):
        chunk = {
            "id": cmpl_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"text": delta, "index": 0, "logprobs": None, "finish_reason": finish}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    assert _engine is not None
    prompt_ids = _chat_to_prompt_ids(_engine, req.messages)
    sp = _to_sampling_params(**req.model_dump(exclude={"messages", "model", "stream"}))

    if req.stream:
        return StreamingResponse(_chat_sse(_engine, prompt_ids, sp, req.model), media_type="text/event-stream")

    tokens = await _engine.generate(prompt_ids, sp)
    content = _engine.tokenizer.decode(tokens, skip_special_tokens=True)
    eos_id = _engine.tokenizer.eos_token_id
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": _finish_reason(tokens, eos_id),
        }],
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(tokens),
                  "total_tokens": len(prompt_ids) + len(tokens)},
    }


async def _chat_sse(engine, prompt_ids, sp, model):
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    async for delta, finish in _stream_deltas(engine, prompt_ids, sp):
        chunk = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": finish}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# 启动
# --------------------------------------------------------------------------- #
def create_engine(model_path: str, **kwargs) -> AsyncLLMEngine:
    global _engine
    _engine = AsyncLLMEngine(model_path, **kwargs)
    _engine.start()
    return _engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    _MODEL_NAME = os.path.basename(args.model.rstrip("/"))
    create_engine(args.model, tensor_parallel_size=args.tensor_parallel_size, enforce_eager=args.enforce_eager)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
