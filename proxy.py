"""
LLM Proxy — OpenAI-compatible reverse proxy with request/response inspector
============================================================================
用法:
  pip install fastapi uvicorn httpx
  python proxy.py
  python proxy.py --upstream http://127.0.0.1:8000 --proxy-port 7654 --ui-port 7655 --max-records 200
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from threading import Lock
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── 配置 ──────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="LLM Proxy Inspector")
_parser.add_argument("--upstream",    default=os.getenv("UPSTREAM_BASE", "http://127.0.0.1:8000"))
_parser.add_argument("--proxy-port",  type=int, default=int(os.getenv("PROXY_PORT", "7654")))
_parser.add_argument("--ui-port",     type=int, default=int(os.getenv("UI_PORT",    "7655")))
_parser.add_argument("--max-records", type=int, default=int(os.getenv("MAX_RECORDS","200")))
_args = _parser.parse_args()

UPSTREAM_BASE = _args.upstream.rstrip("/")
PROXY_PORT    = _args.proxy_port
UI_PORT       = _args.ui_port
MAX_RECORDS   = _args.max_records
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("proxy")

_store: OrderedDict[str, dict] = OrderedDict()
_lock  = Lock()


# ── SSE 解析 ──────────────────────────────────────────────────────────────────

def _merge_delta(acc: dict, delta: dict) -> None:
    """递归合并一个 delta 片段到累积 dict。
    规则（不假设具体字段名）：
      - None 值跳过，不覆盖已有内容
      - str  → 拼接（content / reasoning_content / function.arguments 等增量文本）
      - list → 按元素的 "index" 字段找到同位置条目后递归合并（tool_calls 场景）
      - dict → 递归合并（function: {name, arguments} 等嵌套结构）
      - 其他 → 直接覆盖（role / finish_reason / logprobs 等只出现一次的字段）
    """
    for key, val in delta.items():
        if val is None:
            continue
        if key not in acc or acc[key] is None:
            # 首次出现：深拷贝避免与原始 chunk 共享引用
            acc[key] = copy.deepcopy(val)
        elif isinstance(val, str) and isinstance(acc[key], str):
            # 增量文本拼接，如 content / reasoning_content / arguments
            acc[key] += val
        elif isinstance(val, list) and isinstance(acc[key], list):
            # 列表元素按 "index" 字段定位后递归合并
            # 典型场景：tool_calls，每个 chunk 携带部分 arguments
            for item in val:
                if not isinstance(item, dict):
                    acc[key].append(item)
                    continue
                item_idx = item.get("index")
                existing = next(
                    (x for x in acc[key] if isinstance(x, dict) and x.get("index") == item_idx),
                    None,
                ) if item_idx is not None else None
                if existing is None:
                    # 新的 tool_call 条目，首次出现
                    acc[key].append(copy.deepcopy(item))
                else:
                    _merge_delta(existing, item)
        elif isinstance(val, dict) and isinstance(acc[key], dict):
            # 嵌套 dict 递归合并，如 function: {name, arguments}
            _merge_delta(acc[key], val)
        else:
            # 覆盖：role 等只出现一次的字段，或类型与首次不同时（上游异常场景）
            if type(acc[key]) is not type(val):
                log.warning("merge type mismatch key=%r acc=%r val=%r, overwriting", key, type(acc[key]).__name__, type(val).__name__)
            acc[key] = val


def parse_sse_lines(lines: list[str]) -> dict:
    """将原始 SSE 行列表合并为类非流式响应结构。

    只对 choices（按 index）和 usage 做特殊处理，
    delta 内部字段全部交给 _merge_delta 泛化处理。
    """
    chunks: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunks.append(json.loads(payload))
        except json.JSONDecodeError:
            pass

    if not chunks:
        return {}

    # 顶层元数据取第一个 chunk
    first = chunks[0]
    result: dict = {
        "id":      first.get("id"),
        "object":  "chat.completion",
        "created": first.get("created"),
        "model":   first.get("model"),
        "usage":   None,
    }

    # key: choice index，value: 累积合并后的 choice dict
    choices_acc: dict[int, dict] = {}
    for chunk in chunks:
        # usage 通常只在最后一个 chunk 出现（需请求时开启 stream_options）
        if chunk.get("usage"):
            result["usage"] = chunk["usage"]

        for choice in chunk.get("choices", []):
            idx = choice.get("index", 0)
            if idx not in choices_acc:
                choices_acc[idx] = {"index": idx, "finish_reason": None}
            # delta 内容泛化合并，不假设具体字段
            _merge_delta(choices_acc[idx], choice.get("delta", {}))
            # choice 顶层字段（finish_reason / logprobs / matched_stop 等）
            # 不假设有哪些，跳过 delta 和 index 之外全部合并
            top = {k: v for k, v in choice.items() if k not in ("delta", "index") and v is not None}
            _merge_delta(choices_acc[idx], top)

    result["choices"] = [choices_acc[i] for i in sorted(choices_acc)]
    return result


def _save(record_id: str, data: dict):
    with _lock:
        _store[record_id] = data
        while len(_store) > MAX_RECORDS:
            _store.popitem(last=False)


# ── Proxy App ─────────────────────────────────────────────────────────────────

proxy_app = FastAPI(title="LLM Proxy")


@proxy_app.api_route("/{path:path}", methods=["GET","POST","PUT","DELETE","PATCH","OPTIONS"])
async def proxy(path: str, request: Request):
    url = f"{UPSTREAM_BASE}/{path}"
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    body_bytes = await request.body()
    try:
        req_json = json.loads(body_bytes) if body_bytes else None
    except Exception:
        req_json = None

    is_stream = isinstance(req_json, dict) and req_json.get("stream", False)
    record_id = str(uuid.uuid4())
    ts = datetime.now().strftime("%H:%M:%S")
    model = req_json.get("model", "") if req_json else ""

    # 基础记录（先存，流结束后补充）
    base_record = {
        "id":       record_id,
        "time":     ts,
        "method":   request.method,
        "path":     f"/{path}",
        "model":    model,
        "stream":   is_stream,
        "req_json": req_json,
        "status":   None,
        "latency":  None,
        "is_sse":   False,
        "resp_json":   None,   # 非流式
        "resp_merged": None,   # 流式合并后
        "resp_raw":    None,   # 原始文本（非流式）
        "sse_lines":   None,   # 原始 SSE 行
    }
    _save(record_id, base_record)

    t0 = time.monotonic()
    log.info("→ %s %s  model=%s  stream=%s", request.method, url, model or "-", is_stream)

    # ── 流式 ──────────────────────────────────────────────
    if is_stream:
        async def stream_gen() -> AsyncIterator[bytes]:
            sse_lines: list[str] = []
            status_code = 200
            async with httpx.AsyncClient(timeout=300) as stream_client:
                try:
                    async with stream_client.stream(
                        request.method, url,
                        headers=headers, content=body_bytes
                    ) as upstream:
                        status_code = upstream.status_code
                        log.info("← %s %s  status=%s  [SSE started]", request.method, url, status_code)
                        async for raw_line in upstream.aiter_lines():
                            sse_lines.append(raw_line)
                            yield (raw_line + "\n").encode()
                except Exception as exc:
                    log.error("✗ %s %s  error=%s", request.method, url, exc)
                    yield f"data: [ERROR] {exc}\n\n".encode()
                finally:
                    latency = round((time.monotonic() - t0) * 1000)
                    merged = parse_sse_lines(sse_lines)
                    log.info("← %s %s  status=%s  %dms  [SSE done, chunks=%d]",
                             request.method, url, status_code, latency, len(sse_lines))
                    with _lock:
                        if record_id in _store:
                            _store[record_id].update({
                                "status":      status_code,
                                "latency":     latency,
                                "is_sse":      True,
                                "resp_merged": merged,
                                "sse_lines":   sse_lines,
                            })

        return StreamingResponse(
            stream_gen(),
            media_type="text/event-stream",
            headers={"X-Proxy-Record-Id": record_id},
        )

    # ── 非流式 ────────────────────────────────────────────
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            resp = await client.request(
                request.method, url,
                headers=headers, content=body_bytes
            )
        except Exception as exc:
            log.error("✗ %s %s  error=%s", request.method, url, exc)
            raise
        latency = round((time.monotonic() - t0) * 1000)
        log.info("← %s %s  status=%s  %dms", request.method, url, resp.status_code, latency)
        try:
            resp_json = resp.json()
        except Exception:
            resp_json = None

        with _lock:
            if record_id in _store:
                _store[record_id].update({
                    "status":   resp.status_code,
                    "latency":  latency,
                    "is_sse":   False,
                    "resp_json": resp_json,
                    "resp_raw":  resp.text if resp_json is None else None,
                })

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={
                **dict(resp.headers),
                "X-Proxy-Record-Id": record_id,
            },
            media_type=resp.headers.get("content-type"),
        )


# ── UI App ────────────────────────────────────────────────────────────────────

ui_app = FastAPI(title="LLM Proxy UI")


@ui_app.get("/api/records")
def api_records():
    """侧边栏列表，轻量数据。"""
    with _lock:
        items = list(_store.values())
    result = []
    for r in reversed(items):
        result.append({
            "id":      r["id"],
            "time":    r["time"],
            "method":  r["method"],
            "path":    r["path"],
            "model":   r.get("model", ""),
            "stream":  r.get("stream", False),
            "status":  r.get("status"),
            "latency": r.get("latency"),
            "is_sse":  r.get("is_sse", False),
            "done":    r.get("status") is not None,
        })
    return result


@ui_app.get("/api/records/{record_id}")
def api_record_detail(record_id: str):
    """单条记录完整数据。"""
    with _lock:
        r = _store.get(record_id)
    if not r:
        return JSONResponse({"error": "not found"}, status_code=404)
    return r


@ui_app.get("/ids/{record_id}")
@ui_app.get("/")
def index():
    return FileResponse("static/index.html")


# ── 静态文件 ──────────────────────────────────────────────────────────────────

ui_app.mount("/static", StaticFiles(directory="static"), name="static")


# ── 启动两个服务 ──────────────────────────────────────────────────────────────

async def main():
    cfg_proxy = uvicorn.Config(proxy_app, host="0.0.0.0", port=PROXY_PORT, log_level="warning")
    cfg_ui    = uvicorn.Config(ui_app,    host="0.0.0.0", port=UI_PORT,    log_level="warning")

    print(f"  🔀  Proxy  → http://0.0.0.0:{PROXY_PORT}  (upstream: {UPSTREAM_BASE})")
    print(f"  🔍  UI     → http://0.0.0.0:{UI_PORT}")
    print()

    await asyncio.gather(
        uvicorn.Server(cfg_proxy).serve(),
        uvicorn.Server(cfg_ui).serve(),
    )


if __name__ == "__main__":
    asyncio.run(main())
