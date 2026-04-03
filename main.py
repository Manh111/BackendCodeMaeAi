import asyncio
import inspect
import os
import time
import uuid
from typing import List, Optional, Sequence, Tuple

import g4f
import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from openspace import OpenSpace
    from openspace.tool_layer import OpenSpaceConfig

    HAS_OPENSPACE = True
except Exception:
    HAS_OPENSPACE = False
    OpenSpace = None
    OpenSpaceConfig = None

app = FastAPI(title="MaeAI Backend API", redirect_slashes=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTH_KEY = os.getenv("API_KEY", "maeai-tuxue-v1-key-2026")
LEGACY_AUTH_KEYS = {k.strip() for k in [AUTH_KEY, os.getenv("LEGACY_API_KEY", ""), "silas123"] if k and k.strip()}
OPENSPACE_ENABLED = os.getenv("OPENSPACE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
OPENSPACE_AS_PRIMARY = os.getenv("OPENSPACE_AS_PRIMARY", "1").strip().lower() in {"1", "true", "yes"}
OPENSPACE_TIMEOUT_SECONDS = int(os.getenv("OPENSPACE_TIMEOUT_SECONDS", "60"))
G4F_TIMEOUT_SECONDS = int(os.getenv("G4F_TIMEOUT_SECONDS", "30"))
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))
DEFAULT_G4F_MODEL = os.getenv("G4F_DEFAULT_MODEL", "MaeAI Tuxue V1")
DEFAULT_OPENSPACE_MODEL = os.getenv("OPENSPACE_DEFAULT_MODEL", "default")
OPENSPACE_LOCAL_MODEL = os.getenv("OPENSPACE_LOCAL_MODEL", "ollama/qwen2.5-coder:3b")
UPSTREAM_ENABLED = os.getenv("UPSTREAM_ENABLED", "0").strip().lower() not in {"0", "false", "no"}
UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://contorted-valrie-noneffusively.ngrok-free.dev").strip().rstrip("/")
UPSTREAM_CHAT_PATH = "/" + os.getenv("UPSTREAM_CHAT_PATH", "v1/chat/completions").strip().lstrip("/")
UPSTREAM_LEGACY_CHAT_PATH = "/" + os.getenv("UPSTREAM_LEGACY_CHAT_PATH", "chat").strip().lstrip("/")
UPSTREAM_TIMEOUT_SECONDS = int(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "90"))
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = f"openspace:{DEFAULT_OPENSPACE_MODEL}"
    messages: List[Message]
    stream: bool = False
    max_tokens: Optional[int] = None


def _parse_model(model: str) -> Tuple[str, str]:
    raw = (model or "").strip()
    lower = raw.lower()

    if lower.startswith("openspace:"):
        return "openspace", raw.split(":", 1)[1].strip() or "default"
    if lower in {"openspace", "os"}:
        return "openspace", DEFAULT_OPENSPACE_MODEL or "default"
    if lower.startswith("duck:"):
        return "duck", raw.split(":", 1)[1].strip() or DEFAULT_G4F_MODEL
    if lower.startswith("g4f:"):
        return "g4f", raw.split(":", 1)[1].strip() or DEFAULT_G4F_MODEL
    if lower.startswith("supermax:"):
        return "supermax", raw.split(":", 1)[1].strip() or DEFAULT_G4F_MODEL
    if lower in {"supermax", "super-max", "super_max", "super max"}:
        return "supermax", DEFAULT_G4F_MODEL
    if lower.startswith("backend:"):
        inner = raw.split(":", 1)[1].strip()
        return _parse_model(inner or DEFAULT_G4F_MODEL)

    return "g4f", raw or DEFAULT_G4F_MODEL


def _normalize_messages(messages: Sequence[Message]) -> List[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _openai_response(model: str, content: str, suffix: str) -> dict:
    return {
        "id": f"chatcmpl-{suffix}-{uuid.uuid4().hex[:10]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


async def _run_upstream_chat(req: ChatRequest, auth_value: str) -> Optional[dict]:
    if not UPSTREAM_ENABLED or not UPSTREAM_BASE_URL:
        return None

    request_payload = {
        "model": req.model or DEFAULT_G4F_MODEL,
        "messages": _normalize_messages(req.messages),
        "stream": False,
    }

    if req.max_tokens is not None:
        request_payload["max_tokens"] = req.max_tokens

    headers = {
        "Authorization": f"Bearer {auth_value}",
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{UPSTREAM_BASE_URL}{UPSTREAM_CHAT_PATH}", json=request_payload, headers=headers)

        if response.status_code != 404:
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if choices:
                return data

        # Fallback for legacy FastAPI/Ollama wrapper endpoints: POST /chat?user_input=...
        last_user_message = ""
        for message in reversed(req.messages):
            if message.role == "user":
                last_user_message = message.content
                break
        if not last_user_message and req.messages:
            last_user_message = req.messages[-1].content

        legacy_response = await client.post(
            f"{UPSTREAM_BASE_URL}{UPSTREAM_LEGACY_CHAT_PATH}",
            params={"user_input": last_user_message},
            headers={"ngrok-skip-browser-warning": "true"},
        )
        if legacy_response.status_code == 404:
            return None

        legacy_response.raise_for_status()
        legacy_data = legacy_response.json()
        if isinstance(legacy_data, dict):
            text = str(legacy_data.get("response", "")).strip()
            if text:
                return _openai_response(req.model or DEFAULT_G4F_MODEL, text, "upstream")

    raise RuntimeError("Upstream response is not in supported format")


async def _run_openspace(prompt: str, model_name: str) -> str:
    if not (OPENSPACE_ENABLED and HAS_OPENSPACE):
        raise RuntimeError("OpenSpace is disabled or unavailable")

    resolved_model = (model_name or "").strip()
    if not resolved_model or resolved_model.lower() == "default":
        resolved_model = os.getenv("OPENSPACE_MODEL", "").strip() or OPENSPACE_LOCAL_MODEL

    # Ensure OpenSpace/LiteLLM reads the intended model instead of its internal OpenRouter default.
    os.environ["OPENSPACE_MODEL"] = resolved_model

    agent_config = OpenSpaceConfig(llm_model=resolved_model) if OpenSpaceConfig is not None else None

    async with OpenSpace(config=agent_config) if agent_config is not None else OpenSpace() as agent:
        execute_fn = agent.execute
        signature = inspect.signature(execute_fn)

        if "model" in signature.parameters and resolved_model and resolved_model != "default":
            result = await asyncio.wait_for(execute_fn(prompt, model=resolved_model), timeout=OPENSPACE_TIMEOUT_SECONDS)
        elif "llm_model" in signature.parameters and resolved_model and resolved_model != "default":
            result = await asyncio.wait_for(execute_fn(prompt, llm_model=resolved_model), timeout=OPENSPACE_TIMEOUT_SECONDS)
        else:
            result = await asyncio.wait_for(execute_fn(prompt), timeout=OPENSPACE_TIMEOUT_SECONDS)

    if isinstance(result, dict):
        text = str(result.get("response", "")).strip()
        if not text:
            for key in ("error", "message", "detail", "reason"):
                val = str(result.get(key, "")).strip()
                if val:
                    raise RuntimeError(val)
    else:
        text = str(result or "").strip()

    if not text:
        raise RuntimeError("OpenSpace returned an empty response. Please verify OpenSpace LLM credentials/model configuration.")

    return text


def _provider_retry_chain(provider_hint: str):
    duck_first = [
        getattr(g4f.Provider, "DuckDuckGo", None),
        getattr(g4f.Provider, "ChatGptEs", None),
        getattr(g4f.Provider, "Airforce", None),
        getattr(g4f.Provider, "FreeGpt", None),
        getattr(g4f.Provider, "Pizzagpt", None),
        getattr(g4f.Provider, "Liaobots", None),
    ]
    g4f_first = [
        getattr(g4f.Provider, "ChatGptEs", None),
        getattr(g4f.Provider, "Airforce", None),
        getattr(g4f.Provider, "FreeGpt", None),
        getattr(g4f.Provider, "Pizzagpt", None),
        getattr(g4f.Provider, "Liaobots", None),
        getattr(g4f.Provider, "DuckDuckGo", None),
    ]

    supermax_chain = [
        getattr(g4f.Provider, "DuckDuckGo", None),
        getattr(g4f.Provider, "ChatGptEs", None),
        getattr(g4f.Provider, "Airforce", None),
        getattr(g4f.Provider, "FreeGpt", None),
        getattr(g4f.Provider, "Pizzagpt", None),
        getattr(g4f.Provider, "Liaobots", None),
    ]

    if provider_hint == "duck":
        chain = duck_first
    elif provider_hint == "supermax":
        chain = supermax_chain
    else:
        chain = g4f_first
    chain = [provider for provider in chain if provider is not None]
    return g4f.Provider.RetryProvider(chain)


async def _run_g4f(messages: Sequence[Message], model_name: str, provider_hint: str) -> str:
    def _call() -> str:
        return str(
            g4f.ChatCompletion.create(
                model=model_name,
                messages=_normalize_messages(messages),
                provider=_provider_retry_chain(provider_hint),
                ignore_working=True,
                timeout=G4F_TIMEOUT_SECONDS,
            )
        )

    result = (await asyncio.to_thread(_call)).strip()
    if not result:
        raise RuntimeError("G4F returned an empty response")
    return result


async def _run_ollama_direct(messages: Sequence[Message], model_name: str) -> str:
    model = (model_name or "").strip()
    if model.startswith("ollama/"):
        model = model.split("/", 1)[1].strip()
    if not model:
        model = "qwen2.5-coder:3b"

    payload = {
        "model": model,
        "messages": _normalize_messages(messages),
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
        response = await client.post("http://127.0.0.1:11434/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

    text = str((data.get("message") or {}).get("content", "")).strip()
    if not text:
        raise RuntimeError("Ollama returned empty content")
    return text


@app.get("/")
def health_check():
    return {
        "status": "alive",
        "service": "maeai-backend",
        "openspace_installed": HAS_OPENSPACE,
        "openspace_enabled": OPENSPACE_ENABLED,
    }


@app.get("/v1/chat/completions")
def chat_completion_probe():
    return {
        "status": "ok",
        "message": "Use POST /v1/chat/completions",
    }


@app.options("/{rest_of_path:path}")
async def preflight_handler(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest, authorization: Optional[str] = Header(None)):
    auth_value = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not auth_value or auth_value not in LEGACY_AUTH_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    requested_model = req.model or f"openspace:{DEFAULT_OPENSPACE_MODEL}"
    provider, resolved_model = _parse_model(requested_model)
    request_id = uuid.uuid4().hex[:8]
    openspace_error: Optional[Exception] = None

    try:
        upstream_result = await _run_upstream_chat(req, auth_value)
        if upstream_result is not None:
            print(f"[maeai:{request_id}] provider=upstream status=ok model={requested_model}")
            return upstream_result
    except Exception as exc:
        print(f"[maeai:{request_id}] provider=upstream status=fallback reason={exc}")

    if provider == "openspace" or (provider == "g4f" and OPENSPACE_AS_PRIMARY):
        try:
            prompt = req.messages[-1].content
            openspace_model = resolved_model or DEFAULT_OPENSPACE_MODEL
            openspace_text = await _run_openspace(prompt, openspace_model)
            print(f"[maeai:{request_id}] provider=openspace status=ok")
            return _openai_response(f"openspace:{openspace_model}", openspace_text, "openspace")
        except Exception as exc:
            openspace_error = exc
            print(f"[maeai:{request_id}] provider=openspace status=fallback reason={exc}")

            # If user explicitly selected OpenSpace, return explicit diagnostic instead of silent provider switch.
            if provider == "openspace":
                err_text = str(exc)
                if (
                    "No cookie auth credentials found" in err_text
                    or "AuthenticationError" in err_text
                    or "empty response" in err_text.lower()
                ):
                    err_text = (
                        "OpenSpace chưa có thông tin xác thực/model khả dụng. "
                        "Nếu dùng cloud model, hãy cấu hình key tương ứng (ví dụ OPENROUTER_API_KEY). "
                        "Nếu dùng local model, hãy đặt OPENSPACE_MODEL về model local trong cấu hình OpenSpace và khởi động lại backend."
                    )
                # For local Ollama models, attempt direct Ollama API fallback to keep chat usable.
                if (resolved_model or "").startswith("ollama/") or (openspace_model or "").startswith("ollama/"):
                    try:
                        ollama_text = await _run_ollama_direct(req.messages, openspace_model or resolved_model)
                        return _openai_response(f"ollama:{(openspace_model or resolved_model).replace('ollama/', '')}", ollama_text, "ollama")
                    except Exception as ollama_exc:
                        err_text = f"{err_text} | Ollama fallback failed: {ollama_exc}"

                return _openai_response("openspace:error", err_text, "openspace")

    try:
        if provider == "duck":
            fallback_hint = "duck"
        elif provider == "supermax":
            fallback_hint = "supermax"
        else:
            fallback_hint = "g4f"
        fallback_model = resolved_model or DEFAULT_G4F_MODEL
        g4f_text = await _run_g4f(req.messages, fallback_model, fallback_hint)
        print(f"[maeai:{request_id}] provider=g4f status=ok route={fallback_hint} model={fallback_model}")
        return _openai_response(f"{fallback_hint}:{fallback_model}", g4f_text, "g4f")
    except Exception as exc:
        print(f"[maeai:{request_id}] provider=g4f status=error reason={exc}")
        if openspace_error is not None:
            print(f"[maeai:{request_id}] openspace_root_cause={openspace_error}")
        return _openai_response(
            "fallback:busy",
            "OpenSpace và các provider G4F hiện chưa phản hồi. Vui lòng thử lại sau ít phút.",
            "fallback",
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)