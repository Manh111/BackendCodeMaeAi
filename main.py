import asyncio
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

    HAS_OPENSPACE = True
except Exception:
    HAS_OPENSPACE = False
    OpenSpace = None

app = FastAPI(title="MaeAI Backend API", redirect_slashes=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTH_KEY = os.getenv("API_KEY", "silas123")
LEGACY_AUTH_KEYS = {k.strip() for k in [AUTH_KEY, os.getenv("LEGACY_API_KEY", ""), "silas123"] if k and k.strip()}
OPENSPACE_ENABLED = os.getenv("OPENSPACE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
OPENSPACE_AS_PRIMARY = os.getenv("OPENSPACE_AS_PRIMARY", "0").strip().lower() in {"1", "true", "yes"}
OPENSPACE_TIMEOUT_SECONDS = int(os.getenv("OPENSPACE_TIMEOUT_SECONDS", "60"))
G4F_TIMEOUT_SECONDS = int(os.getenv("G4F_TIMEOUT_SECONDS", "30"))
DEFAULT_G4F_MODEL = os.getenv("G4F_DEFAULT_MODEL", "gpt-4o-mini")
AI_SCRAPER_BASE_URL = os.getenv("AI_SCRAPER_BASE_URL", "").strip().rstrip("/")
AI_SCRAPER_SECRET = os.getenv("AI_SCRAPER_SECRET", "")
AI_SCRAPER_TIMEOUT_SECONDS = int(os.getenv("AI_SCRAPER_TIMEOUT_SECONDS", "180"))


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "g4f:gpt-4o-mini"
    messages: List[Message]
    stream: bool = False
    max_tokens: Optional[int] = None


def _parse_model(model: str) -> Tuple[str, str]:
    raw = (model or "").strip()
    lower = raw.lower()

    if lower.startswith("openspace:"):
        return "openspace", raw.split(":", 1)[1].strip() or "default"
    if lower in {"openspace", "os"}:
        return "openspace", "default"
    if lower.startswith("duck:"):
        return "duck", raw.split(":", 1)[1].strip() or DEFAULT_G4F_MODEL
    if lower.startswith("g4f:"):
        return "g4f", raw.split(":", 1)[1].strip() or DEFAULT_G4F_MODEL
    if lower.startswith("supermax:"):
        return "supermax", raw.split(":", 1)[1].strip() or DEFAULT_G4F_MODEL
    if lower in {"supermax", "super-max", "super_max", "super max"}:
        return "supermax", DEFAULT_G4F_MODEL
    if lower.startswith("gemini:"):
        return "gemini", raw.split(":", 1)[1].strip() or "default"
    if lower == "gemini":
        return "gemini", "default"
    if lower.startswith("chatgpt:"):
        return "chatgpt", raw.split(":", 1)[1].strip() or "default"
    if lower in {"chatgpt", "openai"}:
        return "chatgpt", "default"
    if lower.startswith("grok:"):
        return "grok", raw.split(":", 1)[1].strip() or "default"
    if lower == "grok":
        return "grok", "default"
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


async def _run_openspace(prompt: str) -> str:
    if not (OPENSPACE_ENABLED and HAS_OPENSPACE):
        raise RuntimeError("OpenSpace is disabled or unavailable")

    async with OpenSpace() as agent:
        result = await asyncio.wait_for(agent.execute(prompt), timeout=OPENSPACE_TIMEOUT_SECONDS)

    if isinstance(result, dict):
        text = str(result.get("response", "")).strip()
    else:
        text = str(result or "").strip()

    if not text:
        raise RuntimeError("OpenSpace returned an empty response")

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


async def _run_ai_scraper(ai_name: str, prompt: str) -> str:
    if not AI_SCRAPER_BASE_URL:
        raise RuntimeError("AI_SCRAPER_BASE_URL is not configured")

    headers = {"Content-Type": "application/json"}
    if AI_SCRAPER_SECRET:
        headers["X-API-Secret"] = AI_SCRAPER_SECRET

    url = f"{AI_SCRAPER_BASE_URL}/ask"
    payload = {"ai": ai_name, "prompt": prompt}

    async with httpx.AsyncClient(timeout=AI_SCRAPER_TIMEOUT_SECONDS) as client:
        response = await client.post(url, json=payload, headers=headers)

    if response.status_code == 401:
        raise RuntimeError("AI scraper unauthorized (invalid X-API-Secret)")
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "")
        except Exception:
            detail = response.text
        raise RuntimeError(f"AI scraper error {response.status_code}: {str(detail)[:300]}")

    data = response.json()
    if not data.get("ok"):
        err = data.get("error") or f"{ai_name} returned empty output"
        raise RuntimeError(str(err))

    text = str(data.get("text", "")).strip()
    if not text:
        raise RuntimeError(f"{ai_name} returned empty output")
    return text


@app.get("/")
def health_check():
    return {
        "status": "alive",
        "service": "maeai-backend",
        "openspace_installed": HAS_OPENSPACE,
        "openspace_enabled": OPENSPACE_ENABLED,
        "ai_scraper_configured": bool(AI_SCRAPER_BASE_URL),
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

    requested_model = req.model or DEFAULT_G4F_MODEL
    provider, resolved_model = _parse_model(requested_model)
    request_id = uuid.uuid4().hex[:8]

    if provider in {"gemini", "chatgpt", "grok"}:
        try:
            prompt = req.messages[-1].content
            tool_text = await _run_ai_scraper(provider, prompt)
            print(f"[maeai:{request_id}] provider=ai_scraper status=ok ai={provider}")
            return _openai_response(f"{provider}:{resolved_model}", tool_text, "aiscraper")
        except Exception as exc:
            print(f"[maeai:{request_id}] provider=ai_scraper status=error ai={provider} reason={exc}")
            return _openai_response(
                f"{provider}:error",
                f"{provider} hiện không phản hồi qua ai_scraper: {exc}",
                "aiscraper",
            )

    if provider == "openspace" or (provider == "g4f" and OPENSPACE_AS_PRIMARY):
        try:
            prompt = req.messages[-1].content
            openspace_text = await _run_openspace(prompt)
            print(f"[maeai:{request_id}] provider=openspace status=ok")
            return _openai_response(f"openspace:{resolved_model}", openspace_text, "openspace")
        except Exception as exc:
            print(f"[maeai:{request_id}] provider=openspace status=fallback reason={exc}")

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
        return _openai_response(
            "fallback:busy",
            "OpenSpace và các provider G4F hiện chưa phản hồi. Vui lòng thử lại sau ít phút.",
            "fallback",
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)