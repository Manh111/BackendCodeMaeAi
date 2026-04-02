import asyncio
import os
import time
import uuid
from typing import List, Optional, Sequence, Tuple

import g4f
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
OPENSPACE_ENABLED = os.getenv("OPENSPACE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
OPENSPACE_AS_PRIMARY = os.getenv("OPENSPACE_AS_PRIMARY", "0").strip().lower() in {"1", "true", "yes"}
OPENSPACE_TIMEOUT_SECONDS = int(os.getenv("OPENSPACE_TIMEOUT_SECONDS", "60"))
G4F_TIMEOUT_SECONDS = int(os.getenv("G4F_TIMEOUT_SECONDS", "30"))
DEFAULT_G4F_MODEL = os.getenv("G4F_DEFAULT_MODEL", "gpt-4o-mini")


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

    chain = duck_first if provider_hint == "duck" else g4f_first
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
    if not authorization or authorization != f"Bearer {AUTH_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    requested_model = req.model or DEFAULT_G4F_MODEL
    provider, resolved_model = _parse_model(requested_model)
    request_id = uuid.uuid4().hex[:8]

    if provider == "openspace" or (provider == "g4f" and OPENSPACE_AS_PRIMARY):
        try:
            prompt = req.messages[-1].content
            openspace_text = await _run_openspace(prompt)
            print(f"[maeai:{request_id}] provider=openspace status=ok")
            return _openai_response(f"openspace:{resolved_model}", openspace_text, "openspace")
        except Exception as exc:
            print(f"[maeai:{request_id}] provider=openspace status=fallback reason={exc}")

    try:
        fallback_hint = "duck" if provider == "duck" else "g4f"
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