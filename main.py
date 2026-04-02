import os
import time
import uvicorn
import g4f
from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from duckduckgo_search import DDGS

# Thêm redirect_slashes=False để diệt tận gốc lỗi CORS do dư dấu /
app = FastAPI(title="Silas Pro API", redirect_slashes=False)

# CHUẨN HÓA CORS: Chỉ định đích danh domain, bỏ dấu "*" để tránh xung đột credentials
origins = [
    "https://mae.manh.homes",
    "http://127.0.0.1:5500",
    "http://localhost:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lấy Key từ biến môi trường Railway (Mặc định là silas123 nếu bác quên set)
AUTH_KEY = os.getenv("API_KEY", "silas123")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: List[Message]
    stream: bool = False

@app.get("/")
def health_check():
    return {"status": "alive", "msg": "API Silas đang chạy ngầm nhé bác!"}

# Bùa chú ép CORS cho Preflight (Sửa lỗi 405)
@app.options("/v1/chat/completions")
async def preflight_handler(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "https://mae.manh.homes"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest, authorization: Optional[str] = Header(None)):
    if not authorization or authorization != f"Bearer {AUTH_KEY}":
        raise HTTPException(status_code=401, detail="Key lỏ hoặc chưa nhập Key vào NextChat!")

    prompt = req.messages[-1].content
    current_timestamp = int(time.time())
    
    requested_model = (req.model or "gpt-4o-mini").strip().lower()
    route = "auto"
    if requested_model.startswith("duck:"):
        route = "duck"
        requested_model = requested_model.split(":", 1)[1].strip() or "gpt-4o-mini"
    elif requested_model.startswith("g4f:"):
        route = "g4f"
        requested_model = requested_model.split(":", 1)[1].strip() or "gpt-4o-mini"

    duck_models = {
        "gpt-4": "gpt-4o-mini",
        "gpt-3.5": "gpt-4o-mini",
        "claude-3": "claude-3-haiku",
        "llama-3": "llama-3.1-70b"
    }
    
    selected_model = "gpt-4o-mini" 
    for key, val in duck_models.items():
        if key in requested_model:
            selected_model = val
            break

    def duck_response(model_name: str):
        with DDGS() as ddgs:
            return ddgs.chat(prompt, model=model_name)

    if route in {"duck", "auto"}:
        try:
            response = duck_response(selected_model)
            if response:
                return {
                    "id": f"chatcmpl-silas-duck-{current_timestamp}",
                    "object": "chat.completion",
                    "created": current_timestamp,
                    "model": req.model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}]
                }
        except Exception as e:
            print(f"[SILAS LOG] Duck dở chứng: {e}")

    if route == "duck":
        raise HTTPException(status_code=500, detail="Duck route failed")

    try:
        response = g4f.ChatCompletion.create(
            model=requested_model,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            provider=g4f.Provider.RetryProvider([
                g4f.Provider.Blackbox,
                g4f.Provider.Liaobots,
                g4f.Provider.DuckDuckGo
            ])
        )
        return {
            "id": f"chatcmpl-silas-g4f-{current_timestamp}",
            "object": "chat.completion",
            "created": current_timestamp,
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sập toàn tập: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)