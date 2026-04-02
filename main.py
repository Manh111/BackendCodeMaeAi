import os
import time
import uvicorn
import g4f
from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Silas Pro API - Final Shield", redirect_slashes=False)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {"status": "alive", "msg": "API Silas đã loại bỏ Blackbox lỗi!"}

@app.options("/{rest_of_path:path}")
async def preflight_handler(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest, authorization: Optional[str] = Header(None)):
    if not authorization or authorization != f"Bearer {AUTH_KEY}":
        raise HTTPException(status_code=401, detail="Key lỏ rồi bác!")

    current_timestamp = int(time.time())
    
    # Ép model về bản ổn định nhất
    requested_model = "gpt-4o-mini" if "gpt" in req.model.lower() else "gpt-4o"

    # DANH SÁCH PROVIDER "LỲ" NHẤT HIỆN TẠI (Bỏ Blackbox)
    safe_providers = [
        getattr(g4f.Provider, "Airforce", None),
        getattr(g4f.Provider, "ChatGptEs", None),
        getattr(g4f.Provider, "FreeGpt", None),
        getattr(g4f.Provider, "DuckDuckGo", None),
        getattr(g4f.Provider, "Pizzagpt", None),
    ]
    safe_providers = [p for p in safe_providers if p is not None]

    try:
        # RetryProvider sẽ tự động nhảy qua thằng tiếp theo nếu thằng trước báo 404
        response = g4f.ChatCompletion.create(
            model=requested_model,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            provider=g4f.Provider.RetryProvider(safe_providers),
            ignore_working=True # Bỏ qua kiểm tra trạng thái để ép nó chạy
        )

        if not response:
            raise Exception("Tất cả Provider đều từ chối trả lời")

        return {
            "id": f"chatcmpl-silas-{current_timestamp}",
            "object": "chat.completion",
            "created": current_timestamp,
            "model": req.model,
            "choices": [{
                "index": 0, 
                "message": {"role": "assistant", "content": str(response)}, 
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        print(f"[SILAS ERROR] Sập nguồn: {str(e)}")
        # Trả về text lỗi trực tiếp vào ô chat thay vì văng lỗi 500 sập web
        return {
            "choices": [{
                "message": {"role": "assistant", "content": f"⚠️ AI đang bảo trì (Lỗi: {str(e)}). Bác thử lại sau vài giây nhé!"},
                "finish_reason": "stop"
            }]
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)