import os
import time
import uvicorn
import g4f
from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Silas Pro API", redirect_slashes=False)

# Cấu hình CORS - Giữ nguyên vì bác đã thông quan được 200 OK trước đó
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
    max_tokens: Optional[int] = None

@app.get("/")
def health_check():
    return {"status": "alive", "msg": "API Silas đã sửa lỗi AttributeError!"}

@app.options("/{rest_of_path:path}")
async def preflight_handler(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest, authorization: Optional[str] = Header(None)):
    if not authorization or authorization != f"Bearer {AUTH_KEY}":
        raise HTTPException(status_code=401, detail="Key lỏ rồi bác ơi!")

    current_timestamp = int(time.time())
    requested_model = req.model.split(":")[-1].strip() if ":" in req.model else req.model

    # --- ĐOẠN FIX LỖI ATTRIBUTEERROR ---
    # Tôi dùng Try-Except để lỡ nó đổi tên nữa thì App vẫn không sập
    provider_list = []
    try:
        # Cập nhật tên theo bản g4f mới nhất (BlackboxPro, DuckDuckGo, Liaobots)
        from g4f.Provider import BlackboxPro, DuckDuckGo, Liaobots, Airforce
        provider_list = [BlackboxPro, DuckDuckGo, Liaobots, Airforce]
    except ImportError:
        # Fallback nếu thư viện lại đổi tên nữa
        provider_list = [] 

    try:
        # Nếu provider_list trống, g4f sẽ tự chọn cái tốt nhất
        provider = g4f.Provider.RetryProvider(provider_list) if provider_list else None

        response = g4f.ChatCompletion.create(
            model=requested_model,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            max_tokens=req.max_tokens or 4096,
            provider=provider
        )

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
        print(f"[SILAS ERROR] Lỗi thực thi: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI đang bận: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)