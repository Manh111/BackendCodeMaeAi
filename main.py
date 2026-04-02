import os
import time
import uvicorn
import g4f
from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Silas Pro API - Platinum", redirect_slashes=False)

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
    max_tokens: Optional[int] = None

@app.get("/")
def health_check():
    return {"status": "alive", "msg": "API Silas đã FIX xong lỗi Permission và Model!"}

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
    
    # --- TRICK: MAPPING MODEL CHUẨN CHO G4F 7.x ---
    # Ép các model 'ngáo' về model mà G4F chắc chắn hỗ trợ
    raw_model = req.model.lower()
    if "claude" in raw_model:
        requested_model = "gpt-4o" # Dùng GPT-4 gánh tạ cho Claude vì Claude lậu hay chết
    elif "gpt-4" in raw_model:
        requested_model = "gpt-4o-mini"
    else:
        requested_model = "gpt-4o-mini"

    # --- CHỈ DÙNG PROVIDER "SẠCH" KHÔNG ĐÒI COOKIE/KEY ---
    # Loại bỏ các ông đòi ghi file 'har_and_cookies'
    safe_providers = [
        getattr(g4f.Provider, "BlackboxPro", None),
        getattr(g4f.Provider, "DuckDuckGo", None),
        getattr(g4f.Provider, "Airforce", None),
        getattr(g4f.Provider, "ChatGptEs", None)
    ]
    # Lọc bỏ các Provider bị None (không tồn tại trong version này)
    safe_providers = [p for p in safe_providers if p is not None]

    try:
        response = g4f.ChatCompletion.create(
            model=requested_model,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            provider=g4f.Provider.RetryProvider(safe_providers)
        )

        # Trả về JSON chuẩn để NextChat không báo lỗi
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
        print(f"[SILAS ERROR] Crash: {str(e)}")
        # Trả về lỗi 200 kèm nội dung lỗi để UI không bị treo đỏ lòm
        return {
            "choices": [{
                "message": {"role": "assistant", "content": f"⚠️ Lỗi rồi bác: {str(e)}"},
                "finish_reason": "stop"
            }]
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)