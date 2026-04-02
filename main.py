import os
import time
import uvicorn
import g4f
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from duckduckgo_search import DDGS

app = FastAPI(title="Silas Pro API")

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

@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest, authorization: Optional[str] = Header(None)):
    # 1. Kiểm tra Auth (NextChat gửi token theo format 'Bearer <key>')
    if not authorization or authorization != f"Bearer {AUTH_KEY}":
        raise HTTPException(status_code=401, detail="Key lỏ hoặc chưa nhập Key vào NextChat!")

    prompt = req.messages[-1].content
    current_timestamp = int(time.time())

    # 2. Ưu tiên DuckDuckGo (Trâu, mượt, ít chết)
    # Tự động map tên model từ NextChat sang chuẩn của DuckDuckGo
    duck_models = {
        "gpt-4": "gpt-4o-mini",
        "gpt-3.5": "gpt-4o-mini",
        "claude-3": "claude-3-haiku",
        "llama-3": "llama-3.1-70b"
    }
    
    selected_model = "gpt-4o-mini" # Mặc định
    for key, val in duck_models.items():
        if key in req.model.lower():
            selected_model = val
            break

    try:
        with DDGS() as ddgs:
            # Gọi DuckDuckGo AI
            response = ddgs.chat(prompt, model=selected_model)
            if response:
                return {
                    "id": f"chatcmpl-silas-duck-{current_timestamp}",
                    "object": "chat.completion",
                    "created": current_timestamp,
                    "model": req.model,
                    "choices": [{
                        "index": 0, 
                        "message": {"role": "assistant", "content": response}, 
                        "finish_reason": "stop"
                    }]
                }
    except Exception as e:
        print(f"[SILAS LOG] DuckDuckGo dở chứng: {e}. Đang chuyển qua G4F...")

    # 3. Fallback sang G4F nếu DuckDuckGo bị lỗi (Rate limit hoặc quá tải)
    try:
        # Dùng RetryProvider để G4F tự động thử nhiều nhà cung cấp
        response = g4f.ChatCompletion.create(
            model=req.model,
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
            "choices": [{
                "index": 0, 
                "message": {"role": "assistant", "content": response}, 
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        print(f"[SILAS LOG] Sập toàn tập: {e}")
        raise HTTPException(status_code=500, detail=f"Sập toàn tập rồi bác: {str(e)}")

if __name__ == "__main__":
    # Tự động lấy cổng (PORT) do Railway cấp phát
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)