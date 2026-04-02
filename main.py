import os
import time
import uvicorn
import g4f
from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

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
    max_tokens: Optional[int] = None

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

    current_timestamp = int(time.time())
    
    # Xử lý prefix và model
    requested_model = (req.model or "gpt-4o-mini").strip().lower()
    route = "auto"
    
    if requested_model.startswith("duck:"):
        route = "duck"
        requested_model = requested_model.split(":", 1)[1].strip() or "gpt-4o-mini"
    elif requested_model.startswith("g4f:"):
        route = "g4f"
        requested_model = requested_model.split(":", 1)[1].strip() or "gpt-4o-mini"

    # Gộp toàn bộ luồng gọi vào G4F để tận dụng RetryProvider, tránh lỗi 500 do thư viện ngoài
    providers = []
    if route == "duck":
        providers = [g4f.Provider.DuckDuckGo]
    elif route == "g4f":
        providers = [g4f.Provider.Blackbox, g4f.Provider.Liaobots]
    else:
        providers = [g4f.Provider.Blackbox, g4f.Provider.Liaobots, g4f.Provider.DuckDuckGo]

    try:
        response = g4f.ChatCompletion.create(
            model=requested_model,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            max_tokens=req.max_tokens or 4096,
            provider=g4f.Provider.RetryProvider(providers)
        )
        
        # Đảm bảo response luôn là string an toàn
        final_content = str(response) if response else "AI đang bận, bác thử lại sau nhé."

        return {
            "id": f"chatcmpl-silas-g4f-{current_timestamp}",
            "object": "chat.completion",
            "created": current_timestamp,
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": final_content}, "finish_reason": "stop"}]
        }
    except Exception as e:
        print(f"[SILAS LOG] Sập toàn tập: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sập toàn tập: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)