import os
import time
import uvicorn
import g4f
from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Thử import OpenSpace từ repo HKUDS (Sẽ bỏ qua nếu bác chưa cài)
try:
    from openspace import OpenSpaceAgent
    HAS_OPENSPACE = True
    print("[SILAS SYSTEM] Đã load thành công lõi não OpenSpace!")
except ImportError:
    HAS_OPENSPACE = False
    print("[SILAS SYSTEM] Chưa cài OpenSpace, sẽ dùng 100% G4F.")

app = FastAPI(title="Silas Pro API - Final Success", redirect_slashes=False)

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
    return {"status": "alive", "msg": "API Silas - Đã trang bị lõi OpenSpace!"}

@app.options("/{rest_of_path:path}")
async def preflight_handler(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest, authorization: Optional[str] = Header(None)):
    if not authorization or authorization != f"Bearer {AUTH_KEY}":
        raise HTTPException(status_code=401, detail="Key lỏ!")

    current_timestamp = int(time.time())
    prompt = req.messages[-1].content
    
    # ---------------------------------------------------------
    # TẦNG 1: NÃO BỘ OPENSPACE (Tự tiến hóa & Phân tích logic)
    # ---------------------------------------------------------
    if HAS_OPENSPACE:
        try:
            # Khởi tạo Agent (truyền Auth Key vào nếu cần bảo mật nội bộ)
            os_agent = OpenSpaceAgent(api_key=AUTH_KEY)
            
            # Để OpenSpace xử lý câu hỏi
            os_response = os_agent.run(prompt)
            
            if os_response and len(str(os_response)) > 2:
                print("[SILAS LOG] OpenSpace đã xử lý thành công!")
                return {
                    "id": f"chatcmpl-silas-os-{current_timestamp}",
                    "object": "chat.completion",
                    "created": current_timestamp,
                    "model": "openspace-agent",
                    "choices": [{
                        "index": 0, 
                        "message": {"role": "assistant", "content": str(os_response)}, 
                        "finish_reason": "stop"
                    }]
                }
        except Exception as e:
            print(f"[SILAS LOG] OpenSpace bận hoặc lỗi: {e}. Đang chuyển qua G4F...")

    # ---------------------------------------------------------
    # TẦNG 2: CHIẾN THUẬT VƯỢT RÀO G4F (Dự phòng)
    # ---------------------------------------------------------
    requested_model = "gpt-4o-mini"

    # Danh sách Provider "Cảm tử quân"
    providers = [
        getattr(g4f.Provider, "ChatGptEs", None),
        getattr(g4f.Provider, "Airforce", None),
        getattr(g4f.Provider, "FreeGpt", None),
        getattr(g4f.Provider, "Pizzagpt", None),
        getattr(g4f.Provider, "DuckDuckGo", None),
        getattr(g4f.Provider, "Liaobots", None)
    ]
    providers = [p for p in providers if p is not None]

    try:
        response = g4f.ChatCompletion.create(
            model=requested_model,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            provider=g4f.Provider.RetryProvider(providers),
            ignore_working=True,
            timeout=30 
        )

        if not response or len(str(response)) < 2:
            raise Exception("Tất cả Provider đều im lặng")

        return {
            "id": f"chatcmpl-silas-g4f-{current_timestamp}",
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
        print(f"[SILAS ERROR] Vẫn sập: {str(e)}")
        # CÚ CHỐT: Fallback cuối cùng
        return {
            "choices": [{
                "message": {
                    "role": "assistant", 
                    "content": "Bác Silas ơi, hiện tại toàn bộ Provider AI và OpenSpace đang bận hoặc bị chặn IP. Bác đợi vài phút để hệ thống tự phục hồi nhé!"
                }, 
                "finish_reason": "stop"
            }]
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)