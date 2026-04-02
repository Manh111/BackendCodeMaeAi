FROM python:3.10-slim

# 1. Gia cố HĐH: Cài đặt chứng chỉ SSL và curl để né lỗi mạng khi gọi API lậu
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# 2. Update pip lên bản mới nhất trước khi cài thư viện để tránh lỗi build g4f
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Railway sẽ tự động cấp cổng qua biến môi trường PORT
CMD ["python", "main.py"]