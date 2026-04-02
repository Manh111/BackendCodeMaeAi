# Nâng cấp lên bản 3.12 để chiều lòng OpenSpace
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    OPENSPACE_ENABLED=1

# 1. Gia cố HĐH: Cài đặt git, chứng chỉ SSL và curl
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# 2. Update pip lên bản mới nhất
RUN pip install --upgrade pip

# 3. Cài đặt thư viện (Bây giờ Python 3.12 sẽ khớp với OpenSpace)
RUN pip install --no-cache-dir -r requirements.txt

# Railway sẽ tự động cấp cổng qua biến môi trường PORT
CMD ["python", "main.py"]