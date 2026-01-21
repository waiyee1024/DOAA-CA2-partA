FROM python:3.11-slim

WORKDIR /app

# Pillow 常用依赖（处理 jpg/png）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000

# 用 gunicorn 启动 Flask app（Render 会给 PORT）
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5000} app:app"]


