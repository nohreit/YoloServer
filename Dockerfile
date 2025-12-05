FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git libgl1 && rm -rf /var/lib/apt/lists/*
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python","server/main.py"]
