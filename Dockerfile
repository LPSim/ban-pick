FROM python:3.10-alpine
RUN pip install --no-cache-dir fastapi uvicorn
WORKDIR /ban-pick
COPY . .
CMD ["python3", "server.py"]
