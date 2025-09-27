from fastapi import FastAPI
import os

app = FastAPI(title="Face Template API", version="0.1.0")

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "api",
        "uploads_dir": os.getenv("UPLOADS_DIR"),
        "outputs_dir": os.getenv("OUTPUTS_DIR"),
        "infer_base_url": os.getenv("INFER_BASE_URL"),
    }
