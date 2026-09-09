import os
import tempfile
import shutil
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles

from process_video import remove_grok_watermark, VideoInfo

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
ASSETS = BASE / "assets"
MAX_BYTES = 60 * 1024 * 1024
MAX_SECONDS = 30.0

app = FastAPI(title="Grok Watermark Remover")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")

@app.get("/", response_class=HTMLResponse)
def home():
    return (STATIC / "index.html").read_text(encoding="utf-8")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/remove")
async def remove(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file selected")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".m4v", ".webm", ".mkv"}:
        raise HTTPException(400, "Please upload an MP4, MOV, M4V, WEBM, or MKV video.")

    with tempfile.TemporaryDirectory(prefix="grokremover-") as td:
        td = Path(td)
        input_path = td / f"input{ext}"
        output_path = td / "grok-cleaned.mp4"

        size = 0
        with input_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "Video is larger than the 60 MB limit.")
                f.write(chunk)

        try:
            info = remove_grok_watermark(
                input_path=input_path,
                output_path=output_path,
                mask_path=ASSETS / "grok-mask.png",
                max_seconds=MAX_SECONDS,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            raise HTTPException(500, f"Processing failed: {e}") from e

        # FileResponse reads the file before temp dir cleanup by ASGI in practice,
        # but to be robust copy to a NamedTemporaryFile outside the context.
        persistent = Path(tempfile.mkstemp(prefix="grok-cleaned-", suffix=".mp4")[1])
        shutil.copy2(output_path, persistent)

    return FileResponse(
        persistent,
        media_type="video/mp4",
        filename="grok-cleaned.mp4",
        background=BackgroundTask(lambda: persistent.unlink(missing_ok=True)),
    )
