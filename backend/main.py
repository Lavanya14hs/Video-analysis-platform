from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import json
import asyncio
from pathlib import Path

try:
    from .video_analyzer import analyze_video
except ImportError:
    from video_analyzer import analyze_video

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "temp"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def home():
    return {"message": "API Running 🚀"}


@app.post("/analyze-stream")
async def analyze_stream(file: UploadFile = File(...)):

    UPLOAD_DIR.mkdir(exist_ok=True)
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    async def event_generator():

        queue = []

        def progress_callback(p):
            queue.append({"progress": int(p * 100)})

        loop = asyncio.get_event_loop()

        task = loop.run_in_executor(
            None,
            lambda: analyze_video(file_path, progress_callback=progress_callback)
        )

        try:
            while True:

                while queue:
                    yield f"data: {json.dumps(queue.pop(0))}\n\n"

                if task.done():
                    result = task.result()
                    yield f"data: {json.dumps({'done': True, 'result': result})}\n\n"
                    break

                await asyncio.sleep(0.1)
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Serve static files from the built React frontend
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")