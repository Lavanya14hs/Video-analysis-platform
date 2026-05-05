from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import sys
import json
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from video_analyzer import analyze_video

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "API Running 🚀"}


@app.post("/analyze-stream")
async def analyze_stream(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

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

        while True:

            while queue:
                yield f"data: {json.dumps(queue.pop(0))}\n\n"

            if task.done():
                result = task.result()
                yield f"data: {json.dumps({'done': True, 'result': result})}\n\n"
                break

            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")