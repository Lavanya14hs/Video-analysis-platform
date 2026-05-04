# Video Analysis Platform

A video incident analysis system with object detection, fight detection, accident detection, and crowd detection.

## Project structure

- `api/` - FastAPI service exposing the video analysis endpoint
- `backend/` - analysis logic and model utilities
- `frontend/` - React UI for uploading and viewing results
- `models/` - trained model files used by the analyzer
- `logs/` - generated log files and snapshots
- `temp/` - temporary upload storage

## Getting started

### Python backend

1. Create a Python virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the required Python dependencies.

```powershell
pip install fastapi uvicorn opencv-python ultralytics numpy
```

3. Run the API server.

```powershell
uvicorn api.main:app --reload
```

### Frontend

1. Install frontend dependencies.

```powershell
cd frontend
npm install
```

2. Start the React app.

```powershell
npm start
```

## Notes

- Keep your model weights in the `models/` folder, especially `yolov8n.pt`, `fight_detection.pt`, and `accident_detection.pt`.
- Use `temp/` for uploads and clean it periodically.
- Avoid committing large model files or sensitive keys to Git.
