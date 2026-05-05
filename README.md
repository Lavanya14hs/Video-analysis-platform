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

## Deployment

This project is designed for deployment on platforms that support Python backends with ML models.

### Recommended: Railway

Railway provides the best support for this type of application.

1. Create account at [railway.app](https://railway.app)
2. Install Railway CLI: `npm install -g @railway/cli`
3. Login: `railway login`
4. Initialize: `railway init`
5. Deploy: `railway up`

The project includes `requirements.txt`, `Procfile`, `railway.json`, and a `Dockerfile` for deployment.
The `Dockerfile` installs necessary system libraries for OpenCV in headless mode and is the recommended deployment path.

> Note: The project now uses CPU-only PyTorch wheels via `torch==2.11.0+cpu` and `torchvision==0.16.0+cpu` to avoid downloading large CUDA packages during deployment.

### Other Options

See `DEPLOYMENT.md` for detailed instructions on Railway, Render, and Heroku deployment.

### Vercel Limitations

Vercel is not suitable for this project because:
- No Python runtime for ML models
- Limited storage and compute resources
- Cannot handle video processing workloads

## Notes

- Keep your model weights in the `models/` folder, especially `yolov8n.pt`, `fight_detection.pt`, and `accident_detection.pt`.
- Use `temp/` for uploads and clean it periodically.
- Avoid committing large model files or sensitive keys to Git.
