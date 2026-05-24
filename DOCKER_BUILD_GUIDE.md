# Combined Docker Build Guide

## Overview

The project now uses a **multi-stage Docker build** that combines both the React frontend and Python backend into a single image. This approach:

- **Reduces complexity**: Single image for deployment
- **Optimizes size**: Frontend is built separately, only the production build is included
- **Simplifies deployment**: No need to manage multiple containers or configurations
- **Improves performance**: Static files are served efficiently alongside the API

## Architecture

### Stage 1: Frontend Builder (Node.js)
- Uses `node:18-alpine` as the builder image
- Installs npm dependencies from `frontend/package.json`
- Builds the React app with `npm run build`
- Creates an optimized production build in `frontend/build/`

### Stage 2: Backend + Static Files (Python)
- Uses `python:3.12-slim` as the final image
- Installs all Python dependencies
- Copies backend code
- **Copies the built React frontend** from Stage 1 into the `/app/static/` directory
- FastAPI serves both API endpoints and static files

## Building the Image

### Build the Docker image:
```bash
docker build -t video-analysis-platform:latest .
```

### Run the container:
```bash
docker run -p 8000:8000 video-analysis-platform:latest
```

### With environment variables:
```bash
docker run -p 8000:8000 \
  -e PORT=8000 \
  video-analysis-platform:latest
```

## File Structure in Container

```
/app/
├── main.py                 # FastAPI application
├── video_analyzer.py       # Video analysis logic
├── requirements.txt        # Python dependencies
├── models/                 # Pre-trained model files
├── temp/                   # Temporary upload directory (created at runtime)
└── static/                 # React build output (mounted from Stage 1)
    ├── index.html
    ├── css/
    ├── js/
    └── ...
```

## How It Works

1. **Frontend Request**: User visits `http://localhost:8000/`
   - FastAPI serves `static/index.html` (React app)

2. **API Request**: Frontend makes request to `/analyze-stream`
   - FastAPI handles the API endpoint
   - Backend processes video

3. **Static Assets**: Frontend loads `css/`, `js/` bundles
   - FastAPI serves from `static/` directory

## Updates Made

### Dockerfile
- Replaced single-stage build with multi-stage build
- Stage 1: Builds React frontend
- Stage 2: Sets up Python backend and mounts static files
- Added HEALTHCHECK for container monitoring

### backend/main.py
- Added code to mount static files from built React frontend
- Lines at the end: `app.mount("/", StaticFiles(directory="static", html=True), name="static")`

### .dockerignore
- Added exclusions for build artifacts
- Added exclusions for development directories
- Optimized for faster builds

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `8000` | Port the application listens on |

## Performance Considerations

- **Frontend Build Time**: ~30-60 seconds (cached after first build)
- **Backend Setup Time**: ~20-40 seconds
- **Total Build Time**: ~1-2 minutes (first build with no cache)

### Optimization Tips:
1. Build with BuildKit for better caching: `DOCKER_BUILDKIT=1 docker build -t image:tag .`
2. Use `.dockerignore` to exclude unnecessary files (already optimized)
3. Python dependencies are cached separately from frontend dependencies

## Troubleshooting

### Build fails at npm install
- Ensure `frontend/package.json` exists
- Check Node.js compatibility in `frontend/`

### Frontend not serving
- Verify `frontend/build/` directory is created properly
- Check that main.py mounts the static files correctly

### Static files returning 404
- Confirm React app is built: `npm run build` in frontend/
- Verify the mount path in main.py: `app.mount("/", StaticFiles(directory="static", html=True), name="static")`

### Port conflicts
- Change PORT variable: `docker run -e PORT=9000 ...`

## Deployment

For production deployment (Railway, Render, etc.):

1. Ensure the Dockerfile is in the root directory ✓
2. Set `PORT` environment variable
3. Docker will automatically:
   - Build frontend
   - Copy to backend
   - Serve both together

## Next Steps (Optional)

- Add nginx reverse proxy for better static file serving
- Implement service worker for offline support
- Add docker-compose for local development with live reload
- Set up CI/CD pipeline for automated builds
