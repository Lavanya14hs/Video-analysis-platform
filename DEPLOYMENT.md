# Deployment Guide

## Option 1: Render (Recommended - Simplest)

Render supports full-stack applications with persistent disks and is the easiest to set up.

### Steps:
1. Create account at [render.com](https://render.com)
2. Connect your GitHub repository (or import this one)
3. Click "New +" and select "Web Service"
4. Configure:
   - **Name**: `video-analysis-platform`
   - **Environment**: Docker
   - **Build Command**: leave blank
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Deploy and get your live URL

### Or use `render.yaml`:
This repo now includes `render.yaml` at the repository root so Render can auto-detect the service configuration.
- `root`: `backend`
- `dockerfilePath`: `Dockerfile`
- `startCommand`: `uvicorn main:app --host 0.0.0.0 --port $PORT`

If the repo has `render.yaml`, Render will auto-detect it:
1. Connect your GitHub repo
2. Render auto-deploys on push

## Option 2: Railway (More Complex)

Railway supports Python backends with ML models and provides persistent storage.

### Steps:
1. Create account at [railway.app](https://railway.app)
2. Install Railway CLI: `npm install -g @railway/cli`
3. Login: `railway login`
4. Initialize: `railway init`
5. Deploy: `railway up`

**Note**: Railway requires careful environment variable handling with Docker.

## Option 3: Heroku

Heroku has good Python support but charges for add-ons.

### Steps:
1. Install Heroku CLI
2. Create app: `heroku create`
3. Add buildpacks for Python
4. Deploy: `git push heroku main`

## Current Limitations with Vercel:

- ❌ No Python runtime for ML models
- ❌ Limited storage (100MB free)
- ❌ No persistent file storage
- ❌ Time limits on serverless functions
- ❌ Cannot handle video processing workloads

## Recommended Approach:

Since you want "one link", I recommend **Railway** because:
- ✅ Supports full Python backend with ML models
- ✅ Handles video processing
- ✅ Provides persistent storage
- ✅ Single URL for the entire application
- ✅ Free tier available

Would you like me to help set up Railway deployment instead?