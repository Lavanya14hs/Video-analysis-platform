# Deployment Guide

## Option 1: Railway (Recommended for Full-Stack)

Railway supports Python backends with ML models and provides persistent storage.

### Steps:
1. Create account at [railway.app](https://railway.app)
2. Install Railway CLI: `npm install -g @railway/cli`
3. Login: `railway login`
4. Initialize: `railway init`
5. Deploy: `railway up`

### Required files for Railway:
- `requirements.txt` (Python dependencies)
- `Procfile` or `railway.json` for startup
- Environment variables for any secrets

## Option 2: Render

Render supports full-stack applications with persistent disks.

### Steps:
1. Create account at [render.com](https://render.com)
2. Connect your GitHub repository
3. Create a Web Service for the backend
4. Create a Static Site for the frontend
5. Configure environment variables

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