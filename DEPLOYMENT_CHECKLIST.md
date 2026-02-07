# Quick Deployment Checklist

## Files Created for Your New Deployment Architecture:

### For Hugging Face Spaces (Backend):
- ✅ `app_hf.py` - FastAPI backend optimized for HF Spaces
- ✅ `Dockerfile` - Docker configuration for HF deployment
- ✅ `README_HF.md` - Documentation for HF Space (rename to README.md when uploading)
- ✅ Use existing `requirements.txt` and `src/best.pt`

### For Vercel (Frontend):
- ✅ `frontend/index_vercel.html` - Updated frontend with API URL configuration
- ✅ `vercel.json` - Vercel deployment configuration
- ✅ `.vercelignore` - Files to ignore in Vercel deployment

### Documentation:
- ✅ `DEPLOYMENT_GUIDE.md` - Complete step-by-step deployment guide

---

## Quick Start:

### 1. Deploy to Hugging Face (5 mins)
1. Create new Space on huggingface.co (Docker SDK)
2. Upload: `app_hf.py`, `Dockerfile`, `requirements.txt`, `src/best.pt`
3. Wait for build to complete
4. Copy your Space URL (e.g., `https://username-rugby-detector.hf.space`)

### 2. Update Frontend (1 min)
Open `frontend/index_vercel.html` and update:
```javascript
const API_URL = 'YOUR_HUGGING_FACE_SPACE_URL';
```

### 3. Deploy to Vercel (3 mins)
1. Push code to GitHub
2. Import project to Vercel
3. Set root directory to `frontend`
4. Deploy!

---

## Need Help?
Read the full [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.
