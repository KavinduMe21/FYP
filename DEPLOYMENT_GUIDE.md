# 🏉 Rugby Knock-On Detector - Deployment Guide

## Architecture Overview

- **Backend + Model**: Hugging Face Spaces (FastAPI + YOLO)
- **Frontend**: Vercel (Static HTML)
- **Communication**: Frontend calls Hugging Face API endpoints

---

## 📦 Part 1: Deploy Backend to Hugging Face Spaces

### Step 1: Create a New Hugging Face Space

1. Go to [Hugging Face](https://huggingface.co/)
2. Sign in to your account
3. Click on your profile → **New Space**
4. Configure your space:
   - **Name**: `rugby-knock-on-detector` (or any name you prefer)
   - **SDK**: Select **Docker**
   - **Visibility**: Public or Private
   - Click **Create Space**

### Step 2: Prepare Files for Upload

Create a new folder with these files:

```
huggingface-deployment/
├── app_hf.py           (Main FastAPI application)
├── Dockerfile          (Docker configuration)
├── requirements.txt    (Python dependencies)
├── README_HF.md       (Rename to README.md for HF)
└── src/
    └── best.pt        (Your trained YOLO model)
```

**Files to copy:**
- Copy `app_hf.py` from your project root
- Copy `Dockerfile` from your project root
- Copy `requirements.txt` from your project root
- Copy `src/best.pt` model file
- Rename `README_HF.md` to `README.md`

### Step 3: Upload to Hugging Face

**Option A: Using Git (Recommended)**

```bash
# Clone your space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/rugby-knock-on-detector
cd rugby-knock-on-detector

# Copy your files
cp path/to/app_hf.py ./
cp path/to/Dockerfile ./
cp path/to/requirements.txt ./
cp path/to/README_HF.md ./README.md
cp -r path/to/src ./

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

**Option B: Using Web Interface**

1. In your Hugging Face Space, click **Files** → **Add file**
2. Upload each file one by one:
   - `app_hf.py`
   - `Dockerfile`
   - `requirements.txt`
   - `README.md` (the renamed README_HF.md)
3. Create `src/` folder and upload `best.pt`

### Step 4: Wait for Build

- Hugging Face will automatically build your Docker container
- This may take 5-10 minutes
- Check the **Build logs** tab for progress
- Once complete, you'll see "Running" status

### Step 5: Test Your API

Your API will be available at: `https://YOUR_USERNAME-rugby-knock-on-detector.hf.space`

Test it:
```bash
# Health check
curl https://YOUR_USERNAME-rugby-knock-on-detector.hf.space/health

# Expected response:
# {"status":"healthy","model_loaded":true,"model_path":"src/best.pt"}
```

**📝 Copy your Hugging Face Space URL - you'll need it for the frontend!**

---

## 🌐 Part 2: Deploy Frontend to Vercel

### Step 1: Prepare Frontend Files

Create a folder structure:

```
frontend/
├── index_vercel.html
└── README.md
```

### Step 2: Update API URL in Frontend

1. Open `frontend/index_vercel.html`
2. Find line: `const API_URL = 'YOUR_HUGGING_FACE_SPACE_URL';`
3. Replace with your actual Hugging Face Space URL:
   ```javascript
   const API_URL = 'https://YOUR_USERNAME-rugby-knock-on-detector.hf.space';
   ```
4. Save the file

### Step 3: Create GitHub Repository (if not already)

```bash
# Initialize git in your project
git init
git add .
git commit -m "Initial commit"

# Create repo on GitHub and push
git remote add origin https://github.com/YOUR_USERNAME/rugby-detector.git
git branch -M main
git push -u origin main
```

### Step 4: Deploy to Vercel

**Option A: Using Vercel CLI**

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
vercel --prod
```

**Option B: Using Vercel Web Dashboard (Easier)**

1. Go to [Vercel](https://vercel.com/)
2. Sign in with GitHub
3. Click **Add New** → **Project**
4. Import your GitHub repository
5. Configure:
   - **Framework Preset**: Other
   - **Root Directory**: `./` (or `./frontend` if you want)
   - **Build Command**: Leave empty
   - **Output Directory**: `frontend`
6. Click **Deploy**

### Step 5: Vercel Project Settings

If your `index_vercel.html` is not at root:

1. Go to your project settings in Vercel
2. Navigate to **Settings** → **General**
3. Set **Root Directory** to `frontend`
4. Save and redeploy

### Step 6: Test Your Deployment

1. Vercel will provide a URL like: `https://your-project.vercel.app`
2. Visit the URL
3. You should see: "✅ API Connected & Model Ready"
4. Upload a test video and verify detection works

---

## 🔧 Troubleshooting

### Hugging Face Issues

**Build Fails:**
- Check `Build logs` in HF Space
- Ensure `Dockerfile` and `requirements.txt` are correct
- Verify `src/best.pt` model file exists

**API Returns 500 Error:**
- Check if model is loaded: `https://your-space.hf.space/health`
- Ensure `src/best.pt` path is correct in `app_hf.py`

**CORS Errors:**
- The `app_hf.py` allows all origins (`allow_origins=["*"]`)
- For production, update to specific Vercel domain:
  ```python
  allow_origins=["https://your-project.vercel.app"]
  ```

### Vercel Issues

**404 Page Not Found:**
- Check `vercel.json` configuration
- Ensure `index_vercel.html` is in the correct directory
- Try setting root directory in Vercel project settings

**API Connection Failed:**
- Verify `API_URL` in `index_vercel.html` is correct
- Check browser console for CORS errors
- Test Hugging Face API directly with curl

**Cannot Connect to API:**
- Open browser DevTools (F12) → Console
- Check for error messages
- Verify Hugging Face Space is "Running" (not "Sleeping")

---

## 📝 Environment Variables (Optional)

### For More Secure API URL Management

**Vercel:**
1. Go to Project Settings → Environment Variables
2. Add: `NEXT_PUBLIC_API_URL` = `https://your-space.hf.space`
3. Update `index_vercel.html`:
   ```javascript
   const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://fallback-url.hf.space';
   ```

---

## 🚀 Production Checklist

### Before Going Live:

- [ ] Test video upload with multiple video formats
- [ ] Verify detection accuracy
- [ ] Test on mobile devices
- [ ] Update CORS to specific domain (not "*")
- [ ] Add error logging/monitoring
- [ ] Set up Hugging Face Space persistence (if needed)
- [ ] Consider upgrading HF Space for better performance
- [ ] Add rate limiting to prevent abuse
- [ ] Document API usage for future reference

---

## 📊 Cost & Performance

### Hugging Face Spaces:
- **Free Tier**: CPU-only, may sleep after inactivity
- **Upgraded**: GPU access, persistent, faster processing
- Model loading: ~30 seconds on first request
- Video processing: ~10-30 seconds depending on length

### Vercel:
- **Free Tier**: Sufficient for most static sites
- Unlimited bandwidth on Hobby plan
- Fast global CDN

---

## 🔄 Updates & Maintenance

### Updating Backend (Hugging Face):
```bash
cd huggingface-deployment
git pull
# Make changes to app_hf.py
git add .
git commit -m "Update detection logic"
git push
```

### Updating Frontend (Vercel):
```bash
# Make changes to index_vercel.html
git add .
git commit -m "Update UI"
git push
# Vercel auto-deploys on push
```

---

## 📱 Final URLs

After deployment, you'll have:

- **Backend API**: `https://YOUR_USERNAME-rugby-knock-on-detector.hf.space`
- **Frontend**: `https://your-project.vercel.app`

Share the Vercel URL with users - they don't need to know about the backend!

---

## 🎉 Success!

Your Rugby Knock-On Detector is now live with:
- ✅ Fast backend on Hugging Face
- ✅ Global frontend on Vercel
- ✅ Smooth API integration
- ✅ Professional deployment

For support, check the logs in both platforms and refer to their documentation.
