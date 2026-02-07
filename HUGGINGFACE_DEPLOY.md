# 🚀 Deploy to Hugging Face Spaces - Simple Guide

## Files You Need to Upload

Upload these files to your Hugging Face Space:

### Required Files:
1. ✅ `app_hf.py` - Backend API
2. ✅ `Dockerfile` - Docker configuration  
3. ✅ `requirements.txt` - Python dependencies
4. ✅ `README.md` - Space documentation (already updated with metadata)
5. ✅ `src/best.pt` - Your YOLO model
6. ✅ `frontend/index.html` - Web interface

---

## Step-by-Step Deployment

### Step 1: Create Hugging Face Space

1. Go to https://huggingface.co/
2. Sign in to your account
3. Click **New Space** (top right)
4. Fill in:
   - **Name**: `rugby-knock-on-detector`
   - **SDK**: Select **Docker** (important!)
   - **Visibility**: Public
5. Click **Create Space**

### Step 2: Upload Files

**Option A: Using Git (Recommended)**

```bash
# Clone your new space
git clone https://huggingface.co/spaces/YOUR_USERNAME/rugby-knock-on-detector
cd rugby-knock-on-detector

# Copy files from your project
copy "d:\FYP DUPLICATE\FYP\FYP_Knock_on\app_hf.py" .
copy "d:\FYP DUPLICATE\FYP\FYP_Knock_on\Dockerfile" .
copy "d:\FYP DUPLICATE\FYP\FYP_Knock_on\requirements.txt" .
copy "d:\FYP DUPLICATE\FYP\FYP_Knock_on\README.md" .

# Create directories and copy model + frontend
mkdir src
copy "d:\FYP DUPLICATE\FYP\FYP_Knock_on\src\best.pt" src\
mkdir frontend
copy "d:\FYP DUPLICATE\FYP\FYP_Knock_on\frontend\index.html" frontend\

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

**Option B: Using Web Interface**

1. In your HF Space, click **Files** tab
2. Click **Add file** → **Upload files**
3. Upload these files:
   - `app_hf.py`
   - `Dockerfile`
   - `requirements.txt`
   - `README.md`
4. Click **Create** → **New folder** → name it `src`
5. Upload `best.pt` to the `src` folder
6. Click **Create** → **New folder** → name it `frontend`
7. Upload `index.html` to the `frontend` folder

### Step 3: Wait for Build

- HF will automatically start building (5-10 minutes)
- Watch the **Logs** tab for progress
- Look for: "Application startup complete"
- Status will change to **Running**

### Step 4: Test Your Deployment

Your app will be at: `https://YOUR_USERNAME-rugby-knock-on-detector.hf.space`

**Test it:**
1. Visit the URL
2. You'll see your web interface
3. Upload a rugby video
4. Click "Analyze Video"
5. See results!

---

## Quick File Checklist

Before uploading, verify you have:

```
rugby-knock-on-detector/
├── app_hf.py              ✅ (Created for you)
├── Dockerfile             ✅ (Created for you)
├── requirements.txt       ✅ (Already exists)
├── README.md             ✅ (Updated with HF metadata)
├── src/
│   └── best.pt           ✅ (Your trained model)
└── frontend/
    └── index.html        ✅ (Your web interface)
```

---

## Troubleshooting

### Build Fails
- Check **Logs** tab in HF Space
- Ensure all files are uploaded correctly
- Verify `src/best.pt` exists

### "Model not loaded" Error
- Check if `src/best.pt` is in the correct location
- Look at logs for model loading errors

### Frontend Not Showing
- Verify `frontend/index.html` is uploaded
- Check the `/` route in browser

### Video Upload Fails
- Check browser console (F12) for errors
- Ensure video format is MP4 or AVI
- Try a smaller video file first

---

## After Deployment

Once running, you can:
1. Share the Space URL with anyone
2. Embed it in your website
3. Use the API endpoint from other applications
4. Monitor usage in HF dashboard

Your HF Space URL will be:
```
https://YOUR_USERNAME-rugby-knock-on-detector.hf.space
```

That's it! Your Rugby Knock-On Detector is now live! 🎉
