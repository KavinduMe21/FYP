---
title: Rugby Knock-On Detector
emoji: 🏉
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# 🏉 Rugby Knock-On Detector

An AI-powered system to detect knock-on events in rugby videos using YOLO object detection.

## Features

- 🎥 Video upload and processing
- 🤖 YOLO-based ball detection  
- 🎯 Automatic knock-on event detection
- 📸 Evidence frame extraction with annotations
- ⚡ Optimized processing (samples frames for faster analysis)

## How to Use

1. Visit the Space and you'll see the web interface
2. Click to upload a rugby video (MP4, AVI, MOV)
3. Click "🔍 Analyze Video"
4. Wait for processing (10-30 seconds depending on video length)
5. View results and detection evidence frame

## API Endpoints

### GET /
Web interface for easy video upload

### POST /detect
Upload a rugby video to detect knock-on events

**Example:**
```bash
curl -X POST "https://your-space.hf.space/detect" \
  -F "file=@rugby_video.mp4"
```

### GET /health
Check API health and model status

## Technical Details

- **Model**: YOLOv8 trained on rugby ball detection
- **Backend**: FastAPI with CORS enabled
- **Processing**: Smart frame sampling (6 fps) for efficiency
- **Output**: Base64-encoded annotated frames

## License

MIT License
