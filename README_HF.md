---
title: Rugby Knock-On Detector API
emoji: 🏉
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# Rugby Knock-On Detector API

This is a FastAPI backend for detecting knock-on events in rugby videos using YOLO object detection.

## API Endpoints

### POST /detect
Upload a rugby video to detect knock-on events.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (video file)

**Response:**
```json
{
  "event_detected": true,
  "total_frames": 300,
  "frames_processed": 60,
  "detections_count": 15,
  "first_detection_frame": 45,
  "detected_image": "base64_encoded_image"
}
```

### GET /health
Check API health status.

## Usage

```bash
curl -X POST "https://your-space.hf.space/detect" \
  -F "file=@rugby_video.mp4"
```
