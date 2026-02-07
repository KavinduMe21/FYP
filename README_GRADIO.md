---
title: Rugby Knock-On Detector
emoji: 🏉
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
python_version: 3.11
app_file: app.py
pinned: false
license: mit
---

# 🏉 Rugby Knock-On Detector

An AI-powered system to detect knock-on events in rugby videos using YOLO object detection.

## How to Use

1. Upload a rugby video (MP4, AVI, MOV format)
2. Click "Submit" to process the video
3. View detection results and evidence frame

## Features

- 🎥 Video upload and processing
- 🤖 YOLOv8-based ball detection
- 🎯 Automatic knock-on event detection
- 📸 Evidence frame with annotations
- ⚡ Optimized rame-by-frame analysis

## Technical Details

- **Model**: YOLOv8 trained on rugby ball detection
- **Framework**: Gradio for easy web interface
- **Processing**: Analyzes each frame for ball presence
- **Output**: Detection summary + annotated evidence frame

## About

This system uses computer vision and deep learning to automatically detect knock-on events in rugby footage, helping referees and analysts review game incidents.

## License

MIT License
