import cv2
import numpy as np
import os
import shutil
import base64
import sys
from ultralytics import YOLO

# --- ADD THIS TO FIND 'src' FOLDER ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -------------------------------------

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# --- ENABLE BROWSER ACCESS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------

# Serve frontend static files
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# 1. Load Model
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
# Try specific model first, then fallback
model_paths = [
    os.path.join(src_dir, 'best.pt')
]

model = None
model_path = None
for mp in model_paths:
    if os.path.exists(mp):
        model_path = mp
        break

target_classes = None

if model_path:
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    # Check for 'ball' class to filter
    ball_ids = [k for k, v in model.names.items() if 'ball' in v.lower()]
    if ball_ids:
        target_classes = ball_ids
        print(f"Filtering for ball classes: {target_classes}")
else:
    print("Warning: No  model found.")

@app.post("/detect")
async def detect_knock_on(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model is not loaded."}

    # 2. Save the uploaded video temporarily
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        cap = cv2.VideoCapture(temp_filename)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # OPTIMIZATION: Process every Nth frame instead of all frames
        # For 30fps video, process every 5th frame = 6fps (still plenty for detection)
        frame_skip = max(1, fps // 6)  # Process ~6 frames per second
        
        frame_count = 0
        frames_processed = 0
        detections_found = []
        detected_image_base64 = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for faster processing
            if frame_count % frame_skip != 0:
                continue
            
            frames_processed += 1
            
            # conf=0.25 is standard, lower if detection is missed
            results = model(frame, conf=0.25, verbose=False, classes=target_classes)
            
            # results[0].boxes contains the detections
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                # Format boxes for JSON response: [x1, y1, x2, y2]
                current_boxes = boxes.xyxy.cpu().numpy().tolist()
                
                detections_found.append({
                    "frame": frame_count,
                    "box_count": len(current_boxes),
                    "boxes": current_boxes
                })

                # --- Capture the FIRST evidence image ---
                if detected_image_base64 is None:
                    # Draw boxes on this specific frame for the evidence image
                    annotated_frame = results[0].plot() 
                    
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    detected_image_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # OPTIMIZATION: Stop after first detection for faster response
                    # Remove this break if you want to scan entire video
                    break

        cap.release()
        
        print(f"Processed {frames_processed} out of {total_frames} frames (skipped {frame_skip-1} of every {frame_skip} frames)")
        
        # 3. Return Results
        is_knock_on = len(detections_found) > 0 
        
        return {
            "filename": file.filename,
            "total_frames": frame_count,
            "event_detected": is_knock_on,
            "detected_image": detected_image_base64
        }
    except Exception as e:
        print(f"Error processing video: {e}")
        return {"error": str(e)}

    finally:
        # 4. Cleanup
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass

@app.get("/")
def root():
    """Serve the frontend HTML"""
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Ball Detection API is running"}

@app.get("/health")
def health_check():
    """Health check endpoint - prevents cold starts"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_classes": list(model.names.values()) if model else []
    }

if __name__ == "__main__":
    import uvicorn
    # Run slightly differently so it works as a script
    uvicorn.run(app, host="127.0.0.1", port=8000)