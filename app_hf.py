import cv2
import numpy as np
import os
import base64
import shutil
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model_path = os.path.join('src', 'best.pt')

model = None
target_classes = None

if os.path.exists(model_path):
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    ball_ids = [k for k, v in model.names.items() if 'ball' in v.lower()]
    if ball_ids:
        target_classes = ball_ids
        print(f"Filtering for ball classes: {target_classes}")
else:
    print("Warning: No model found at", model_path)

@app.get("/")
async def root():
    return {"message": "Rugby Knock-On Detector API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": model_path
    }

@app.post("/detect")
async def detect_knock_on(file: UploadFile = File(...)):
    """Process video and detect knock-on events"""
    
    if model is None:
        return {"error": "Model is not loaded.", "event_detected": False}
    
    # Save the uploaded video temporarily
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(temp_filename)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Process every Nth frame for optimization
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
            
            # Run YOLO detection
            results = model(frame, conf=0.25, verbose=False, classes=target_classes)
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                current_boxes = boxes.xyxy.cpu().numpy().tolist()
                
                detections_found.append({
                    "frame": frame_count,
                    "box_count": len(current_boxes),
                })

                # Capture first detection frame
                if detected_image_base64 is None:
                    annotated_frame = results[0].plot()
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    detected_image_base64 = base64.b64encode(buffer).decode('utf-8')

        cap.release()
        
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        if len(detections_found) > 0:
            return {
                "event_detected": True,
                "total_frames": frame_count,
                "frames_processed": frames_processed,
                "detections_count": len(detections_found),
                "first_detection_frame": detections_found[0]['frame'],
                "detected_image": detected_image_base64
            }
        else:
            return {
                "event_detected": False,
                "total_frames": frame_count,
                "frames_processed": frames_processed,
                "message": "No knock-on event detected"
            }
            
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return {
            "error": str(e),
            "event_detected": False
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
