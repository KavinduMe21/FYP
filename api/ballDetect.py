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
        
        frame_count = 0
        detections_found = []
        detected_image_base64 = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
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

        cap.release()
        
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
    return {"message": "Ball Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    # Run slightly differently so it works as a script
    uvicorn.run(app, host="127.0.0.1", port=8000)