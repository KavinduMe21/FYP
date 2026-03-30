import cv2
import numpy as np
import os
import shutil
import base64
import sys
import uuid
import time
import torch
import torch.nn as nn
import torchvision.models.video as video_models
from collections import deque
from ultralytics import YOLO

# --- ADD THIS TO FIND 'src' FOLDER ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -------------------------------------

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Knock-On Detection API", version="1.0.0")

# ═══════════════════════════════════════════════════════════════════
#  Security Configuration
# ═══════════════════════════════════════════════════════════════════
API_KEY = os.environ.get("API_KEY", "knock-on-detect-2026")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000").split(",")
MAX_FILE_SIZE_MB = 100
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── API Key verification ───────────────────────────────────────────
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


# ── File validation helpers ────────────────────────────────────────
def sanitize_filename(filename: str) -> str:
    """Strip path components and generate a safe temp filename."""
    safe_name = os.path.basename(filename)
    _, ext = os.path.splitext(safe_name)
    return f"temp_{uuid.uuid4().hex}{ext}"


def validate_video_file(filename: str, file_size: int):
    """Validate file extension and size."""
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB} MB"
        )

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ═══════════════════════════════════════════════════════════════════
#  1. Load YOLO Ball Detector
# ═══════════════════════════════════════════════════════════════════
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
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    ball_ids = [k for k, v in model.names.items() if 'ball' in v.lower()]
    if ball_ids:
        target_classes = ball_ids
        print(f"Filtering for ball classes: {target_classes}")
else:
    print("Warning: No YOLO model found.")

# ═══════════════════════════════════════════════════════════════════
#  2. Load R3D-18 Knock-On Classifier
# ═══════════════════════════════════════════════════════════════════
CLASSIFIER_PATH = os.path.join(src_dir, 'knock_on_classifier.pt')
CLASS_NAMES = ["normal_play", "knock_on"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = None
if os.path.exists(CLASSIFIER_PATH):
    try:
        classifier = video_models.r3d_18(weights=None)
        in_features = classifier.fc.in_features
        classifier.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 2),
        )
        checkpoint = torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=True)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        classifier.to(DEVICE)
        classifier.eval()
        val_acc = checkpoint.get("val_acc", 0)
        print(f"[R3D-18] Loaded classifier (val acc {val_acc:.1%})")
    except Exception as e:
        print(f"Warning: Failed to load R3D-18 classifier: {e}")
        classifier = None
else:
    print("Warning: R3D-18 classifier not found at", CLASSIFIER_PATH)


# ═══════════════════════════════════════════════════════════════════
#  Helper functions (from inference.py)
# ═══════════════════════════════════════════════════════════════════
def detect_ball(yolo, frame, ball_ids, conf=0.25):
    """Run YOLO on a single frame. Returns (cx, cy, w, h, conf) or None."""
    results = yolo(frame, conf=conf, verbose=False, classes=ball_ids)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None
    best_idx = boxes.conf.argmax()
    box = boxes.xyxy[best_idx].cpu().numpy()
    c = boxes.conf[best_idx].cpu().item()
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (cx, cy, w, h, c)


def check_ball_drop(ball_history, frame_h):
    """
    Analyse ball positions across the 16-frame window to detect a
    ball-drop pattern consistent with a knock-on.
    Returns (is_ball_drop, reason_str)
    """
    n = len(ball_history)
    if n == 0:
        return False, "no_ball_data"

    half = n // 2
    first_half = ball_history[:half]
    second_half = ball_history[half:]

    first_visible = sum(1 for b in first_half if b is not None)
    second_visible = sum(1 for b in second_half if b is not None)

    first_ys = [b[1] for b in first_half if b is not None]
    second_ys = [b[1] for b in second_half if b is not None]

    # Pattern 1: Ball visible then disappears with downward trend
    if first_visible >= half * 0.5 and second_visible <= half * 0.25:
        if len(first_ys) >= 2:
            early_y = np.mean(first_ys[:len(first_ys) // 2]) if len(first_ys) >= 2 else first_ys[0]
            late_y = np.mean(first_ys[len(first_ys) // 2:])
            if late_y > early_y + frame_h * 0.02:
                return True, "ball_dropped_then_lost"

    # Pattern 2: Ball drops significantly (y increases)
    if first_ys and second_ys:
        avg_y_first = np.mean(first_ys)
        avg_y_second = np.mean(second_ys)
        drop = avg_y_second - avg_y_first
        if drop > frame_h * 0.12:
            return True, "ball_dropped"

    # Pattern 3: Ball was visible then lost
    if first_visible >= half * 0.5 and second_visible <= half * 0.25:
        return True, "ball_lost_after_contact"

    return False, "ball_stable"


def preprocess_clip(frames, frame_size=224):
    """Convert a list of 16 BGR frames into a model-ready tensor."""
    processed = []
    for f in frames:
        f = cv2.resize(f, (frame_size, frame_size))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        processed.append(f)

    clip = np.stack(processed).astype(np.float32) / 255.0
    clip = np.transpose(clip, (3, 0, 1, 2))

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)
    clip = (clip - mean) / std

    return torch.FloatTensor(clip).unsqueeze(0)

@app.post("/detect")
async def detect_knock_on(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    if model is None:
        raise HTTPException(status_code=503, detail="YOLO ball detection model is not loaded.")

    if classifier is None:
        raise HTTPException(status_code=503, detail="R3D-18 knock-on classifier is not loaded.")

    # ── Validate uploaded file ─────────────────────────────────────
    contents = await file.read()
    validate_video_file(file.filename, len(contents))
    await file.seek(0)

    # ── Save with sanitized filename ───────────────────────────────
    temp_filename = sanitize_filename(file.filename)
    with open(temp_filename, "wb") as buffer:
        buffer.write(contents)

    try:
        cap = cv2.VideoCapture(temp_filename)
        if not cap.isOpened():
            return {"error": "Could not open the uploaded video file."}

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── Sliding window settings (same as inference.py) ─────────
        BUFFER_SIZE = 16
        STRIDE = 4
        COOLDOWN_MAX = 90
        THRESHOLD = 0.85

        frame_buffer = deque(maxlen=BUFFER_SIZE)
        ball_buffer = deque(maxlen=BUFFER_SIZE)
        frame_count = 0
        cooldown = 0
        events = []
        detected_image_base64 = None
        detected_detail = ""

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_buffer.append(frame.copy())

            # ── Stage 1: YOLO ball detection on every frame ────────
            current_ball = detect_ball(model, frame, target_classes, conf=0.25)
            ball_buffer.append(current_ball)

            if cooldown > 0:
                cooldown -= 1

            # ── Stage 2: R3D-18 + ball-drop gate ──────────────────
            if (frame_count % STRIDE == 0
                    and frame_count > BUFFER_SIZE
                    and len(frame_buffer) >= BUFFER_SIZE
                    and cooldown == 0):

                clip_tensor = preprocess_clip(list(frame_buffer), frame_size=224)
                clip_tensor = clip_tensor.to(DEVICE)

                with torch.no_grad():
                    output = classifier(clip_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence = probs[0, 1].item()

                if confidence >= THRESHOLD:
                    knock_reason = ""

                    # High confidence (>=90%) — trust R3D-18 directly
                    if confidence >= 0.90:
                        knock_reason = "high_confidence"
                    else:
                        # Borderline — use ball-drop to verify
                        is_drop, ball_reason = check_ball_drop(
                            list(ball_buffer), frame_h
                        )
                        if is_drop:
                            knock_reason = ball_reason
                        else:
                            knock_reason = ""

                    if knock_reason:
                        cooldown = COOLDOWN_MAX

                        total_seconds = int(frame_count / fps)
                        minutes = total_seconds // 60
                        seconds = total_seconds % 60
                        timestamp = f"{minutes:02d}:{seconds:02d}"

                        event = {
                            "frame": frame_count,
                            "timestamp": timestamp,
                            "confidence": round(confidence, 4),
                            "reason": knock_reason,
                        }
                        events.append(event)

                        # Capture evidence image on first detection
                        if detected_image_base64 is None:
                            results = model(frame, conf=0.25, verbose=False, classes=target_classes)
                            annotated_frame = results[0].plot()
                            _, buffer_img = cv2.imencode('.jpg', annotated_frame)
                            detected_image_base64 = base64.b64encode(buffer_img).decode('utf-8')

        cap.release()

        # ── Build detected_detail ──────────────────────────────────
        is_knock_on = len(events) > 0

        # Map technical reason codes to human-readable descriptions
        REASON_LABELS = {
            "ball_dropped": "Handling Error - Ball dropped toward the ground",
            "ball_dropped_then_lost": "Handling Error - Ball dropped then lost from view",
            "ball_lost_after_contact": "Handling Error - Ball lost after player contact",
            "high_confidence": "Handling Error - Knock-on motion detected with high confidence",
        }

        if is_knock_on:
            # Use the first detection's reason as the primary detail
            first_reason = events[0]["reason"]
            detected_detail = REASON_LABELS.get(first_reason, "Handling Error")

            # Build full list of reasons for all events
            reason_summary = []
            for e in events:
                label = REASON_LABELS.get(e["reason"], "Handling Error")
                if label not in reason_summary:
                    reason_summary.append(label)

            detected_events = {
                "knock_on_count": len(events),
                "reasons": reason_summary,
                "events": events,
            }
        else:
            detected_detail = "No knock-on detected"
            detected_events = {
                "knock_on_count": 0,
                "reasons": [],
                "events": [],
            }

        return {
            "filename": file.filename,
            "total_frames": frame_count,
            "event_detected": is_knock_on,
            "detected_image": detected_image_base64,
            "detected_detail": detected_detail,
            "detected_events": detected_events,
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

if __name__ == "__main__":
    import uvicorn
    # Run slightly differently so it works as a script
    uvicorn.run(app, host="127.0.0.1", port=8000)