import cv2
import numpy as np
import os
import gradio as gr
from ultralytics import YOLO

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

def detect_knock_on(video_file):
    """Process video and detect knock-on events"""
    
    if model is None:
        return "❌ Error: Model is not loaded.", None
    
    if video_file is None:
        return "⚠️ Please upload a video file.", None
    
    try:
        cap = cv2.VideoCapture(video_file)
        
        frame_count = 0
        detections_found = []
        detected_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            results = model(frame, conf=0.25, verbose=False, classes=target_classes)
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                current_boxes = boxes.xyxy.cpu().numpy().tolist()
                
                detections_found.append({
                    "frame": frame_count,
                    "box_count": len(current_boxes),
                })

                if detected_frame is None:
                    annotated_frame = results[0].plot()
                    detected_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        cap.release()
        
        if len(detections_found) > 0:
            result_text = f"✅ **Knock-on / Ball Event Detected!**\n\n"
            result_text += f"📊 Total Frames: {frame_count}\n"
            result_text += f"🎯 Detections in {len(detections_found)} frames\n"
            result_text += f"🏉 First detection at frame {detections_found[0]['frame']}"
            return result_text, detected_frame
        else:
            return f"❌ No event detected.\n\n📊 Total Frames Analyzed: {frame_count}", None
            
    except Exception as e:
        return f"❌ Error processing video: {str(e)}", None

demo = gr.Interface(
    fn=detect_knock_on,
    inputs=gr.Video(label="Upload Rugby Video"),
    outputs=[
        gr.Textbox(label="Detection Result", lines=5),
        gr.Image(label="Evidence Frame (if detected)")
    ],
    title="🏈 Rugby Knock-On Detector",
    description="Upload a rugby video to detect knock-on events using AI. The system will analyze each frame and highlight any ball detections.",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
