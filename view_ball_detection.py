"""
Quick visual test for ball detection
Shows detections in a popup window with bounding boxes
"""
import cv2
import os
import sys
from ultralytics import YOLO
from tkinter import Tk, filedialog

def select_video(default_path="D:\\Dataset"):
    """Open file dialog to select a video"""
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    # Set initial directory
    initial_dir = default_path if os.path.exists(default_path) else os.getcwd()
    
    print(f"📁 Opening file picker (starting in: {initial_dir})")
    
    video_path = filedialog.askopenfilename(
        title="Select a video file",
        initialdir=initial_dir,
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("MP4 files", "*.mp4"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return video_path

def visualize_detection(video_path):
    # Load model
    model_path = os.path.join('src', 'best.pt')
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return
    
    print(f"✓ Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Find ball class
    ball_ids = [k for k, v in model.names.items() if 'ball' in v.lower()]
    target_classes = ball_ids if ball_ids else None
    
    if ball_ids:
        print(f"✓ Filtering for ball class: {[model.names[i] for i in ball_ids]}")
    else:
        print(f"⚠ Available classes: {list(model.names.values())}")
    
    # Open video
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✓ Video loaded: {total_frames} frames @ {fps}fps")
    print("\nPress 'q' to quit, 'p' to pause/play")
    print("-" * 50)
    
    cv2.namedWindow('Ball Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Ball Detection', 1280, 720)
    
    frame_num = 0
    paused = False
    detections = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✓ Video finished")
                break
            frame_num += 1
        
        # Detect
        results = model(frame, conf=0.25, verbose=False, classes=target_classes)
        annotated = results[0].plot()
        
        # Count detections
        if len(results[0].boxes) > 0:
            detections += 1
            for box in results[0].boxes:
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                print(f"Frame {frame_num}: {model.names[cls]} detected (conf: {conf:.2f})")
        
        # Add info overlay
        info = f"Frame: {frame_num}/{total_frames} | Detections: {detections}"
        cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        cv2.imshow('Ball Detection', annotated)
        
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDetected ball in {detections}/{frame_num} frames ({detections/frame_num*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # No video provided - open file picker
        print("=" * 60)
        print("🏉 BALL DETECTION VIEWER")
        print("=" * 60)
        video_path = select_video("D:\\Dataset")
        
        if not video_path:
            print("❌ No video selected. Exiting.")
        else:
            print(f"✓ Selected: {os.path.basename(video_path)}")
            visualize_detection(video_path)
    else:
        # Video path provided as argument
        visualize_detection(sys.argv[1])
