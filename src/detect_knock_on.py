import cv2
import os
import argparse
from ultralytics import YOLO
from src.config import DATASET_DIR, BASE_DIR

def main():
    parser = argparse.ArgumentParser(description='Detect Knock-on with YOLOv8')
    parser.add_argument('--video', type=str, help='Path to video file', default=None)
    parser.add_argument('--model', type=str, help='Path to YOLO model', default=None)
    parser.add_argument('--conf', type=float, help='Confidence threshold', default=0.25)
    args = parser.parse_args()

    if args.model and os.path.exists(args.model):
        model_path = args.model
    else:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        default_model = os.path.join(src_dir, 'best.pt') 
        if os.path.exists(default_model):
            model_path = default_model
        else:
             possible_models = [
                os.path.join(src_dir, 'best.pt'),
                os.path.join(src_dir, 'best.pt')
             ]
             model_path = None
             for p in possible_models:
                 if os.path.exists(p):
                     model_path = p
                     break
    
    if not model_path:
        print("model not found.")
        return

    print(f"Loading ...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"{e}")
        return

    target_classes = None
    print("Model classes found:", model.names)
    
    ball_ids = []
    for class_id, class_name in model.names.items():
        if 'ball' in class_name.lower():
            ball_ids.append(class_id)
    
    if ball_ids:
        target_classes = ball_ids
        print(f"-> Auto-filtering mainly for ball detection (Class IDs: {ball_ids}).")
    else:
        print("-> 'ball' class not explicitly found in names. Displaying all detections.")

    video_path = args.video
    if not video_path or not os.path.exists(video_path):
        possible_dirs = [
            DATASET_DIR, 
            os.path.join(BASE_DIR, 'videos'),
            r"D:\FYP DUPLICATE\FYP\Dataset"
        ]
        
        found = False
        for d in possible_dirs:
            if os.path.exists(d):
                try:
                    files = [f for f in os.listdir(d) if f.endswith('.mp4')]
                    if files:
                        video_path = os.path.join(d, files[0])
                        print(f"Using video: {video_path}")
                        found = True
                        break
                except Exception as e:
                    print(f"Error accessing directory {d}: {e}")
        
        if not found:
            print("No video found in Dataset or videos folder.")
            return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    print("Press 'q' to quit.")
    
    # Set desired window size
    display_width = 1280
    display_height = 720
    cv2.namedWindow('Knock-on Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Knock-on Detector', display_width, display_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False, classes=target_classes)

        annotated_frame = results[0].plot()
        
        # Resize frame to fit the window while maintaining aspect ratio
        h, w = annotated_frame.shape[:2]
        aspect = w / h
        if aspect > display_width / display_height:
            new_w = display_width
            new_h = int(display_width / aspect)
        else:
            new_h = display_height
            new_w = int(display_height * aspect)
        
        resized_frame = cv2.resize(annotated_frame, (new_w, new_h))

        cv2.imshow('Knock-on Detector', resized_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
