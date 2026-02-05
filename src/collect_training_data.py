import cv2
import os
import random
import numpy as np
from src.config import DATASET_DIR, PROCESSED_DATA_DIR

def collect_samples_from_video(video_name):
    video_path = os.path.join(DATASET_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video: {video_name}")
        return

    frame_count = 0
    
    print(f"Processing {video_name}. Press 's' to stop and identify ball, 'q' to skip video, 'ESC' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Resize for consistent processing if needed, but let's keep original for quality for now, 
        # or resize for display.
        display_frame = cv2.resize(frame, (800, 600))
        
        cv2.imshow('Video', display_frame)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27: # ESC
            cap.release()
            cv2.destroyAllWindows()
            return 'exit'
            
        elif key == ord('q'): # Skip video
            break
            
        elif key == ord('s'): # Stop to label
            # Allow user to select ROI
            # Note: SelectROI works on the displayed frame. Scaling factor needed if we resize.
            # Using display_frame for selection
            
            # Select ROI
            roi = cv2.selectROI("Select Ball", display_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Ball")
            
            if roi != (0, 0, 0, 0):
                x, y, w, h = roi
                
                # Save Positive Sample
                # We normalize size (e.g., 64x64) for HOG
                patch = display_frame[y:y+h, x:x+w]
                if patch.size > 0:
                    patch_resized = cv2.resize(patch, (64, 64))
                    save_path = os.path.join(PROCESSED_DATA_DIR, 'positive', f"pos_{video_name}_{frame_count}.jpg")
                    cv2.imwrite(save_path, patch_resized)
                    print(f"Saved positive sample: {save_path}")

                    # Generate Negative Samples (Random crops NOT overlapping with ROI)
                    # We'll generate 5 negatives for every positive
                    for i in range(5):
                        tries = 0
                        while tries < 50:
                            rand_x = random.randint(0, display_frame.shape[1] - 64)
                            rand_y = random.randint(0, display_frame.shape[0] - 64)
                            
                            # Simple IOU or distance check. 
                            # If the random patch is far enough from ROI.
                            # ROI center:
                            roi_cx, roi_cy = x + w//2, y + h//2
                            rand_cx, rand_cy = rand_x + 32, rand_y + 32
                            
                            dist = np.sqrt((roi_cx - rand_cx)**2 + (roi_cy - rand_cy)**2)
                            
                            # If distance is greater than some threshold (approx ball size)
                            if dist > max(w, h) * 2:
                                neg_patch = display_frame[rand_y:rand_y+64, rand_x:rand_x+64]
                                save_path_neg = os.path.join(PROCESSED_DATA_DIR, 'negative', f"neg_{video_name}_{frame_count}_{i}.jpg")
                                cv2.imwrite(save_path_neg, neg_patch)
                                break
                            tries += 1

    cap.release()
    cv2.destroyAllWindows()
    return 'continue'

def main():
    video_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    print(f"Found {len(video_files)} videos.")
    print("Instruction: Watch video. When you see the rugby ball clearly, press 's' to pause and select it.")
    
    for video in video_files:
        result = collect_samples_from_video(video)
        if result == 'exit':
            break

if __name__ == "__main__":
    main()
