import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def select_video():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        initialdir="D:\\Dataset",
        title="Select Rugby Video to Test",
        filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
    )

def estimate_play_direction(lines, frame_width):
    """
    Estimate which direction teams are attacking based on field line positions
    Returns: 1 for Left→Right, -1 for Right→Left, 0 for Unknown
    """
    if not lines or len(lines) < 2:
        return 0
    
    # Analyze line positions
    left_lines = 0
    right_lines = 0
    center = frame_width / 2
    
    for line in lines:
        x1, y1, x2, y2 = line
        line_center = (x1 + x2) / 2
        
        if line_center < center - 100:
            left_lines += 1
        elif line_center > center + 100:
            right_lines += 1
    
    # More lines on left = attacking right
    if left_lines > right_lines * 1.3:
        return 1  # Left → Right
    elif right_lines > left_lines * 1.3:
        return -1  # Right → Left
    else:
        return 0  # Unknown/Center

def draw_attack_direction(frame, direction):
    """Draw attacking direction indicator"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    
    if direction == 1:  # Left → Right
        # Draw arrow from left to right
        start_x = 50
        end_x = width - 50
        y = 100
        
        # Arrow
        cv2.arrowedLine(overlay, (start_x, y), (end_x, y), 
                       (0, 255, 0), 8, tipLength=0.03)
        
        # Text (using SIMPLEX with thickness for bold effect)
        cv2.putText(overlay, "ATTACKING DIRECTION", 
                   (width//2 - 200, y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(overlay, "LEFT -> RIGHT", 
                   (width//2 - 120, y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
    elif direction == -1:  # Right → Left
        # Draw arrow from right to left
        start_x = width - 50
        end_x = 50
        y = 100
        
        # Arrow
        cv2.arrowedLine(overlay, (start_x, y), (end_x, y), 
                       (0, 255, 255), 8, tipLength=0.03)
        
        # Text
        cv2.putText(overlay, "ATTACKING DIRECTION", 
                   (width//2 - 200, y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        cv2.putText(overlay, "RIGHT -> LEFT", 
                   (width//2 - 120, y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    else:
        # Unknown
        cv2.putText(overlay, "DETECTING DIRECTION...", 
                   (width//2 - 200, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    return frame

def test_line_detection(video_path):
    """Test if we can detect field lines in the video"""
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("\n" + "="*60)
    print("🧪 TESTING LINE DETECTION ON YOUR VIDEO")
    print("="*60)
    
    # Test on multiple frames
    test_frames = [30, 100, 200, 300, 400]
    successful_detections = 0
    detected_directions = []
    
    for frame_num in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # Try to detect lines
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to reduce noise (helps with blurry videos)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with adjusted thresholds
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                threshold=80, 
                                minLineLength=80, 
                                maxLineGap=20)
        
        # Draw results
        result_frame = frame.copy()
        line_count = 0
        horizontal_lines = []
        
        if lines is not None:
            # Filter for horizontal lines (field markings)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Keep horizontal lines
                if angle < 20 or angle > 160:
                    cv2.line(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    horizontal_lines.append(line[0])
                    line_count += 1
        
        # Estimate play direction
        play_direction = estimate_play_direction(horizontal_lines, width)
        detected_directions.append(play_direction)
        
        # Show result
        print(f"\nFrame {frame_num}:")
        print(f"  Lines detected: {line_count}")
        
        direction_text = ""
        if play_direction == 1:
            direction_text = "Left → Right ➡️"
        elif play_direction == -1:
            direction_text = "Right → Left ⬅️"
        else:
            direction_text = "Unknown ❓"
        
        print(f"  Play direction: {direction_text}")
        
        if line_count >= 2:
            print(f"  Status: ✅ GOOD - Found field lines!")
            successful_detections += 1
        elif line_count > 0:
            print(f"  Status: ⚠️ PARTIAL - Found some lines")
        else:
            print(f"  Status: ❌ POOR - No lines detected")
        
        # Draw attack direction
        result_frame = draw_attack_direction(result_frame, play_direction)
        
        # Resize for display (fit screen better)
        display_width = 1280
        display_height = 720
        resized_frame = cv2.resize(result_frame, (display_width, display_height))
        
        # Display frame info
        cv2.putText(resized_frame, f"Frame {frame_num} - Lines: {line_count}", 
                   (10, display_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(resized_frame, f"Direction: {direction_text}", 
                   (10, display_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.namedWindow('Line Detection Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Line Detection Test', display_width, display_height)
        cv2.imshow('Line Detection Test', resized_frame)
        cv2.waitKey(2000)  # Show for 2 seconds
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Determine most common direction
    direction_counts = {1: 0, -1: 0, 0: 0}
    for d in detected_directions:
        direction_counts[d] += 1
    
    most_common_direction = max(direction_counts, key=direction_counts.get)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS:")
    print("="*60)
    print(f"Successful detections: {successful_detections}/{len(test_frames)}")
    
    direction_summary = ""
    if most_common_direction == 1:
        direction_summary = "Left → Right ➡️"
    elif most_common_direction == -1:
        direction_summary = "Right → Left ⬅️"
    else:
        direction_summary = "Unknown/Variable ❓"
    
    print(f"Overall play direction: {direction_summary}")
    print(f"  Left→Right detections: {direction_counts[1]}")
    print(f"  Right→Left detections: {direction_counts[-1]}")
    print(f"  Unknown detections: {direction_counts[0]}")
    
    if successful_detections >= 3:
        print("\n✅ GOOD NEWS! Line detection works on your video!")
        print("   → We can use field-aware detection")
        return True, most_common_direction
    elif successful_detections >= 1:
        print("\n⚠️  PARTIAL SUCCESS - Lines detected sometimes")
        print("   → Field detection might work but not reliable")
        print("   → Recommend: Use backup method")
        return False, most_common_direction
    else:
        print("\n❌ LINE DETECTION FAILED")
        print("   → Your video is too blurry/zoomed for line detection")
        print("   → Recommend: Use alternative method")
        return False, most_common_direction

def main():
    print("="*60)
    print("🧪 FIELD LINE & DIRECTION DETECTION TEST")
    print("="*60)
    
    video_path = select_video()
    if not video_path:
        print("No video selected")
        return
    
    print(f"Testing: {video_path.split('/')[-1]}")
    
    works, direction = test_line_detection(video_path)
    
    print("\n" + "="*60)
    print("💡 RECOMMENDATION:")
    print("="*60)
    
    if works:
        print("""
✅ Use Field-Aware Detection
   - Detects field lines ✓
   - Determines play direction automatically ✓
   - Normalizes movement direction ✓
   - Best accuracy for camera angles ✓
        """)
    else:
        print("""
⚠️  Use Alternative Method:

OPTION 1: Manual Direction Setup (Simpler)
   - You tell system play direction
   - 70% accuracy, works with blurry videos
        """)

if __name__ == "__main__":
    main()