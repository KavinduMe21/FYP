import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models.video as video_models
from ultralytics import YOLO
from collections import deque
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "knock_on_classifier.pt")
YOLO_PATH = os.path.join(BASE_DIR, "best.pt")
CAPTURES_DIR = os.path.join(BASE_DIR, "..", "knock_on_captures")
CLASS_NAMES = ["normal_play", "knock_on"]


def load_classifier(model_path, device):
    model = video_models.r3d_18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2),
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_acc = checkpoint.get("val_acc", 0)
    epoch = checkpoint.get("epoch", "?")
    print(f"Loaded R3D-18 classifier (epoch {epoch}, val acc {val_acc:.1%})")
    return model


def load_ball_detector(yolo_path):
    yolo = YOLO(yolo_path)
    ball_ids = [k for k, v in yolo.names.items() if "ball" in v.lower()]
    if not ball_ids:
        print(f"WARNING: no 'ball' class found in YOLO model. Classes: {list(yolo.names.values())}")
        ball_ids = None
    else:
        print(f"Ball detector loaded, ball class IDs: {ball_ids}")
    return yolo, ball_ids


def detect_ball(yolo, frame, ball_ids, conf=0.25):
    # best ball detection -> (cx, cy, w, h, conf) or None
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

    # ball visible first half, gone second half
    if first_visible >= half * 0.5 and second_visible <= half * 0.25:
        if len(first_ys) >= 2:
            early_y = np.mean(first_ys[:len(first_ys)//2]) if len(first_ys) >= 2 else first_ys[0]
            late_y = np.mean(first_ys[len(first_ys)//2:])
            if late_y > early_y + frame_h * 0.02:
                return True, "ball_dropped_then_lost"

    # big y increase = drop
    if first_ys and second_ys:
        avg_y_first = np.mean(first_ys)
        avg_y_second = np.mean(second_ys)
        drop = avg_y_second - avg_y_first
        if drop > frame_h * 0.12:
            return True, "ball_dropped"

    # visible then gone
    if first_visible >= half * 0.5 and second_visible <= half * 0.25:
        return True, "ball_lost_after_contact"

    return False, "ball_stable"


def preprocess_clip(frames, frame_size=224):
    # prep 16 BGR frames -> tensor for R3D-18
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


def draw_overlay(frame, knock_on, confidence, frame_count, event_count,
                 ball_pos=None, ball_drop=False, ball_reason=""):
    h, w = frame.shape[:2]

    # conf bar
    bar_w = 200
    bar_color = (0, 255, 0) if confidence < 0.3 else (0, 255, 255) if confidence < 0.5 else (0, 0, 255)
    cv2.rectangle(frame, (10, h - 50), (10 + int(bar_w * confidence), h - 35), bar_color, -1)
    cv2.rectangle(frame, (10, h - 50), (10 + bar_w, h - 35), (150, 150, 150), 1)
    cv2.putText(frame, f"Knock-on: {confidence:.0%}",
                (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Knock-ons: {event_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ball crosshair
    if ball_pos is not None:
        cx, cy = int(ball_pos[0]), int(ball_pos[1])
        cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 2)
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
        ball_status = "BALL TRACKED"
        ball_color = (0, 255, 0)
    else:
        ball_status = "BALL LOST"
        ball_color = (0, 0, 255)
    cv2.putText(frame, ball_status, (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 2)

    if knock_on:
        overlay = frame.copy()
        cv2.rectangle(overlay, (w // 4, h // 3), (3 * w // 4, 2 * h // 3 + 30), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, "KNOCK-ON!", (w // 4 + 50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(frame, f"Confidence: {confidence:.0%}  |  Ball: {ball_reason}",
                    (w // 4 + 20, h // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'S' to SAVE  |  SPACE to skip",
                    (w // 4 + 10, h // 2 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


def save_clip(frames, fps, frame_count):
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"knock_on_frame{frame_count}_{timestamp}.mp4"
    filepath = os.path.join(CAPTURES_DIR, filename)

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    print(f"   Saved clip: {filepath}")
    return filepath


def select_video(cli_path):
    if cli_path and os.path.isfile(cli_path):
        return cli_path

    if not HAS_TK:
        print("No video path given and tkinter not available.")
        print("Usage: python inference.py --video path/to/video.mp4")
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    initial_dir = os.path.join(BASE_DIR, "knock_on_dataset")
    if not os.path.isdir(initial_dir):
        initial_dir = BASE_DIR

    path = filedialog.askopenfilename(
        title="Select a Rugby Video for Inference",
        initialdir=initial_dir,
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
    )
    root.destroy()

    if not path:
        print("No video selected.")
        sys.exit(0)
    return path


def main():
    parser = argparse.ArgumentParser(description="Knock-On Inference (YOLO + R3D-18)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to R3D-18 classifier")
    parser.add_argument("--yolo", type=str, default=YOLO_PATH, help="Path to YOLO ball detector")
    parser.add_argument("--threshold", type=float, default=0.85, help="Knock-on confidence threshold")
    parser.add_argument("--ball-conf", type=float, default=0.25, help="YOLO ball detection confidence")
    parser.add_argument("--no-ball-check", action="store_true", help="Disable ball-drop verification")
    parser.add_argument("--save", action="store_true", help="Save annotated output video")
    parser.add_argument("--auto-save", action="store_true", help="Auto-save every knock-on clip")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"Classifier not found: {args.model}")
        print("Train it first with: python train.py")
        return

    use_ball_check = not args.no_ball_check
    yolo = None
    ball_ids = None

    if use_ball_check:
        if not os.path.isfile(args.yolo):
            print(f"Ball detector not found: {args.yolo}")
            print("Running without ball-drop verification.")
            use_ball_check = False
        else:
            yolo, ball_ids = load_ball_detector(args.yolo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classifier = load_classifier(args.model, device)

    video_path = select_video(args.video)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo: {video_path}")
    print(f"Frames: {total_frames}, FPS: {fps:.1f}")
    print(f"Threshold: {args.threshold}, Ball check: {'ON' if use_ball_check else 'OFF'}")
    print("-" * 60)
    print("Controls: Q = Quit | S = Save clip | SPACE = Skip")
    print("-" * 60)

    writer = None
    if args.save:
        out_path = os.path.splitext(video_path)[0] + "_knock_on_annotated.mp4"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"Saving annotated video to: {out_path}")

    BUFFER_SIZE = 24
    STRIDE = 4        # classify every 4th frame
    COOLDOWN_MAX = 90  # min frames between detections

    frame_buffer = deque(maxlen=BUFFER_SIZE)
    ball_buffer = deque(maxlen=BUFFER_SIZE)
    frame_count = 0
    cooldown = 0
    events = []
    saved_count = 0
    current_ball = None

    cv2.namedWindow("Knock-On Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Knock-On Inference", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_buffer.append(frame.copy())

        # ball tracking
        if use_ball_check and yolo is not None:
            current_ball = detect_ball(yolo, frame, ball_ids, conf=args.ball_conf)
        ball_buffer.append(current_ball)

        if cooldown > 0:
            cooldown -= 1

        knock_on = False
        confidence = 0.0
        ball_reason = ""

        # classify every STRIDE frames
        if (frame_count % STRIDE == 0
                and frame_count > BUFFER_SIZE
                and len(frame_buffer) >= BUFFER_SIZE
                and cooldown == 0):

            clip_tensor = preprocess_clip(list(frame_buffer), frame_size=224)
            clip_tensor = clip_tensor.to(device)

            with torch.no_grad():
                output = classifier(clip_tensor)
                probs = torch.softmax(output, dim=1)
                confidence = probs[0, 1].item()

            if confidence >= args.threshold:
                if use_ball_check:
                    # high conf -> skip ball check
                    if confidence >= 0.90:
                        knock_on = True
                        cooldown = COOLDOWN_MAX
                        ball_reason = "high_confidence"
                        events.append({
                            "frame": frame_count,
                            "confidence": confidence,
                            "ball_reason": ball_reason,
                        })
                        print(f"\n  KNOCK-ON at frame {frame_count}  "
                              f"(conf {confidence:.0%}, skipped ball check)")
                    else:
                        # borderline, verify with ball drop
                        is_drop, ball_reason = check_ball_drop(
                            list(ball_buffer), frame_h
                        )
                        if is_drop:
                            knock_on = True
                            cooldown = COOLDOWN_MAX
                            events.append({
                                "frame": frame_count,
                                "confidence": confidence,
                                "ball_reason": ball_reason,
                            })
                            print(f"\n  KNOCK-ON at frame {frame_count}  "
                                  f"(conf {confidence:.0%}, ball: {ball_reason})")
                        else:
                            pass
                else:
                    knock_on = True
                    cooldown = COOLDOWN_MAX
                    ball_reason = "no_ball_check"
                    events.append({
                        "frame": frame_count,
                        "confidence": confidence,
                        "ball_reason": ball_reason,
                    })
                    print(f"\n  KNOCK-ON at frame {frame_count}  "
                          f"(conf {confidence:.0%})")

        display = frame.copy()
        display = draw_overlay(display, knock_on, confidence, frame_count,
                               len(events), ball_pos=current_ball,
                               ball_drop=knock_on, ball_reason=ball_reason)

        if writer:
            writer.write(display)

        # fit to display
        dh, dw = 720, 1280
        h, w = display.shape[:2]
        aspect = w / h
        if aspect > dw / dh:
            nw, nh = dw, int(dw / aspect)
        else:
            nh, nw = dh, int(dh * aspect)
        resized = cv2.resize(display, (nw, nh))
        cv2.imshow("Knock-On Inference", resized)

        if knock_on:
            if args.auto_save:
                save_clip(list(frame_buffer), fps, frame_count)
                saved_count += 1
                key = cv2.waitKey(1500) & 0xFF
            else:
                print("   >> Press 'S' to SAVE, SPACE to skip, 'Q' to quit")
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("s") or key == ord("S"):
                        save_clip(list(frame_buffer), fps, frame_count)
                        saved_count += 1
                        break
                    elif key == ord(" "):
                        print("   >> Skipped.")
                        break
                    elif key == ord("q") or key == ord("Q"):
                        break
                if key == ord("q") or key == ord("Q"):
                    break
        else:
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # summary
    print(f"\n{'='*60}")
    print(f"  INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Pipeline:          {'YOLO + R3D-18' if use_ball_check else 'R3D-18 only'}")
    print(f"Frames processed:  {frame_count}")
    print(f"Knock-ons found:   {len(events)}")
    print(f"Clips saved:       {saved_count}")
    if saved_count:
        print(f"Saved to:          {CAPTURES_DIR}")
    for i, ev in enumerate(events, 1):
        print(f"  #{i}  Frame {ev['frame']}  "
              f"Conf {ev['confidence']:.0%}  "
              f"Ball: {ev.get('ball_reason', 'n/a')}")


if __name__ == "__main__":
    main()
