"""
VisionPort Task 6: Detect cars in a traffic video using YOLO.
Expected Output: Car detection with bounding boxes
"""

import os
import sys
import time
import cv2
from ultralytics import YOLO

def detect_cars(video_path: str = "traffic_sample.mp4",
                model_path: str = "yolov8n.pt",
                conf_threshold: float = 0.35,
                max_frames: int = 150):
    print("=" * 60)
    print("VisionPort Task 6: Detect Cars in Traffic Video Using YOLOv8")
    print("=" * 60)

    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    CAR_CLASS_ID = 2  # COCO class 2 = car

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_name = "VisionPort Task 6 - Car Detection (YOLOv8)"
    print(f"[INFO] Processing video: {width}x{height} @ {fps:.1f} FPS (Target: Cars Only)")
    print("[INFO] Press 'q' to quit early")

    frame_idx = 0
    total_cars_detected = 0
    saved_sample = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run YOLO inference filtering for car class
        results = model(frame, classes=[CAR_CLASS_ID], conf=conf_threshold, verbose=False)[0]

        cars_in_frame = 0
        annotated_frame = frame.copy()

        # Draw green bounding boxes and confidence scores
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            cars_in_frame += 1

            # Bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label banner
            label = f"Car: {conf * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + tw + 6, y1), (0, 200, 0), -1)
            cv2.putText(annotated_frame, label, (x1 + 3, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        total_cars_detected += cars_in_frame

        # Top HUD Dashboard
        hud = annotated_frame.copy()
        cv2.rectangle(hud, (10, 10), (320, 75), (20, 20, 20), -1)
        cv2.addWeighted(hud, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        cv2.putText(annotated_frame, "VisionPort: Car Detection", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
        cv2.putText(annotated_frame, f"Cars in Frame: {cars_in_frame} | Total Logged: {total_cars_detected}", (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save first frame with detections as verification image
        if not saved_sample and cars_in_frame > 0:
            sample_output = "task6_cars_detected.jpg"
            cv2.imwrite(sample_output, annotated_frame)
            print(f"[OK] Saved detection snapshot to: {sample_output}")
            saved_sample = True

        if frame_idx % 15 == 0:
            print(f"  Frame {frame_idx:03d}/{total_video_frames}: Detected {cars_in_frame} cars")

        try:
            cv2.imshow(window_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            pass

        if frame_idx >= max_frames:
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print("-" * 60)
    print(f"[SUCCESS] Task 6 complete: Processed {frame_idx} frames. Car detections logged successfully.")

if __name__ == "__main__":
    detect_cars()
