"""
VisionPort Task 11: Detect traffic congestion at signal junctions.
Expected Output: Congestion alerts (Visual banners, threshold alerts, queue monitoring)
"""

import os
import sys
import cv2
from ultralytics import YOLO

CLASSES = [1, 2, 3, 5, 7]  # vehicles

def detect_congestion(video_path: str = "traffic_sample.mp4",
                      model_path: str = "yolov8n.pt",
                      congestion_threshold: int = 6,
                      conf_threshold: float = 0.35):
    print("=" * 60)
    print("VisionPort Task 11: Traffic Congestion Detection at Signal Junction")
    print("=" * 60)

    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    saved_alert_sample = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model(frame, classes=CLASSES, conf=conf_threshold, verbose=False)[0]
        vehicle_count = len(results.boxes)

        is_congested = (vehicle_count >= congestion_threshold)
        annotated = frame.copy()

        # Draw vehicle boxes
        box_color = (0, 0, 255) if is_congested else (0, 220, 0)
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Congestion Alert Header Banner
        banner_h = 60
        if is_congested:
            banner = np_banner = annotated.copy()
            cv2.rectangle(np_banner, (0, 0), (width, banner_h), (0, 0, 200), -1)
            cv2.addWeighted(np_banner, 0.85, annotated, 0.15, 0, annotated)

            cv2.putText(annotated, "⚠ CONGESTION ALERT: JUNCTION SLOWDOWN DETECTED!", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(annotated, f"Vehicles: {vehicle_count} (Threshold: {congestion_threshold})", (width - 280, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        else:
            banner = np_banner = annotated.copy()
            cv2.rectangle(np_banner, (0, 0), (width, banner_h), (20, 100, 20), -1)
            cv2.addWeighted(np_banner, 0.75, annotated, 0.25, 0, annotated)

            cv2.putText(annotated, "✔ TRAFFIC FLOW: JUNCTION NORMAL / CLEAR", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(annotated, f"Vehicles: {vehicle_count}", (width - 180, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)

        if not saved_alert_sample and (is_congested or frame_idx > 20):
            sample_out = "task11_congestion_alert.jpg"
            cv2.imwrite(sample_out, annotated)
            print(f"[OK] Saved congestion alert preview to: {sample_out}")
            saved_alert_sample = True

        if frame_idx % 15 == 0 or frame_idx == 1:
            status_text = "CONGESTED" if is_congested else "CLEAR"
            print(f"  Frame {frame_idx:03d}: Vehicles = {vehicle_count:<2} | Status = {status_text}")

        try:
            cv2.imshow("VisionPort Task 11 - Congestion Alert", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            pass

        if frame_idx >= 50:
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print("-" * 60)
    print(f"[SUCCESS] Task 11 complete: Congestion detection and alert monitoring completed.")

if __name__ == "__main__":
    detect_congestion()
