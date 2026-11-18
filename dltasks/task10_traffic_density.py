"""
VisionPort Task 10: Calculate traffic density based on vehicle count.
Expected Output: Low, medium and high traffic density estimation
"""

import os
import sys
import cv2
from ultralytics import YOLO

CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

def evaluate_density(vehicle_count: int, low_thresh: int = 4, med_thresh: int = 9):
    if vehicle_count <= low_thresh:
        return "LOW", (0, 220, 0), "Smooth Flow / Clear Lanes"
    elif vehicle_count <= med_thresh:
        return "MEDIUM", (0, 215, 255), "Moderate Volume / Steady Pace"
    else:
        return "HIGH", (0, 0, 255), "Dense Congestion / Potential Delay"

def calculate_traffic_density(video_path: str = "traffic_sample.mp4",
                             image_path: str = "sample_traffic.jpg",
                             model_path: str = "yolov8n.pt",
                             conf_threshold: float = 0.35):
    print("=" * 60)
    print("VisionPort Task 10: Traffic Density Calculation (Low / Med / High)")
    print("=" * 60)

    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    source = video_path if os.path.exists(video_path) else image_path
    cap = cv2.VideoCapture(source)

    frame_idx = 0
    saved_sample = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model(frame, classes=CLASSES, conf=conf_threshold, verbose=False)[0]

        vehicle_count = len(results.boxes)
        density_level, level_color, description = evaluate_density(vehicle_count)

        annotated = frame.copy()

        # Draw vehicle boxes
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), level_color, 2)

        # Traffic Density HUD Card
        card_w, card_h = 360, 130
        hud = annotated.copy()
        cv2.rectangle(hud, (15, 15), (15 + card_w, 15 + card_h), (20, 20, 25), -1)
        cv2.addWeighted(hud, 0.85, annotated, 0.15, 0, annotated)

        # Outer highlight border in density color
        cv2.rectangle(annotated, (15, 15), (15 + card_w, 15 + card_h), level_color, 2)

        # Title
        cv2.putText(annotated, "TRAFFIC DENSITY MONITOR", (25, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Density Badge
        badge_text = f"DENSITY: {density_level}"
        cv2.putText(annotated, badge_text, (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, level_color, 2)

        # Vehicle Count & Description
        cv2.putText(annotated, f"Vehicles Detected: {vehicle_count}", (25, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, description, (25, 126),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)

        # Density Progress Meter Bar
        bar_x, bar_y, bar_w, bar_h = 15, 15 + card_h + 6, card_w, 12
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_pct = min(1.0, vehicle_count / 14.0)
        fill_w = int(bar_w * fill_pct)
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), level_color, -1)

        if not saved_sample:
            sample_out = "task10_traffic_density.jpg"
            cv2.imwrite(sample_out, annotated)
            print(f"[OK] Saved traffic density indicator preview to: {sample_out}")
            saved_sample = True

        if frame_idx % 15 == 0 or frame_idx == 1:
            print(f"  Frame {frame_idx:03d}: Vehicles = {vehicle_count:<2} | Density Level = {density_level:<6} ({description})")

        try:
            cv2.imshow("VisionPort Task 10 - Traffic Density", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            pass

        if frame_idx >= 150:
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print("-" * 60)
    print(f"[SUCCESS] Task 10 complete: Traffic density evaluated across {frame_idx} video frames.")

if __name__ == "__main__":
    calculate_traffic_density()
