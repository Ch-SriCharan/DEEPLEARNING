"""
VisionPort Task 9: Count the number of vehicles in a video frame.
Expected Output: Vehicle count (Overall count + Per-class breakdown)
"""

import os
import sys
import cv2
from ultralytics import YOLO

CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

def count_vehicles_in_video(video_path: str = "traffic_sample.mp4",
                            model_path: str = "yolov8n.pt",
                            conf_threshold: float = 0.35,
                            max_frames: int = 150):
    print("=" * 60)
    print("VisionPort Task 9: Count Vehicles in Video Frames")
    print("=" * 60)

    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_idx = 0
    saved_sample = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model(frame, classes=list(CLASSES.keys()), conf=conf_threshold, verbose=False)[0]

        counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0, "Bicycle": 0}
        total_vehicles = 0

        annotated = frame.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            class_name = CLASSES.get(cls_id, "Vehicle")
            if class_name in counts:
                counts[class_name] += 1
            total_vehicles += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 200), 2)

        # Counter Dashboard Widget
        hud_w, hud_h = 320, 140
        hud = annotated.copy()
        cv2.rectangle(hud, (15, 15), (15 + hud_w, 15 + hud_h), (25, 25, 30), -1)
        cv2.addWeighted(hud, 0.8, annotated, 0.2, 0, annotated)

        # Border
        cv2.rectangle(annotated, (15, 15), (15 + hud_w, 15 + hud_h), (50, 180, 255), 2)

        # Header
        cv2.putText(annotated, "VEHICLE COUNTER", (28, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 255), 2)
        cv2.putText(annotated, f"Total Vehicles: {total_vehicles}", (28, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

        # Class breakdown row
        breakdown_str = f"Cars: {counts['Car']} | Buses: {counts['Bus']} | Trucks: {counts['Truck']} | Bikes: {counts['Motorcycle']}"
        cv2.putText(annotated, breakdown_str, (28, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        cv2.putText(annotated, f"Frame: #{frame_idx:03d}/{total_video_frames}", (28, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        if not saved_sample and total_vehicles > 0:
            sample_out = "task9_vehicle_count.jpg"
            cv2.imwrite(sample_out, annotated)
            print(f"[OK] Saved vehicle count HUD snapshot to: {sample_out}")
            saved_sample = True

        if frame_idx % 15 == 0 or frame_idx == 1:
            print(f"  Frame {frame_idx:03d}: Total Vehicles = {total_vehicles} "
                  f"(Cars: {counts['Car']}, Buses: {counts['Bus']}, Trucks: {counts['Truck']}, Bikes: {counts['Motorcycle']})")

        try:
            cv2.imshow("VisionPort Task 9 - Vehicle Counting", annotated)
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
    print(f"[SUCCESS] Task 9 complete: Vehicle counting executed across {frame_idx} video frames.")

if __name__ == "__main__":
    count_vehicles_in_video()
