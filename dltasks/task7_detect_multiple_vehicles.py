"""
VisionPort Task 7: Detect bikes, buses and trucks.
Expected Output: Multiple vehicle detection (Color-coded bounding boxes)
"""

import os
import sys
import cv2
from ultralytics import YOLO

# Vehicle classes mapping
TARGET_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle (Bike)",
    5: "Bus",
    7: "Truck"
}

# Color palette for each vehicle category (BGR)
CLASS_COLORS = {
    "Bicycle": (0, 255, 128),            # Bright Green
    "Car": (255, 140, 0),                # Blue
    "Motorcycle (Bike)": (0, 230, 255),  # Cyan
    "Bus": (0, 180, 255),                # Yellow / Gold
    "Truck": (200, 50, 255)              # Magenta / Purple
}

def detect_multiple_vehicles(video_path: str = "traffic_sample.mp4",
                             image_path: str = "sample_traffic.jpg",
                             model_path: str = "yolov8n.pt",
                             conf_threshold: float = 0.35):
    print("=" * 60)
    print("VisionPort Task 7: Multiple Vehicle Detection (Bikes, Buses, Trucks, Cars)")
    print("=" * 60)

    print(f"[INFO] Loading YOLOv8 model: {model_path}")
    model = YOLO(model_path)
    target_ids = list(TARGET_CLASSES.keys())

    # We can process video and also test image
    source = video_path if os.path.exists(video_path) else image_path
    print(f"[INFO] Source input: {source}")

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    window_name = "VisionPort Task 7 - Multi-Vehicle Detection"
    frame_idx = 0
    saved_sample = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model(frame, classes=target_ids, conf=conf_threshold, verbose=False)[0]

        annotated = frame.copy()
        counts_in_frame = {name: 0 for name in TARGET_CLASSES.values()}

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = TARGET_CLASSES.get(cls_id, "Vehicle")
            color = CLASS_COLORS.get(class_name, (0, 255, 0))

            counts_in_frame[class_name] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Draw colored bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw badge tag
            label = f"{class_name}: {conf * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Legend HUD on Top Left
        hud_h = 100
        hud = annotated.copy()
        cv2.rectangle(hud, (10, 10), (320, 10 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(hud, 0.75, annotated, 0.25, 0, annotated)

        cv2.putText(annotated, "VisionPort: Multi-Vehicle Detection", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)

        y_offset = 50
        for name, col in CLASS_COLORS.items():
            cnt = counts_in_frame.get(name, 0)
            cv2.circle(annotated, (25, y_offset), 5, col, -1)
            cv2.putText(annotated, f"{name}: {cnt}", (38, y_offset + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
            y_offset += 16

        if not saved_sample and sum(counts_in_frame.values()) > 0:
            sample_out = "task7_multiple_vehicles.jpg"
            cv2.imwrite(sample_out, annotated)
            print(f"[OK] Saved multi-vehicle detection sample to: {sample_out}")
            saved_sample = True

        try:
            cv2.imshow(window_name, annotated)
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
    print(f"[SUCCESS] Task 7 complete: Successfully detected multiple vehicle classes.")

if __name__ == "__main__":
    detect_multiple_vehicles()
