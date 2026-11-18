"""
VisionPort Task 8: Display vehicle names and confidence scores.
Expected Output: Labelled detection output with formatted labels and confidence ratings
"""

import os
import sys
import cv2
from ultralytics import YOLO

VEHICLE_LABELS = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

def display_labels_and_confidence(image_path: str = "sample_traffic.jpg",
                                  model_path: str = "yolov8n.pt",
                                  conf_threshold: float = 0.30):
    print("=" * 60)
    print("VisionPort Task 8: Display Vehicle Names & Confidence Scores")
    print("=" * 60)

    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    # Use sample_traffic.jpg or extracted frame
    if not os.path.exists(image_path):
        if os.path.exists("extracted_frames/frame_0000.jpg"):
            image_path = "extracted_frames/frame_0000.jpg"
        elif os.path.exists("traffic_sample.mp4"):
            # grab first frame
            cap = cv2.VideoCapture("traffic_sample.mp4")
            ret, frame = cap.read()
            cap.release()
            if ret:
                cv2.imwrite("sample_traffic.jpg", frame)
                image_path = "sample_traffic.jpg"

    print(f"[INFO] Reading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Could not load image.")
        return

    results = model(img, classes=list(VEHICLE_LABELS.keys()), conf=conf_threshold, verbose=False)[0]

    annotated = img.copy()
    detections_summary = []

    print("-" * 60)
    print(f"{'#':<4} | {'Vehicle Name':<16} | {'Confidence':<12} | {'Coordinates (x1,y1,x2,y2)'}")
    print("-" * 60)

    for idx, box in enumerate(results.boxes, 1):
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        class_name = VEHICLE_LABELS.get(cls_id, "Vehicle")
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        conf_str = f"{conf * 100:.1f}%"
        label_text = f"{class_name}: {conf_str}"
        detections_summary.append((idx, class_name, conf_str, (x1, y1, x2, y2)))

        print(f"{idx:<4} | {class_name:<16} | {conf_str:<12} | ({x1}, {y1}, {x2}, {y2})")

        # Color based on confidence (Green = high >= 80%, Amber = med >= 50%, Red = low)
        if conf >= 0.80:
            box_color = (0, 220, 0)
        elif conf >= 0.50:
            box_color = (0, 215, 255)
        else:
            box_color = (0, 100, 255)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Draw label pill badge
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), box_color, -1)
        cv2.putText(annotated, label_text, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    output_path = "task8_labelled_detections.jpg"
    cv2.imwrite(output_path, annotated)
    print("-" * 60)
    print(f"[SUCCESS] Task 8 complete: {len(detections_summary)} vehicles labelled.")
    print(f"[OK] Labelled visualization saved to: {output_path}")

    try:
        cv2.namedWindow("VisionPort Task 8 - Labels & Confidence", cv2.WINDOW_NORMAL)
        cv2.imshow("VisionPort Task 8 - Labels & Confidence", annotated)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == "__main__":
    display_labels_and_confidence()
