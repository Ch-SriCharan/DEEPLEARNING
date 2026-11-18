"""
VisionPort Task 12: Track individual vehicles across video frames.
Expected Output: Vehicle tracking IDs and trajectory motion trails
"""

import os
import sys
import cv2
from collections import defaultdict
from ultralytics import YOLO

CLASSES = [1, 2, 3, 5, 7]  # vehicles

def track_vehicles_in_video(video_path: str = "traffic_sample.mp4",
                            model_path: str = "yolov8n.pt",
                            conf_threshold: float = 0.35,
                            max_frames: int = 150):
    print("=" * 60)
    print("VisionPort Task 12: Vehicle Tracking Across Video Frames (Unique IDs)")
    print("=" * 60)

    print(f"[INFO] Loading YOLO model for tracking: {model_path}")
    model = YOLO(model_path)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: [])
    unique_ids_seen = set()

    frame_idx = 0
    saved_sample = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model.track(frame, classes=CLASSES, conf=conf_threshold, persist=True, verbose=False)[0]

        annotated = frame.copy()

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.int().cpu().tolist()
            track_ids = results.boxes.id.int().cpu().tolist()
            confs = results.boxes.conf.float().cpu().tolist()
            classes_detected = results.boxes.cls.int().cpu().tolist()

            for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, classes_detected):
                x1, y1, x2, y2 = box
                unique_ids_seen.add(track_id)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Store trajectory trail points
                track = track_history[track_id]
                track.append((cx, cy))
                if len(track) > 25:
                    track.pop(0)

                # Draw trajectory trail line
                for i in range(1, len(track)):
                    if track[i - 1] is None or track[i] is None:
                        continue
                    thickness = int(2.5 * (i / len(track))) + 1
                    cv2.line(annotated, track[i - 1], track[i], (0, 255, 255), thickness)

                # Draw vehicle box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)

                # Tracking ID badge
                id_label = f"ID #{track_id}"
                cv2.rectangle(annotated, (x1, y1 - 24), (x1 + 80, y1), (0, 200, 255), -1)
                cv2.putText(annotated, id_label, (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Top tracking HUD
        hud = annotated.copy()
        cv2.rectangle(hud, (15, 15), (340, 90), (20, 20, 25), -1)
        cv2.addWeighted(hud, 0.8, annotated, 0.2, 0, annotated)
        cv2.rectangle(annotated, (15, 15), (340, 90), (0, 200, 255), 2)

        cv2.putText(annotated, "VEHICLE TRACKING (ID)", (25, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        active_now = len(results.boxes.id) if (results.boxes and results.boxes.id is not None) else 0
        cv2.putText(annotated, f"Active Tracks Now : {active_now}", (25, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Cumulative Unique IDs: {len(unique_ids_seen)}", (25, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

        if not saved_sample and len(unique_ids_seen) >= 3:
            sample_out = "task12_vehicle_tracking.jpg"
            cv2.imwrite(sample_out, annotated)
            print(f"[OK] Saved vehicle tracking sample to: {sample_out}")
            saved_sample = True

        if frame_idx % 15 == 0 or frame_idx == 1:
            print(f"  Frame {frame_idx:03d}: Active Tracks = {active_now:<2} | Cumulative Unique IDs = {len(unique_ids_seen)}")

        try:
            cv2.imshow("VisionPort Task 12 - Vehicle Tracking", annotated)
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
    print(f"[SUCCESS] Task 12 complete: Tracked {len(unique_ids_seen)} unique vehicles across {frame_idx} frames.")

if __name__ == "__main__":
    track_vehicles_in_video()
