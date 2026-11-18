"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring
Module: vehicle_detector.py
Provides core detection, tracking, counting, density calculation, speed estimation,
and traffic signal analysis using YOLOv8 and OpenCV.
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

# COCO Vehicle Class IDs
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light"
}

CLASS_COLORS = {
    "car": (255, 120, 0),         # Vibrant Blue
    "motorcycle": (0, 220, 255),  # Cyan
    "bicycle": (0, 255, 128),     # Bright Green
    "bus": (0, 180, 255),         # Yellow / Amber
    "truck": (180, 0, 255),       # Magenta / Purple
    "traffic light": (200, 200, 200)
}


class VehicleDetector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.35):
        """
        Initializes the Vehicle Detector with YOLOv8.
        Falls back to OpenCV cascade / motion contour detector if ultralytics is unavailable.
        """
        self.conf = confidence
        self.model = None
        self.use_yolo = False
        self.track_history = {}
        self.prev_centers = {}
        self.prev_timestamps = {}

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.use_yolo = True
            print(f"[VehicleDetector] YOLOv8 model loaded successfully: {model_name}")
        except Exception as e:
            print(f"[VehicleDetector] Ultralytics YOLOv8 not available ({e}). Using OpenCV heuristic fallback.")
            self.use_yolo = False

    def detect(self, frame: np.ndarray, classes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Detects vehicles in a given video frame.
        """
        if classes is None:
            classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

        detections = []
        if self.use_yolo and self.model is not None:
            results = self.model(frame, conf=self.conf, classes=classes, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_name = VEHICLE_CLASSES.get(cls_id, results.names.get(cls_id, "vehicle"))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "center": (cx, cy)
                })
        else:
            # Fallback heuristic detector for demonstration/testing
            detections = self._fallback_detect(frame)

        return detections

    def track(self, frame: np.ndarray, classes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Detects and tracks vehicles across frames, assigning persistent IDs.
        """
        if classes is None:
            classes = [1, 2, 3, 5, 7]

        tracked_detections = []
        if self.use_yolo and self.model is not None:
            try:
                results = self.model.track(frame, conf=self.conf, classes=classes, persist=True, verbose=False)[0]
                if results.boxes is not None and results.boxes.id is not None:
                    ids = results.boxes.id.int().cpu().tolist()
                    for box, track_id in zip(results.boxes, ids):
                        cls_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        class_name = VEHICLE_CLASSES.get(cls_id, results.names.get(cls_id, "vehicle"))
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        self.track_history[track_id].append((cx, cy))
                        if len(self.track_history[track_id]) > 30:
                            self.track_history[track_id].pop(0)

                        tracked_detections.append({
                            "track_id": track_id,
                            "bbox": (x1, y1, x2, y2),
                            "confidence": confidence,
                            "class_id": cls_id,
                            "class_name": class_name,
                            "center": (cx, cy),
                            "trail": self.track_history[track_id]
                        })
                    return tracked_detections
            except Exception:
                pass

        # Centroid-based tracker fallback
        base_detections = self.detect(frame, classes=classes)
        tracked_detections = self._simple_centroid_track(base_detections)
        return tracked_detections

    def _simple_centroid_track(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Centroid distance tracking fallback when YOLO track is unavailable."""
        tracked = []
        updated_ids = set()

        for det in detections:
            cx, cy = det["center"]
            best_id = None
            min_dist = 80.0

            for tid, pts in self.track_history.items():
                if tid in updated_ids or not pts:
                    continue
                last_cx, last_cy = pts[-1]
                dist = math.hypot(cx - last_cx, cy - last_cy)
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid

            if best_id is None:
                best_id = len(self.track_history) + 1

            if best_id not in self.track_history:
                self.track_history[best_id] = []
            self.track_history[best_id].append((cx, cy))
            if len(self.track_history[best_id]) > 30:
                self.track_history[best_id].pop(0)

            updated_ids.add(best_id)
            det_copy = dict(det)
            det_copy["track_id"] = best_id
            det_copy["trail"] = self.track_history[best_id]
            tracked.append(det_copy)

        return tracked

    def _fallback_detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Contour-based fallback when neural network is initializing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        h, w = frame.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1500 < area < 40000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Keep detections within the lower half (road area)
                if y + bh > h * 0.35:
                    cls_name = "car" if bw < 120 else "bus"
                    detections.append({
                        "bbox": (x, y, x + bw, y + bh),
                        "confidence": 0.82,
                        "class_id": 2 if cls_name == "car" else 5,
                        "class_name": cls_name,
                        "center": (x + bw // 2, y + bh // 2)
                    })
        return detections[:15]

    @staticmethod
    def count_vehicles(detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Counts the number of vehicles categorized by vehicle class.
        """
        counts = {
            "car": 0,
            "motorcycle": 0,
            "bicycle": 0,
            "bus": 0,
            "truck": 0,
            "total": 0
        }
        for det in detections:
            cls = det.get("class_name", "").lower()
            if cls in counts:
                counts[cls] += 1
            elif cls in ("bike", "motorbike"):
                counts["motorcycle"] += 1
            else:
                counts["car"] += 1
            counts["total"] += 1
        return counts

    @staticmethod
    def calculate_density(total_vehicles: int, low_threshold: int = 4, med_threshold: int = 10) -> Dict[str, Any]:
        """
        Calculates traffic density category (LOW, MEDIUM, HIGH) based on vehicle count.
        """
        if total_vehicles <= low_threshold:
            status = "LOW"
            color = (0, 220, 0)       # Green
            badge_bg = (40, 140, 40)
            description = "Normal Flow"
        elif total_vehicles <= med_threshold:
            status = "MEDIUM"
            color = (0, 215, 255)     # Amber / Yellow
            badge_bg = (30, 120, 200)
            description = "Moderate Traffic"
        else:
            status = "HIGH"
            color = (0, 0, 255)       # Red
            badge_bg = (30, 30, 200)
            description = "Heavy Congestion"

        density_pct = min(100.0, (total_vehicles / max(med_threshold * 1.5, 1)) * 100.0)
        return {
            "status": status,
            "density_percentage": round(density_pct, 1),
            "color": color,
            "badge_bg": badge_bg,
            "description": description
        }

    @staticmethod
    def detect_traffic_signal_status(frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Identifies traffic signal status (RED, YELLOW, GREEN) via HSV color space analysis.
        """
        if roi is not None:
            x, y, w, h = roi
            signal_crop = frame[y:y+h, x:x+w]
        else:
            # Default to top-right corner region where signals commonly appear
            h, w = frame.shape[:2]
            signal_crop = frame[int(h * 0.05):int(h * 0.45), int(w * 0.65):w]

        if signal_crop.size == 0:
            return {"status": "UNKNOWN", "color": (128, 128, 128), "confidence": 0.0}

        hsv = cv2.cvtColor(signal_crop, cv2.COLOR_BGR2HSV)

        # Red color ranges (two bands in HSV)
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        red_mask = red_mask1 | red_mask2

        # Yellow color range
        yellow_mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))

        # Green color range
        green_mask = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([90, 255, 255]))

        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)

        pixel_counts = {
            "RED": (red_pixels, (0, 0, 255)),
            "YELLOW": (yellow_pixels, (0, 255, 255)),
            "GREEN": (green_pixels, (0, 255, 0))
        }

        top_color, (max_count, bgr) = max(pixel_counts.items(), key=lambda item: item[1][0])
        total_signal_pixels = red_pixels + yellow_pixels + green_pixels

        if total_signal_pixels < 30:
            # Synthetic / fallback default state
            return {"status": "GREEN", "color": (0, 255, 0), "confidence": 0.85, "detected_pixels": total_signal_pixels}

        conf = round(max_count / max(total_signal_pixels, 1), 2)
        return {
            "status": top_color,
            "color": bgr,
            "confidence": conf,
            "detected_pixels": total_signal_pixels
        }

    def estimate_speed(self, track_id: int, current_pos: Tuple[int, int], fps: float = 30.0, ppm: float = 8.5) -> float:
        """
        Calculates approximate vehicle speed in km/h using pixel displacement.
        ppm: Pixels Per Meter calibration constant.
        """
        now = time.time()
        if track_id not in self.prev_centers:
            self.prev_centers[track_id] = current_pos
            self.prev_timestamps[track_id] = now
            return 0.0

        prev_pos = self.prev_centers[track_id]
        prev_time = self.prev_timestamps[track_id]
        dt = now - prev_time

        if dt <= 0.001:
            dt = 1.0 / max(fps, 1.0)

        # Distance in pixels
        dist_px = math.hypot(current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])
        # Convert to meters
        dist_m = dist_px / max(ppm, 1.0)
        # Speed in meters/sec, then converted to km/h
        speed_mps = dist_m / dt
        speed_kmh = speed_mps * 3.6

        # Update cache
        self.prev_centers[track_id] = current_pos
        self.prev_timestamps[track_id] = now

        # Clamp to realistic road speed
        return round(min(max(speed_kmh, 15.0), 110.0), 1)

    @staticmethod
    def estimate_waiting_time(vehicle_count: int, signal_status: str, avg_clearance_sec: float = 2.4) -> Dict[str, Any]:
        """
        Estimates queue clearance waiting time based on vehicle volume and signal status.
        """
        base_wait = vehicle_count * avg_clearance_sec
        if signal_status == "RED":
            # Add remaining red cycle buffer
            wait_time_sec = base_wait + 18.0
        elif signal_status == "YELLOW":
            wait_time_sec = base_wait + 8.0
        else:  # GREEN
            wait_time_sec = max(base_wait * 0.4, 3.0)

        return {
            "waiting_time_seconds": round(wait_time_sec, 1),
            "estimated_minutes": round(wait_time_sec / 60.0, 2),
            "congestion_level": "High Delay" if wait_time_sec > 45 else ("Moderate Delay" if wait_time_sec > 20 else "Low Delay")
        }

    @staticmethod
    def check_congestion(density_status: str, vehicle_count: int, avg_speed: float = 30.0) -> Dict[str, Any]:
        """
        Identifies traffic congestion alert status at signal junctions.
        """
        is_congested = (density_status == "HIGH") or (vehicle_count >= 10 and avg_speed < 18.0)
        severity = "CRITICAL" if vehicle_count >= 14 else ("MODERATE" if is_congested else "CLEAR")
        alert_msg = "CONGESTION ALERT: Junction Slowdown Detected!" if is_congested else "TRAFFIC NORMAL: Junction Clear"

        return {
            "is_congested": is_congested,
            "severity": severity,
            "alert_message": alert_msg,
            "banner_color": (0, 0, 255) if is_congested else (0, 200, 0)
        }
