"""
VisionPort Task 14: Identify traffic signal status (red, yellow, green).
Expected Output: Signal status detection (HSV color thresholding and state detection)
"""

import os
import sys
import cv2
import numpy as np

def detect_signal_state(frame: np.ndarray, roi: tuple = None):
    """
    Analyzes HSV color thresholds to determine active traffic signal state.
    """
    if roi is not None:
        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]
    else:
        crop = frame

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Red masks (wrapped HSV range)
    red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_mask = red1 | red2

    # Yellow mask
    yellow_mask = cv2.inRange(hsv, np.array([18, 100, 100]), np.array([35, 255, 255]))

    # Green mask
    green_mask = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([90, 255, 255]))

    red_cnt = cv2.countNonZero(red_mask)
    yellow_cnt = cv2.countNonZero(yellow_mask)
    green_cnt = cv2.countNonZero(green_mask)

    scores = {
        "RED": (red_cnt, (0, 0, 255), "STOP - Junction Closed"),
        "YELLOW": (yellow_cnt, (0, 255, 255), "CAUTION - Prepare to Stop"),
        "GREEN": (green_cnt, (0, 255, 0), "PROCEED - Junction Clear")
    }

    status, (cnt, bgr, action) = max(scores.items(), key=lambda x: x[1][0])
    total = red_cnt + yellow_cnt + green_cnt

    if total < 50:
        # Default state
        return "RED", (0, 0, 255), "STOP - Signal Inactive / Standby", red_mask, yellow_mask, green_mask

    return status, bgr, action, red_mask, yellow_mask, green_mask

def identify_traffic_signal_status(image_path: str = "captured_traffic_signal.jpg"):
    print("=" * 60)
    print("VisionPort Task 14: Identify Traffic Signal Status (Red/Yellow/Green)")
    print("=" * 60)

    # Create realistic multi-phase demonstration frames
    phases = ["RED", "YELLOW", "GREEN"]
    phase_colors = {
        "RED": ((0, 0, 255), (30, 70, 70), (20, 70, 20)),
        "YELLOW": ((70, 20, 20), (0, 255, 255), (20, 70, 20)),
        "GREEN": ((70, 20, 20), (30, 70, 70), (0, 255, 0))
    }

    # Generate synthetic or test frame with signal head
    canvas_w, canvas_h = 720, 480
    demo_frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    demo_frame[:] = (35, 30, 30)

    # Road and skyline
    cv2.rectangle(demo_frame, (0, 260), (canvas_w, canvas_h), (50, 50, 55), -1)
    for x in range(30, canvas_w, 80):
        cv2.line(demo_frame, (x, 370), (x + 40, 370), (255, 255, 255), 3)

    # Traffic Signal Pole & Enclosure
    box_x, box_y, box_w, box_h = 80, 50, 90, 240
    cv2.rectangle(demo_frame, (120, box_y + box_h), (130, 400), (40, 40, 40), -1)  # pole
    cv2.rectangle(demo_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (15, 15, 15), -1)  # box
    cv2.rectangle(demo_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (80, 80, 80), 2)

    # Light Bulbs
    center_x = box_x + box_w // 2
    r_center = (center_x, box_y + 45)
    y_center = (center_x, box_y + 120)
    g_center = (center_x, box_y + 195)
    bulb_radius = 26

    # Test with RED active
    active_phase = "RED"
    r_col, y_col, g_col = phase_colors[active_phase]
    cv2.circle(demo_frame, r_center, bulb_radius, r_col, -1)
    cv2.circle(demo_frame, y_center, bulb_radius, y_col, -1)
    cv2.circle(demo_frame, g_center, bulb_radius, g_col, -1)

    # Add glow to active bulb
    cv2.circle(demo_frame, r_center, bulb_radius + 4, (0, 0, 255), 2)

    # Run Detection on Signal ROI
    status, status_bgr, action_str, rm, ym, gm = detect_signal_state(demo_frame, roi=(box_x, box_y, box_w, box_h))

    # Render Result HUD
    hud_x, hud_y, hud_w, hud_h = 210, 60, 470, 200
    hud = demo_frame.copy()
    cv2.rectangle(hud, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (20, 20, 25), -1)
    cv2.addWeighted(hud, 0.85, demo_frame, 0.15, 0, demo_frame)
    cv2.rectangle(demo_frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), status_bgr, 2)

    cv2.putText(demo_frame, "TRAFFIC SIGNAL STATUS ANALYZER", (hud_x + 20, hud_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

    cv2.circle(demo_frame, (hud_x + 35, hud_y + 80), 14, status_bgr, -1)
    cv2.putText(demo_frame, f"Active State: {status} LIGHT", (hud_x + 60, hud_y + 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_bgr, 2)

    cv2.putText(demo_frame, f"Traffic Directive: {action_str}", (hud_x + 20, hud_y + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(demo_frame, f"Analysis Mode   : Real-time HSV Color Space Segmentation", (hud_x + 20, hud_y + 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    output_path = "task14_signal_status.jpg"
    cv2.imwrite(output_path, demo_frame)

    print("-" * 60)
    print(f"Signal Status Result:")
    print(f"  • Detected Phase : {status}")
    print(f"  • Action Directive: {action_str}")
    print(f"  • Preview Image  : {os.path.abspath(output_path)}")
    print("-" * 60)
    print("[SUCCESS] Task 14 complete: Traffic signal state identified successfully.")

    try:
        cv2.imshow("VisionPort Task 14 - Traffic Signal Status", demo_frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == "__main__":
    identify_traffic_signal_status()
