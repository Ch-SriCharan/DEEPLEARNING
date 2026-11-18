"""
VisionPort Task 15: Estimate traffic waiting time.
Expected Output: Waiting-time report (Terminal Summary + Saved Report & Visual Chart)
"""

import os
import sys
import time
import cv2
import numpy as np

def generate_waiting_time_report(vehicle_counts: dict = None, signal_status: str = "RED"):
    print("=" * 60)
    print("VisionPort Task 15: Traffic Waiting-Time Estimation Report")
    print("=" * 60)

    if vehicle_counts is None:
        # Realistic junction sample queue
        vehicle_counts = {
            "Car": 7,
            "Motorcycle": 4,
            "Bus": 2,
            "Truck": 1
        }

    # Clearance discharge rate factors (seconds per vehicle type)
    clearance_factors = {
        "Car": 2.2,
        "Motorcycle": 1.2,
        "Bus": 4.5,
        "Truck": 5.0
    }

    total_vehicles = sum(vehicle_counts.values())
    raw_clearance_time = sum(vehicle_counts[k] * clearance_factors.get(k, 2.0) for k in vehicle_counts)

    # Signal phase delay component
    red_phase_remaining = 24.0 if signal_status == "RED" else (6.0 if signal_status == "YELLOW" else 0.0)
    total_waiting_time_sec = raw_clearance_time + red_phase_remaining
    total_waiting_time_min = total_waiting_time_sec / 60.0

    # Congestion severity category
    if total_waiting_time_sec < 25.0:
        delay_grade = "MINIMAL DELAY (Grade A)"
        status_color = (0, 220, 0)
    elif total_waiting_time_sec < 55.0:
        delay_grade = "MODERATE QUEUE (Grade B)"
        status_color = (0, 215, 255)
    else:
        delay_grade = "HEAVY DELAY (Grade C - Congested)"
        status_color = (0, 0, 255)

    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # Generate text report
    report_text = f"""============================================================
           VISIONPORT TRAFFIC WAITING-TIME REPORT
============================================================
Generated At        : {timestamp_str}
Signal State        : {signal_status} (Buffer: +{red_phase_remaining:.1f}s)
Total Queue Volume  : {total_vehicles} vehicles
Queue Delay Grade   : {delay_grade}
------------------------------------------------------------
VEHICLE QUEUE COMPOSITION & CLEARANCE ESTIMATION:
"""
    for v_type, cnt in vehicle_counts.items():
        sub_time = cnt * clearance_factors.get(v_type, 2.0)
        report_text += f"  • {v_type:<14} : {cnt:>2} units | Sub-clearance: {sub_time:>4.1f} sec\n"

    report_text += f"""------------------------------------------------------------
ESTIMATED QUEUE CLEARANCE : {raw_clearance_time:.1f} seconds
SIGNAL CYCLE OVERHEAD     : {red_phase_remaining:.1f} seconds
TOTAL ESTIMATED WAIT TIME : {total_waiting_time_sec:.1f} seconds ({total_waiting_time_min:.2f} mins)
RECOMMENDED ACTION        : {'Extend Green Phase by +15s' if total_waiting_time_sec > 45 else 'Maintain Standard Adaptive Signal Timer'}
============================================================
"""

    print(report_text)

    # Save text report to file
    text_report_path = "waiting_time_report.txt"
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[OK] Report saved to disk: {os.path.abspath(text_report_path)}")

    # Generate visual infographic card
    canvas_w, canvas_h = 750, 480
    card = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    card[:] = (24, 22, 28)

    # Header banner
    cv2.rectangle(card, (0, 0), (canvas_w, 70), (35, 30, 45), -1)
    cv2.putText(card, "VisionPort: Traffic Waiting-Time Report", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

    # Summary Cards
    cv2.rectangle(card, (30, 95), (360, 210), (35, 35, 45), -1)
    cv2.rectangle(card, (30, 95), (360, 210), status_color, 2)
    cv2.putText(card, "ESTIMATED WAITING TIME", (45, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(card, f"{total_waiting_time_sec:.0f} SECONDS", (45, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    cv2.putText(card, f"Approx: {total_waiting_time_min:.2f} minutes delay", (45, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # Queue Metrics Card
    cv2.rectangle(card, (390, 95), (720, 210), (35, 35, 45), -1)
    cv2.rectangle(card, (390, 95), (720, 210), (80, 80, 100), 1)
    cv2.putText(card, "JUNCTION OVERVIEW", (405, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(card, f"Queue Length : {total_vehicles} vehicles", (405, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(card, f"Signal Status: {signal_status}", (405, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(card, f"Delay Grade  : {delay_grade.split('(')[0]}", (405, 202),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

    # Breakdown bars
    cv2.putText(card, "Queue Breakdown By Vehicle Classification:", (30, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    y_bar = 275
    for v_type, cnt in vehicle_counts.items():
        pct = cnt / max(total_vehicles, 1)
        bar_len = int(450 * pct)
        cv2.putText(card, f"{v_type:<10} ({cnt})", (30, y_bar + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.rectangle(card, (180, y_bar), (180 + 450, y_bar + 20), (45, 45, 55), -1)
        cv2.rectangle(card, (180, y_bar), (180 + bar_len, y_bar + 20), (0, 200, 255), -1)
        cv2.putText(card, f"{pct * 100:.0f}%", (180 + bar_len + 10, y_bar + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        y_bar += 38

    # Adaptive Recommendation
    cv2.rectangle(card, (30, 420), (720, 460), (40, 50, 40), -1)
    cv2.putText(card, "AI Recommendation: Adaptive Traffic Signal Phase Optimization Active", (45, 445),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

    img_report_path = "task15_waiting_time_report.jpg"
    cv2.imwrite(img_report_path, card)
    print(f"[OK] Visual report saved to: {os.path.abspath(img_report_path)}")
    print("[SUCCESS] Task 15 complete: Waiting-time report calculated and formatted.")

    try:
        cv2.imshow("VisionPort Task 15 - Waiting-Time Report", card)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == "__main__":
    generate_waiting_time_report()
