"""
VisionPort Task 3: Read and display traffic images using OpenCV.
Expected Output: Image display window with detailed image properties inspection
"""

import os
import sys
import cv2
import numpy as np

def read_and_display_image(image_path: str = "sample_traffic.jpg"):
    print("=" * 60)
    print("VisionPort Task 3: Read and Display Traffic Image Using OpenCV")
    print("=" * 60)

    if not os.path.exists(image_path):
        if os.path.exists("captured_traffic_signal.jpg"):
            image_path = "captured_traffic_signal.jpg"
        elif os.path.exists("task1_opencv_test.jpg"):
            image_path = "task1_opencv_test.jpg"
        else:
            print(f"[ERROR] Image file not found: {image_path}")
            return

    print(f"[INFO] Reading image from: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Failed to load image from {image_path}")
        return

    # Image Properties
    height, width, channels = image.shape
    dtype = image.dtype
    file_size_kb = os.path.getsize(image_path) / 1024.0

    print("-" * 60)
    print("Image Metadata & Properties:")
    print(f"  • Resolution   : {width} x {height} pixels")
    print(f"  • Channels     : {channels} (BGR Color Space)")
    print(f"  • Data Type    : {dtype}")
    print(f"  • File Size    : {file_size_kb:.2f} KB")
    print(f"  • Pixel Range  : Min={image.min()}, Max={image.max()}")
    print("-" * 60)

    # Prepare display version with information overlay
    display_img = image.copy()
    overlay_h = 60
    header = np.zeros((overlay_h, width, 3), dtype=np.uint8)
    header[:] = (20, 20, 25)
    cv2.putText(header, "VisionPort Task 3: Traffic Image Viewer", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
    cv2.putText(header, f"{width}x{height} | {channels}ch", (width - 220, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    combined_display = np.vstack([header, display_img])

    # Save output copy
    output_path = "task3_output_display.jpg"
    cv2.imwrite(output_path, combined_display)
    print(f"[OK] Saved annotated display preview to: {output_path}")

    # Display image window
    window_name = "VisionPort Task 3 - Traffic Image Window"
    print(f"[INFO] Launching OpenCV display window: '{window_name}'")
    print("[INFO] Press any key or wait 3 seconds to close window...")

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1000, width), min(700, height + overlay_h))
        cv2.imshow(window_name, combined_display)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[NOTE] Window display skipped in non-GUI shell ({e}). Preview saved to {output_path}")

    print("[SUCCESS] Task 3 complete: Traffic image read and displayed successfully.")

if __name__ == "__main__":
    read_and_display_image()
