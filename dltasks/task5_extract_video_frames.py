"""
VisionPort Task 5: Extract individual frames from traffic videos.
Expected Output: Frames saved as images in extracted_frames directory
"""

import os
import sys
import cv2

def extract_frames(video_path: str = "traffic_sample.mp4",
                   output_dir: str = "extracted_frames",
                   frame_interval: int = 15,
                   max_extract: int = 25):
    print("=" * 60)
    print("VisionPort Task 5: Extract Video Frames as Images")
    print("=" * 60)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Destination Directory: {os.path.abspath(output_dir)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video file: {video_path}")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Processing {total_video_frames} frames ({width}x{height} @ {fps:.1f} FPS)...")
    print(f"[INFO] Extraction rate: 1 frame every {frame_interval} frames (max {max_extract} frames)")

    saved_count = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0 and saved_count < max_extract:
            frame_filename = f"frame_{saved_count:04d}.jpg"
            output_filepath = os.path.join(output_dir, frame_filename)

            # Add timestamp / frame number watermark for clarity
            watermark = frame.copy()
            cv2.rectangle(watermark, (10, height - 35), (280, height - 8), (0, 0, 0), -1)
            cv2.addWeighted(watermark, 0.6, frame, 0.4, 0, frame)
            timestamp_sec = frame_index / fps
            cv2.putText(frame, f"Frame {frame_index} ({timestamp_sec:.2f}s)", (15, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imwrite(output_filepath, frame)
            saved_count += 1
            print(f"  [SAVED] {frame_filename:<18} (Video frame #{frame_index:04d} at {timestamp_sec:.2f}s)")

        frame_index += 1
        if saved_count >= max_extract:
            break

    cap.release()
    print("-" * 60)
    print(f"[SUCCESS] Task 5 complete: Extracted {saved_count} frames into '{output_dir}/'")

if __name__ == "__main__":
    extract_frames()
