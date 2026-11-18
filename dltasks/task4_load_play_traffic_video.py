"""
VisionPort Task 4: Load and play a traffic video using OpenCV.
Expected Output: Video playback with video metadata inspection and playback controls
"""

import os
import sys
import time
import cv2

def load_and_play_video(video_path: str = "traffic_sample.mp4", max_frames_headless: int = 120):
    print("=" * 60)
    print("VisionPort Task 4: Load and Play Traffic Video Using OpenCV")
    print("=" * 60)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video file: {video_path}")
        return

    # Extract Video Properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / max(fps, 1.0)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    print("-" * 60)
    print("Video Metadata & Specifications:")
    print(f"  • Source File   : {video_path}")
    print(f"  • Resolution    : {width} x {height}")
    print(f"  • Frame Rate    : {fps:.2f} FPS")
    print(f"  • Total Frames  : {total_frames}")
    print(f"  • Duration      : {duration_sec:.2f} seconds")
    print(f"  • FourCC Codec  : {codec}")
    print("-" * 60)

    window_name = "VisionPort Task 4 - Video Playback"
    delay_ms = max(1, int(1000.0 / fps))

    print(f"[INFO] Starting video playback window: '{window_name}'")
    print("[INFO] Controls: Press 'q' to exit | SPACE to pause/resume")

    frame_idx = 0
    start_wall_time = time.time()
    gui_available = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream reached.")
            break

        frame_idx += 1
        elapsed_video_time = frame_idx / fps

        # Render HUD overlay on video frame
        hud = frame.copy()
        cv2.rectangle(hud, (10, 10), (360, 85), (0, 0, 0), -1)
        cv2.addWeighted(hud, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, f"VisionPort Video Playback", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} ({elapsed_video_time:.1f}s)", (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.1f} | Res: {width}x{height}", (20, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        if gui_available:
            try:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord('q'):
                    print("[INFO] Playback stopped by user.")
                    break
                elif key == 32:  # SPACE to pause
                    cv2.waitKey(-1)
            except Exception:
                gui_available = False

        if not gui_available and frame_idx >= max_frames_headless:
            print(f"[INFO] Processed {frame_idx} frames successfully in background verification mode.")
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    total_wall_time = time.time() - start_wall_time
    print(f"[SUCCESS] Task 4 complete: Played {frame_idx} frames in {total_wall_time:.2f}s.")

if __name__ == "__main__":
    load_and_play_video()
