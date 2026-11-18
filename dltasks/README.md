# VisionPort: Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning

VisionPort is an end-to-end computer vision and deep learning traffic monitoring system built with **Python**, **OpenCV**, **Ultralytics YOLOv8**, **PyTorch**, and **FastAPI**. It provides automated real-time vehicle detection, multi-class vehicle identification, counting, density calculation, tracking, speed estimation, junction congestion alerts, signal phase identification, and queue waiting-time estimation.

---

## 📁 Project Directory Structure

```text
Deep Learning/
├── .vscode/
│   ├── launch.json              # VS Code debug and run configurations
│   └── settings.json            # Python interpreter and workspace settings
├── extracted_frames/            # Extracted individual traffic video frames (Task 5)
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
├── README.md                    # Project documentation and task instructions
├── requirements.txt             # Python package dependencies
├── sample_traffic.jpg           # Real-world traffic reference image
├── task1_opencv_test.jpg        # Output of Task 1 environment test
├── task1_setup_and_verify.py    # Task 1: Environment & library verification
├── task2_capture_traffic_signal.py # Task 2: Camera traffic signal capture
├── task3_read_display_traffic_image.py # Task 3: Read & display traffic images
├── task4_load_play_traffic_video.py # Task 4: Video playback with OpenCV
├── task5_extract_video_frames.py   # Task 5: Frame extraction from video
├── task6_detect_cars.py         # Task 6: Car detection using YOLO
├── task7_detect_multiple_vehicles.py # Task 7: Multi-vehicle detection (bikes, buses, trucks)
├── task8_display_labels_confidence.py # Task 8: Labels and confidence scores
├── task9_count_vehicles.py      # Task 9: Vehicle counting per frame
├── task10_traffic_density.py    # Task 10: Traffic density (Low, Medium, High)
├── task11_detect_congestion.py  # Task 11: Congestion detection at junctions
├── task12_track_vehicles.py     # Task 12: Individual vehicle tracking with persistent IDs
├── task13_calculate_vehicle_speed.py # Task 13: Approximate vehicle speed estimation
├── task14_identify_traffic_signal_status.py # Task 14: Signal status detection (Red/Yellow/Green)
├── task15_estimate_waiting_time.py # Task 15: Traffic queue waiting-time report
├── traffic_sample.mp4           # Video sample for traffic analysis
├── vehicle_detector.py          # Core modular VehicleDetector class
├── visionport.html              # Modern web dashboard for live traffic inference
├── visionport.py                # FastAPI server endpoint for real-time inference
├── visionportsrcode.txt         # Text source code reference of visionport
└── yolov8n.pt                   # Pretrained YOLOv8 Nano model weights
```

---

## 🚀 Tasks and Assignments Overview

| Task No. | Task Description | Expected Output | Script File |
| :---: | :--- | :--- | :--- |
| **1** | Install Python, OpenCV and required libraries | Working Python environment | `task1_setup_and_verify.py` |
| **2** | Capture an image of a traffic signal using a camera | Traffic image capture | `task2_capture_traffic_signal.py` |
| **3** | Read and display traffic images using OpenCV | Image display window | `task3_read_display_traffic_image.py` |
| **4** | Load and play a traffic video using OpenCV | Video playback | `task4_load_play_traffic_video.py` |
| **5** | Extract individual frames from traffic videos | Frames saved as images | `task5_extract_video_frames.py` |
| **6** | Detect cars in a traffic video using YOLO | Car detection with bounding boxes | `task6_detect_cars.py` |
| **7** | Detect bikes, buses and trucks | Multiple vehicle detection | `task7_detect_multiple_vehicles.py` |
| **8** | Display vehicle names and confidence scores | Labelled detection output | `task8_display_labels_confidence.py` |
| **9** | Count the number of vehicles in a video frame | Vehicle count | `task9_count_vehicles.py` |
| **10** | Calculate traffic density based on vehicle count | Low, medium and high traffic | `task10_traffic_density.py` |
| **11** | Detect traffic congestion at signal junctions | Congestion alerts | `task11_detect_congestion.py` |
| **12** | Track individual vehicles across video frames | Vehicle tracking IDs | `task12_track_vehicles.py` |
| **13** | Calculate approximate vehicle speed | Speed estimation | `task13_calculate_vehicle_speed.py` |
| **14** | Identify traffic signal status (red, yellow, green) | Signal status detection | `task14_identify_traffic_signal_status.py` |
| **15** | Estimate traffic waiting time | Waiting-time report | `task15_estimate_waiting_time.py` |

---

## 🛠️ Installation & Setup

1. **Verify or install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Task 1 verification:**
   ```bash
   python task1_setup_and_verify.py
   ```

---

## 🏃 Running Individual Tasks

To execute any task, run the corresponding python script:

```bash
# Task 1 to 5: Core Computer Vision Foundations
python task1_setup_and_verify.py
python task2_capture_traffic_signal.py
python task3_read_display_traffic_image.py
python task4_load_play_traffic_video.py
python task5_extract_video_frames.py

# Task 6 to 10: Deep Learning Vehicle Detection & Analysis
python task6_detect_cars.py
python task7_detect_multiple_vehicles.py
python task8_display_labels_confidence.py
python task9_count_vehicles.py
python task10_traffic_density.py

# Task 11 to 15: Advanced Traffic Analytics & Tracking
python task11_detect_congestion.py
python task12_track_vehicles.py
python task13_calculate_vehicle_speed.py
python task14_identify_traffic_signal_status.py
python task15_estimate_waiting_time.py
```

---

## 🌐 VisionPort Web Portal & API

Launch the live interactive web interface and FastAPI inference service:

```bash
python visionport.py
```
Open your browser and navigate to:
- **Web Portal UI:** `http://localhost:8000/`
- **Interactive API Docs:** `http://localhost:8000/docs`

Upload any traffic snapshot (`sample_traffic.jpg` or any road image) to see real-time YOLOv8 vehicle detection streaming back.
