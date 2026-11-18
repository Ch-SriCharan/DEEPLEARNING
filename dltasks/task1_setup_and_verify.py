"""
Expected Output: Working Python Environment Verification
"""

import sys
import importlib

print("=" * 60)
print("VisionPort Task 1: Environment & Library Verification")
print("=" * 60)

# Check Python version
print(f"Python Version: {sys.version}")
if sys.version_info < (3, 8):
    print("WARNING: Python 3.8 or higher is recommended.")
else:
    print("[OK] Python version is compatible.\n")

# Required libraries for the VisionPort project
required_libraries = [
    ("cv2", "OpenCV (opencv-python)"),
    ("numpy", "NumPy"),
    ("PIL", "Pillow"),
    ("ultralytics", "Ultralytics YOLOv8"),
    ("torch", "PyTorch"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn")
]

src_detected = False
all_passed = True

print("Checking installed packages:")
print("-" * 60)
for module_name, display_name in required_libraries:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "Installed")
        print(f"[PASSED] {display_name:<28} : v{version}")
    except (ImportError, ModuleNotFoundError):
        print(f"[MISSING] {display_name:<28} : NOT INSTALLED")
        all_passed = False

print("-" * 60)
if all_passed:
    print("[SUCCESS] All required libraries are installed successfully!")
    try:
        import cv2
        import numpy as np

        test_image_path = "task1_opencv_test.jpg"
        test_img = np.zeros((400, 600, 3), dtype=np.uint8)
        test_img[:] = (35, 30, 30)
        cv2.putText(test_img, "VisionPort Environment Verified!", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(test_img, "OpenCV + YOLOv8 + PyTorch Ready", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.imwrite(test_image_path, test_img)
        print(f"[OpenCV Test] Successfully created test image: {test_image_path}")
    except Exception as e:
        print(f"[ERROR] Failed to run OpenCV image test: {e}")
else:
    print("[WARNING] Please install missing packages using: pip install -r requirements.txt")
