import os
from ultralytics import YOLO

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths relative to project root
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8m.pt")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "helmet.yaml")

model = YOLO(MODEL_PATH)

model.train(
    data=CONFIG_PATH,
    imgsz=640,
    batch=8,
    epochs=100,
    workers=0,
    device=0,
    project=os.path.join(PROJECT_ROOT, "models"),  # Save trained model to models/
    name="helmet_detection"  # Creates models/helmet_detection/weights/best.pt
)