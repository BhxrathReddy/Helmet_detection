# Helmet Violation Detection System

## Features
- Detects motorbike riders without helmets using YOLOv8
- Tracks vehicles with Deep SORT to avoid duplicate captures
- Detects and OCRs license plates using YOLO + Tesseract
- Saves violation frames only once per unique vehicle

## Requirements
Install dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Place trained models in `models/`
2. Put your input video as `input_video.mp4` in the root directory
3. Run:
```
python src/detect_and_track.py
```

Violation images and number plates will be saved in `violations/`
