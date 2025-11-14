# Helmet Violation Detection System

## Features
- Detects motorbike riders without helmets using YOLOv8
- Tracks vehicles using license plate OCR to avoid duplicate captures
- Detects and OCRs license plates using YOLO + Tesseract
- Saves violation frames only once per unique vehicle

## Project Structure
```
Helmet_detection/
├── src/              # Source code
│   ├── predict.py   # Main inference script
│   └── train.py     # Training script
├── models/          # Model files (.pt)
├── data/            # Data files
│   └── videos/      # Input videos
├── config/          # Configuration files
│   ├── helmet.yaml  # Dataset config for training
│   └── classes.txt  # Class names
├── output/          # Output directory
│   └── violations/  # Saved violation images
└── requirements.txt
```

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
1. Prepare your dataset and update `config/helmet.yaml` with the correct paths
2. Run training:
```bash
python src/train.py
```
Trained model will be saved in `models/helmet_detection/weights/best.pt`

### Inference
1. Place your trained model as `models/best.pt` (or update the path in `src/predict.py`)
2. Put your input video in `data/videos/` (default: `hd.mp4`)
3. Run:
```bash
python src/predict.py
```

Violation images will be saved in `output/violations/`
