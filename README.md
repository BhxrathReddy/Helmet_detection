# Helmet Violation Detection System

A comprehensive computer vision system for detecting motorcycle riders without helmets using YOLO, with automatic license plate recognition and violation tracking.

## Features

- **Helmet Detection**: Detects motorbike riders without helmets using YOLOv8/YOLO11
- **License Plate Recognition**: Automatic OCR of license plates using Tesseract
- **Duplicate Prevention**: Tracks vehicles using license plate text to avoid duplicate captures
- **Modular Architecture**: Clean, maintainable codebase with separated concerns
- **CLI Interface**: Command-line arguments for flexible configuration
- **Progress Tracking**: Real-time progress bar during video processing
- **Comprehensive Logging**: Detailed logging with configurable levels
- **Error Handling**: Robust error handling and validation
- **Cross-Platform**: Works on Windows, Linux, and macOS

## Project Structure

```
Helmet_detection/
├── src/                      # Source code
│   ├── __init__.py
│   ├── predict.py           # Main inference script
│   ├── train.py             # Training script
│   ├── batch_process.py     # Batch processing script
│   ├── detector.py          # Detection logic module
│   ├── config.py            # Configuration management
│   ├── utils.py             # Utility functions
│   └── violation_tracker.py # Violation tracking and export
├── models/                  # Model files (.pt)
│   ├── yolov8m.pt          # Pre-trained base models
│   ├── yolov8n.pt
│   └── yolo11n.pt
├── data/                    # Data files
│   └── videos/              # Input videos
├── config/                  # Configuration files
│   ├── helmet.yaml          # Dataset config for training
│   ├── inference_config.yaml # Inference configuration template
│   └── classes.txt          # Class names
├── output/                  # Output directory
│   └── violations/          # Saved violation images
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR**: 
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Helmet_detection
```

2. Create virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a custom model on your dataset:

```bash
python src/train.py --model yolov8m.pt --data config/helmet.yaml --epochs 100
```

**Training Options:**
- `--model`: Base model (yolov8n.pt, yolov8m.pt, yolov8l.pt, etc.)
- `--data`: Path to dataset YAML config
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 8)
- `--imgsz`: Image size (default: 640)
- `--device`: Device (0 for GPU, 'cpu' for CPU)
- `--project`: Project directory (default: models/)
- `--name`: Experiment name (default: helmet_detection)

**Example:**
```bash
python src/train.py --model models/yolov8m.pt --epochs 150 --batch 16 --device 0
```

Trained model will be saved in `models/helmet_detection/weights/best.pt`

### Inference

#### Basic Usage

Process a video with default settings:
```bash
python src/predict.py
```

This uses:
- Video: `data/videos/hd.mp4`
- Model: `models/best.pt`
- Output: `output/violations/`

#### Command Line Arguments

```bash
python src/predict.py \
    --video path/to/video.mp4 \
    --model models/best.pt \
    --output output/violations \
    --conf 0.3 \
    --show
```

**Options:**
- `--video`: Path to input video file
- `--model`: Path to YOLO model file (.pt)
- `--output`: Output directory for violation images
- `--conf`: Confidence threshold (default: 0.3)
- `--show`: Show video display during processing
- `--config`: Path to YAML configuration file
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-file`: Path to log file (optional)
- `--export-csv`: Export violations to CSV file
- `--export-json`: Export violations to JSON file
- `--export-report`: Export summary report to text file

#### Using Configuration File

Create a configuration file (see `config/inference_config.yaml` for template):

```yaml
video_path: "data/videos/my_video.mp4"
model_path: "models/best.pt"
output_folder: "output/violations"
conf_threshold: 0.3
show_display: false
tesseract_cmd: null  # Auto-detect
overlap_threshold: 0.3
min_plate_length: 3
max_plate_length: 15
```

Then run:
```bash
python src/predict.py --config config/inference_config.yaml
```

## Configuration

### Dataset Configuration (`config/helmet.yaml`)

Update the dataset path in `config/helmet.yaml`:

```yaml
path: /path/to/your/dataset
train: train/images
val: val/images
nc: 4
names: ["with helmet", "without helmet", "rider", "number plate"]
```

### Inference Configuration

See `config/inference_config.yaml` for all available inference parameters.

## Output

### Violation Images

Violation images are saved in the output directory with the format:
```
{license_plate}_frame{frame_number}.jpg
```

Example: `ABC123_frame1523.jpg`

### Export Violations

Export violations with metadata to CSV, JSON, or text report:

```bash
# Export to CSV
python src/predict.py --video data/videos/hd.mp4 --export-csv violations.csv

# Export to JSON
python src/predict.py --video data/videos/hd.mp4 --export-json violations.json

# Export summary report
python src/predict.py --video data/videos/hd.mp4 --export-report report.txt

# Export all formats
python src/predict.py --video data/videos/hd.mp4 \
    --export-csv violations.csv \
    --export-json violations.json \
    --export-report report.txt
```

**Export formats include:**
- License plate text
- Frame number
- Timestamp (HH:MM:SS.mmm)
- Image path
- Detection confidence
- Detection date/time

### Batch Processing

Process multiple videos at once:

```bash
python src/batch_process.py \
    --input-dir data/videos \
    --model models/best.pt \
    --output-dir output/violations \
    --export-csv batch_violations.csv \
    --export-json batch_violations.json
```

**Batch Processing Options:**
- `--input-dir`: Directory containing input videos
- `--model`: Path to YOLO model file
- `--output-dir`: Output directory for violation images
- `--conf`: Confidence threshold (default: 0.3)
- `--export-csv`: Export all violations to CSV
- `--export-json`: Export all violations to JSON

## Logging

Logs are displayed in the console by default. To save logs to a file:

```bash
python src/predict.py --log-file logs/processing.log --log-level DEBUG
```

## Troubleshooting

### Tesseract Not Found

If Tesseract is not automatically detected, specify the path:
- In config file: Set `tesseract_cmd` in `inference_config.yaml`
- In code: The system will try common installation paths automatically

### Model Not Found

Ensure your trained model exists at the specified path. Default location: `models/best.pt`

### Video Processing Issues

- Check video file format (supports common formats: .mp4, .avi, .mov)
- Ensure video file is not corrupted
- Check available disk space for output

## Development

### Code Structure

- **`src/detector.py`**: Core detection logic and violation processing
- **`src/config.py`**: Configuration management and validation
- **`src/utils.py`**: Utility functions (OCR, image processing, validation)
- **`src/predict.py`**: Main inference script with CLI
- **`src/train.py`**: Training script

### Adding Features

The modular architecture makes it easy to extend:
1. Add new detection logic in `detector.py`
2. Add utility functions in `utils.py`
3. Extend configuration in `config.py`

## Requirements

- Python 3.8+
- ultralytics>=8.0.20
- opencv-python>=4.8.0
- pytesseract>=0.3.10
- numpy>=1.24.0
- pyyaml>=6.0
- tqdm>=4.66.0



- YOLO models by Ultralytics
- Tesseract OCR by Google
