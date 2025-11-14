"""
Utility functions for helmet detection system.
"""
import os
import logging
import platform
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2
import pytesseract


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("helmet_detection")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def find_tesseract_executable() -> Optional[str]:
    """
    Find Tesseract OCR executable path based on platform.
    
    Returns:
        Path to tesseract executable or None if not found
    """
    system = platform.system()
    
    # Common paths
    if system == "Windows":
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    elif system == "Darwin":  # macOS
        common_paths = [
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
        ]
    else:  # Linux
        common_paths = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]
    
    # Check if tesseract is in PATH
    import shutil
    tesseract_cmd = shutil.which("tesseract")
    if tesseract_cmd:
        return tesseract_cmd
    
    # Check common paths
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def validate_file_path(file_path: Path, file_type: str = "file") -> bool:
    """
    Validate that a file or directory exists.
    
    Args:
        file_path: Path to validate
        file_type: Type of path ("file" or "directory")
        
    Returns:
        True if valid, False otherwise
    """
    if file_type == "file":
        if not file_path.exists():
            return False
        if not file_path.is_file():
            return False
    elif file_type == "directory":
        if not file_path.exists():
            return False
        if not file_path.is_dir():
            return False
    
    return True


def preprocess_plate_image(plate_crop: np.ndarray) -> np.ndarray:
    """
    Preprocess license plate image for OCR.
    
    Args:
        plate_crop: Cropped license plate image
        
    Returns:
        Preprocessed binary image
    """
    if plate_crop.size == 0:
        return plate_crop
    
    # Convert to grayscale
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop
    
    # Histogram equalization
    eq = cv2.equalizeHist(gray)
    
    # Threshold
    _, binary = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Resize for better OCR
    h, w = binary.shape
    processed = cv2.resize(binary, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    
    return processed


def extract_plate_text(plate_image: np.ndarray, logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Extract text from license plate image using OCR.
    
    Args:
        plate_image: Preprocessed license plate image
        logger: Optional logger instance
        
    Returns:
        Extracted plate text or None if extraction fails
    """
    if plate_image.size == 0:
        if logger:
            logger.warning("Empty plate image provided for OCR")
        return None
    
    try:
        config = (
            "--oem 3 --psm 7 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        raw_text = pytesseract.image_to_string(plate_image, config=config)
        plate_text = "".join(ch for ch in raw_text if ch.isalnum()).upper()
        
        if not plate_text:
            if logger:
                logger.debug("No text extracted from plate image")
            return None
        
        return plate_text
    except Exception as e:
        if logger:
            logger.error(f"OCR extraction failed: {str(e)}")
        return None


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: Bounding box [x1, y1, x2, y2]
        box2: Bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def validate_bbox(bbox: np.ndarray, frame_shape: Tuple[int, int, int]) -> bool:
    """
    Validate bounding box coordinates.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        frame_shape: Frame shape (height, width, channels)
        
    Returns:
        True if valid, False otherwise
    """
    if len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    
    if x1 >= x2 or y1 >= y2:
        return False
    
    h, w = frame_shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    
    return True

