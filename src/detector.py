"""
Helmet detection and violation processing module.
"""
import os
import logging
from typing import List, Optional, Tuple, Set
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

from utils import (
    preprocess_plate_image,
    extract_plate_text,
    calculate_iou,
    validate_bbox,
    get_project_root
)
from config import DetectionConfig


class HelmetDetector:
    """Helmet violation detector using YOLO."""
    
    def __init__(self, config: DetectionConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize detector.
        
        Args:
            config: Detection configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger("helmet_detection")
        self.model: Optional[YOLO] = None
        self.seen_vehicles: Set[str] = set()
        
        # Class mapping: 0=with helmet, 1=without helmet, 2=rider, 3=number plate
        self.class_map = {
            0: 'with helmet',
            1: 'without helmet',
            2: 'rider',
            3: 'number plate'
        }
    
    def load_model(self) -> bool:
        """
        Load YOLO model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.config.model_path.exists():
                self.logger.error(f"Model file not found: {self.config.model_path}")
                return False
            
            self.logger.info(f"Loading model from {self.config.model_path}")
            self.model = YOLO(str(self.config.model_path))
            self.logger.info("Model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def separate_detections(self, boxes) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Separate detections by class.
        
        Args:
            boxes: YOLO detection boxes
            
        Returns:
            Tuple of (without_helmet_boxes, rider_boxes, plate_boxes)
        """
        wh_boxes = []  # without helmet
        rider_boxes = []
        plate_boxes = []
        
        if boxes is None or len(boxes) == 0:
            return wh_boxes, rider_boxes, plate_boxes
        
        for i in range(len(boxes.cls)):
            cls_id = int(boxes.cls[i])
            coords = boxes.xyxy[i].cpu().numpy().astype(int)
            
            if cls_id == 1:  # without helmet
                wh_boxes.append(coords)
            elif cls_id == 2:  # rider
                rider_boxes.append(coords)
            elif cls_id == 3:  # number plate
                plate_boxes.append(coords)
        
        return wh_boxes, rider_boxes, plate_boxes
    
    def find_matching_rider(
        self,
        wh_box: np.ndarray,
        rider_boxes: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Find the rider box that best matches a "without helmet" detection.
        
        Args:
            wh_box: "Without helmet" bounding box
            rider_boxes: List of rider bounding boxes
            
        Returns:
            Best matching rider box or None
        """
        x1, y1, x2, y2 = wh_box
        center_wh = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        best_rider = None
        min_distance = float('inf')
        
        for rider_box in rider_boxes:
            rx1, ry1, rx2, ry2 = rider_box
            center_r = ((rx1 + rx2) // 2, (ry1 + ry2) // 2)
            
            # Calculate distance
            distance = (center_r[0] - center_wh[0]) ** 2 + (center_r[1] - center_wh[1]) ** 2
            
            # Calculate IoU
            iou = calculate_iou(wh_box, rider_box)
            
            # Check if overlap is significant
            if iou >= self.config.overlap_threshold and distance < min_distance:
                min_distance = distance
                best_rider = rider_box
        
        return best_rider
    
    def find_nearest_plate(
        self,
        wh_box: np.ndarray,
        plate_boxes: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Find the nearest number plate to a "without helmet" detection.
        
        Args:
            wh_box: "Without helmet" bounding box
            plate_boxes: List of plate bounding boxes
            
        Returns:
            Nearest plate box or None
        """
        x1, y1, x2, y2 = wh_box
        center_wh = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        best_plate = None
        min_distance = float('inf')
        
        for plate_box in plate_boxes:
            px1, py1, px2, py2 = plate_box
            center_p = ((px1 + px2) // 2, (py1 + py2) // 2)
            distance = (center_p[0] - center_wh[0]) ** 2 + (center_p[1] - center_wh[1]) ** 2
            
            if distance < min_distance:
                min_distance = distance
                best_plate = plate_box
        
        return best_plate
    
    def process_violation(
        self,
        frame: np.ndarray,
        wh_box: np.ndarray,
        rider_box: np.ndarray,
        plate_box: np.ndarray,
        frame_index: int
    ) -> Optional[str]:
        """
        Process a helmet violation: extract plate text and save rider image.
        
        Args:
            frame: Current video frame
            wh_box: "Without helmet" bounding box
            rider_box: Rider bounding box
            plate_box: Plate bounding box
            frame_index: Current frame index
            
        Returns:
            Plate text if violation was saved, None otherwise
        """
        # Validate bounding boxes
        frame_shape = frame.shape
        if not validate_bbox(plate_box, frame_shape) or not validate_bbox(rider_box, frame_shape):
            self.logger.warning(f"Invalid bounding boxes at frame {frame_index}")
            return None
        
        # Extract and preprocess plate
        px1, py1, px2, py2 = plate_box.astype(int)
        plate_crop = frame[py1:py2, px1:px2]
        
        if plate_crop.size == 0:
            self.logger.warning(f"Empty plate crop at frame {frame_index}")
            return None
        
        # Preprocess and extract text
        processed_plate = preprocess_plate_image(plate_crop)
        plate_text = extract_plate_text(processed_plate, self.logger)
        
        if not plate_text:
            return None
        
        # Validate plate text length
        if not (self.config.min_plate_length <= len(plate_text) <= self.config.max_plate_length):
            self.logger.debug(f"Plate text '{plate_text}' length out of range")
            return None
        
        # Check if we've seen this vehicle
        if plate_text in self.seen_vehicles:
            self.logger.debug(f"Vehicle {plate_text} already processed")
            return None
        
        # Save violation
        try:
            self.seen_vehicles.add(plate_text)
            rx1, ry1, rx2, ry2 = rider_box.astype(int)
            rider_crop = frame[ry1:ry2, rx1:rx2]
            
            if rider_crop.size == 0:
                self.logger.warning(f"Empty rider crop at frame {frame_index}")
                return None
            
            os.makedirs(self.config.output_folder, exist_ok=True)
            filename = f"{plate_text}_frame{frame_index}.jpg"
            filepath = os.path.join(self.config.output_folder, filename)
            cv2.imwrite(filepath, rider_crop)
            
            self.logger.info(f"Saved violation: {plate_text} at frame {frame_index}")
            return plate_text
        except Exception as e:
            self.logger.error(f"Failed to save violation: {str(e)}")
            return None
    
    def process_frame(
        self,
        frame: np.ndarray,
        result: Results,
        frame_index: int
    ) -> int:
        """
        Process a single frame for violations.
        
        Args:
            frame: Current video frame
            result: YOLO detection results
            frame_index: Current frame index
            
        Returns:
            Number of violations found in this frame
        """
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return 0
        
        # Separate detections
        wh_boxes, rider_boxes, plate_boxes = self.separate_detections(boxes)
        
        if not wh_boxes:
            return 0
        
        violations_count = 0
        
        # Process each "without helmet" detection
        for wh_box in wh_boxes:
            # Find matching rider
            rider_box = self.find_matching_rider(wh_box, rider_boxes)
            if rider_box is None:
                continue
            
            # Find nearest plate
            plate_box = self.find_nearest_plate(wh_box, plate_boxes)
            if plate_box is None:
                continue
            
            # Process violation
            plate_text = self.process_violation(frame, wh_box, rider_box, plate_box, frame_index)
            if plate_text:
                violations_count += 1
        
        return violations_count

