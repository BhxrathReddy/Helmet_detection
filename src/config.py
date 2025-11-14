"""
Configuration management for helmet detection system.
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import yaml


@dataclass
class DetectionConfig:
    """Configuration for helmet detection."""
    video_path: Path
    model_path: Path
    output_folder: Path
    conf_threshold: float = 0.3
    show_display: bool = False
    tesseract_cmd: Optional[str] = None
    overlap_threshold: float = 0.3
    min_plate_length: int = 3
    max_plate_length: int = 15
    
    @classmethod
    def from_dict(cls, config_dict: dict, project_root: Path) -> "DetectionConfig":
        """
        Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            project_root: Project root directory
            
        Returns:
            DetectionConfig instance
        """
        # Resolve relative paths
        video_path = Path(config_dict.get("video_path", "data/videos/hd.mp4"))
        if not video_path.is_absolute():
            video_path = project_root / video_path
        
        model_path = Path(config_dict.get("model_path", "models/best.pt"))
        if not model_path.is_absolute():
            model_path = project_root / model_path
        
        output_folder = Path(config_dict.get("output_folder", "output/violations"))
        if not output_folder.is_absolute():
            output_folder = project_root / output_folder
        
        return cls(
            video_path=video_path,
            model_path=model_path,
            output_folder=output_folder,
            conf_threshold=float(config_dict.get("conf_threshold", 0.3)),
            show_display=bool(config_dict.get("show_display", False)),
            tesseract_cmd=config_dict.get("tesseract_cmd"),
            overlap_threshold=float(config_dict.get("overlap_threshold", 0.3)),
            min_plate_length=int(config_dict.get("min_plate_length", 3)),
            max_plate_length=int(config_dict.get("max_plate_length", 15)),
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: Path, project_root: Path) -> "DetectionConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML config file
            project_root: Project root directory
            
        Returns:
            DetectionConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict, project_root)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "video_path": str(self.video_path),
            "model_path": str(self.model_path),
            "output_folder": str(self.output_folder),
            "conf_threshold": self.conf_threshold,
            "show_display": self.show_display,
            "tesseract_cmd": self.tesseract_cmd,
            "overlap_threshold": self.overlap_threshold,
            "min_plate_length": self.min_plate_length,
            "max_plate_length": self.max_plate_length,
        }

