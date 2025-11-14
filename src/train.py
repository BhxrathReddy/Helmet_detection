"""
Training script for helmet detection model.
"""
import os
import sys
import argparse
from pathlib import Path
import logging

from ultralytics import YOLO
from utils import setup_logging, get_project_root, validate_file_path


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train YOLO model for helmet detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Base YOLO model to use (yolov8n.pt, yolov8m.pt, etc.)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="config/helmet.yaml",
        help="Path to dataset configuration YAML file"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use (0 for GPU, cpu for CPU)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker threads for data loading"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        help="Project directory (default: models/)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="helmet_detection",
        help="Experiment name"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Get project root
    project_root = get_project_root()
    
    # Resolve paths
    model_name = args.model
    if not model_name.endswith('.pt'):
        model_name += '.pt'
    
    model_path = project_root / "models" / model_name
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.info(f"Available models in models/: {list((project_root / 'models').glob('*.pt'))}")
        sys.exit(1)
    
    config_path = Path(args.data)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    if not validate_file_path(config_path, "file"):
        logger.error(f"Dataset config file not found: {config_path}")
        sys.exit(1)
    
    # Project directory
    if args.project:
        project_dir = Path(args.project)
        if not project_dir.is_absolute():
            project_dir = project_root / project_dir
    else:
        project_dir = project_root / "models"
    
    logger.info("=" * 50)
    logger.info("Helmet Detection Model Training")
    logger.info("=" * 50)
    logger.info(f"Base model: {model_path}")
    logger.info(f"Dataset config: {config_path}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch}")
    logger.info(f"Image size: {args.imgsz}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Project: {project_dir}")
    logger.info(f"Experiment name: {args.name}")
    logger.info("=" * 50)
    
    try:
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = YOLO(str(model_path))
        
        # Train model
        logger.info("Starting training...")
        results = model.train(
            data=str(config_path),
            imgsz=args.imgsz,
            batch=args.batch,
            epochs=args.epochs,
            workers=args.workers,
            device=args.device,
            project=str(project_dir),
            name=args.name,
        )
        
        logger.info("=" * 50)
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {project_dir / args.name / 'weights' / 'best.pt'}")
        logger.info("=" * 50)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
