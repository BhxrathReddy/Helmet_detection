"""
Helmet violation detection script with video processing.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
import cv2
from tqdm import tqdm

from utils import (
    setup_logging,
    get_project_root,
    find_tesseract_executable,
    validate_file_path
)
from config import DetectionConfig
from detector import HelmetDetector


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Detect helmet violations in video using YOLO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--video",
        type=str,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Path to YOLO model file (.pt)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for violation images"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold for detections"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show video display during processing"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (optional)"
    )
    
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export violations to CSV file"
    )
    
    parser.add_argument(
        "--export-json",
        type=str,
        help="Export violations to JSON file"
    )
    
    parser.add_argument(
        "--export-report",
        type=str,
        help="Export summary report to text file"
    )
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace, project_root: Path) -> DetectionConfig:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        project_root: Project root directory
        
    Returns:
        DetectionConfig instance
    """
    config_dict = {}
    
    # Load from YAML if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
        
        if config_path.exists():
            config = DetectionConfig.from_yaml(config_path, project_root)
            # Override with command line arguments
            if args.video:
                video_path = Path(args.video)
                if not video_path.is_absolute():
                    video_path = project_root / video_path
                config.video_path = video_path
            if args.model:
                model_path = Path(args.model)
                if not model_path.is_absolute():
                    model_path = project_root / model_path
                config.model_path = model_path
            if args.output:
                output_path = Path(args.output)
                if not output_path.is_absolute():
                    output_path = project_root / output_path
                config.output_folder = output_path
            if args.conf:
                config.conf_threshold = args.conf
            if args.show:
                config.show_display = True
            return config
        else:
            print(f"Warning: Config file not found: {config_path}", file=sys.stderr)
    
    # Create from arguments or defaults
    video_path = Path(args.video) if args.video else project_root / "data" / "videos" / "hd.mp4"
    if not video_path.is_absolute():
        video_path = project_root / video_path
    
    model_path = Path(args.model) if args.model else project_root / "models" / "best.pt"
    if not model_path.is_absolute():
        model_path = project_root / model_path
    
    output_folder = Path(args.output) if args.output else project_root / "output" / "violations"
    if not output_folder.is_absolute():
        output_folder = project_root / output_folder
    
    return DetectionConfig(
        video_path=video_path,
        model_path=model_path,
        output_folder=output_folder,
        conf_threshold=args.conf,
        show_display=args.show,
    )


def validate_config(config: DetectionConfig, logger) -> bool:
    """
    Validate configuration and required files.
    
    Args:
        config: Detection configuration
        logger: Logger instance
        
    Returns:
        True if valid, False otherwise
    """
    # Validate video file
    if not validate_file_path(config.video_path, "file"):
        logger.error(f"Video file not found: {config.video_path}")
        return False
    
    # Validate model file
    if not validate_file_path(config.model_path, "file"):
        logger.error(f"Model file not found: {config.model_path}")
        return False
    
    # Create output directory
    try:
        config.output_folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory: {str(e)}")
        return False
    
    # Setup Tesseract
    if config.tesseract_cmd is None:
        tesseract_path = find_tesseract_executable()
        if tesseract_path:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"Using Tesseract at: {tesseract_path}")
        else:
            logger.warning("Tesseract not found in PATH. OCR may fail.")
    else:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
        logger.info(f"Using Tesseract at: {config.tesseract_cmd}")
    
    return True


def get_video_info(video_path: Path) -> Tuple[int, int, float]:
    """
    Get video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (total_frames, fps, duration_seconds)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, 0.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0.0
    
    cap.release()
    return total_frames, fps, duration


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Get project root
    project_root = get_project_root()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Create configuration
    try:
        config = create_config_from_args(args, project_root)
    except Exception as e:
        logger.error(f"Failed to create configuration: {str(e)}")
        sys.exit(1)
    
    # Validate configuration
    if not validate_config(config, logger):
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Initialize detector
    detector = HelmetDetector(config, logger)
    
    # Load model
    if not detector.load_model():
        logger.error("Failed to load model")
        sys.exit(1)
    
    # Get video info
    total_frames, fps, duration = get_video_info(config.video_path)
    logger.info(f"Video: {config.video_path}")
    logger.info(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
    
    # Open video
    cap = cv2.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {config.video_path}")
        sys.exit(1)
    
    # Process video
    logger.info("Starting video processing...")
    frame_index = 0
    total_violations = 0
    
    try:
        # Create progress bar
        pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
        
        # Run inference
        results = detector.model.predict(
            source=str(config.video_path),
            stream=True,
            conf=config.conf_threshold,
            show=config.show_display,
            verbose=False
        )
        
        for result in results:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            violations = detector.process_frame(frame, result, frame_index)
            total_violations += violations
            
            # Update timestamps for newly added violations
            if fps > 0 and violations > 0:
                # Update only the last N violations (where N = violations found in this frame)
                recent_violations = detector.tracker.violations[-violations:]
                for violation in recent_violations:
                    if violation.get("timestamp_seconds") is None:
                        violation["timestamp_seconds"] = violation["frame_index"] / fps
                        violation["timestamp_formatted"] = detector.tracker._format_timestamp(
                            violation["timestamp_seconds"], violation["frame_index"], fps
                        )
            
            frame_index += 1
            pbar.update(1)
            
            # Update progress bar description
            pbar.set_postfix({
                'violations': len(detector.seen_vehicles),
                'frame': frame_index
            })
        
        pbar.close()
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
    finally:
        cap.release()
    
    # Export violations if requested
    if args.export_csv:
        csv_path = Path(args.export_csv)
        if not csv_path.is_absolute():
            csv_path = get_project_root() / csv_path
        detector.tracker.export_csv(csv_path)
    
    if args.export_json:
        json_path = Path(args.export_json)
        if not json_path.is_absolute():
            json_path = get_project_root() / json_path
        detector.tracker.export_json(json_path)
    
    if args.export_report:
        report_path = Path(args.export_report)
        if not report_path.is_absolute():
            report_path = get_project_root() / report_path
        detector.tracker.export_summary_report(report_path)
    
    # Get statistics
    stats = detector.tracker.get_statistics()
    
    # Summary
    logger.info("=" * 50)
    logger.info("Processing complete!")
    logger.info(f"Total frames processed: {frame_index}")
    logger.info(f"Unique violations detected: {len(detector.seen_vehicles)}")
    logger.info(f"Total violation records: {stats['total_violations']}")
    logger.info(f"Violation images saved to: {config.output_folder}")
    
    if stats['plates']:
        logger.info("\nTop violating plates:")
        for plate_info in stats['plates'][:5]:  # Top 5
            logger.info(f"  {plate_info['plate_text']}: {plate_info['count']} violations ({plate_info['percentage']}%)")
    
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
