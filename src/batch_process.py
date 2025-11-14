"""
Batch processing script for multiple videos.
"""
import argparse
import sys
from pathlib import Path
from typing import List
import logging

from utils import setup_logging, get_project_root, validate_file_path
from config import DetectionConfig
from detector import HelmetDetector
from violation_tracker import ViolationTracker


def find_video_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    """
    Find all video files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions (default: ['.mp4', '.avi', '.mov', '.mkv'])
        
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    video_files = []
    for ext in extensions:
        video_files.extend(directory.glob(f"*{ext}"))
        video_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(video_files)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch process multiple videos for helmet violations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input videos"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Path to YOLO model file (.pt)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for violation images (default: output/violations)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold"
    )
    
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export all violations to CSV file"
    )
    
    parser.add_argument(
        "--export-json",
        type=str,
        help="Export all violations to JSON file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def process_single_video(
    video_path: Path,
    config: DetectionConfig,
    logger: logging.Logger,
    global_tracker: ViolationTracker
) -> bool:
    """
    Process a single video file.
    
    Args:
        video_path: Path to video file
        config: Detection configuration
        logger: Logger instance
        global_tracker: Global violation tracker for batch processing
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {video_path.name}")
    logger.info(f"{'='*60}")
    
    # Update config for this video
    config.video_path = video_path
    
    # Create detector
    detector = HelmetDetector(config, logger)
    
    if not detector.load_model():
        logger.error(f"Failed to load model for {video_path.name}")
        return False
    
    # Import here to avoid circular imports
    import cv2
    from tqdm import tqdm
    
    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    logger.info(f"Frames: {total_frames}, FPS: {fps:.2f}")
    
    # Process video
    try:
        results = detector.model.predict(
            source=str(video_path),
            stream=True,
            conf=config.conf_threshold,
            show=False,
            verbose=False
        )
        
        frame_index = 0
        pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", unit="frame")
        
        cap = cv2.VideoCapture(str(video_path))
        for result in results:
            ret, frame = cap.read()
            if not ret:
                break
            
            violations = detector.process_frame(frame, result, frame_index)
            
            # Update timestamps
            if fps > 0:
                for violation in detector.tracker.violations:
                    if violation.get("timestamp_seconds") is None:
                        violation["timestamp_seconds"] = violation["frame_index"] / fps
                        violation["timestamp_formatted"] = detector.tracker._format_timestamp(
                            violation["timestamp_seconds"], violation["frame_index"], fps
                        )
            
            frame_index += 1
            pbar.update(1)
            pbar.set_postfix({
                'violations': len(detector.seen_vehicles),
                'video': video_path.stem[:20]
            })
        
        pbar.close()
        cap.release()
        
        # Merge violations into global tracker
        for violation in detector.tracker.violations:
            violation["source_video"] = str(video_path.name)
            global_tracker.violations.append(violation)
        
        logger.info(f"Completed: {video_path.name} - {len(detector.seen_vehicles)} violations")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {video_path.name}: {str(e)}", exc_info=True)
        return False


def main():
    """Main execution function."""
    args = parse_arguments()
    
    project_root = get_project_root()
    logger = setup_logging(args.log_level)
    
    # Resolve input directory
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = project_root / input_dir
    
    if not validate_file_path(input_dir, "directory"):
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Find video files
    video_files = find_video_files(input_dir)
    if not video_files:
        logger.error(f"No video files found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(video_files)} video file(s)")
    
    # Create configuration
    model_path = Path(args.model) if args.model else project_root / "models" / "best.pt"
    if not model_path.is_absolute():
        model_path = project_root / model_path
    
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "output" / "violations"
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    
    config = DetectionConfig(
        video_path=video_files[0],  # Temporary, will be updated per video
        model_path=model_path,
        output_folder=output_dir,
        conf_threshold=args.conf,
        show_display=False
    )
    
    # Validate model
    if not validate_file_path(config.model_path, "file"):
        logger.error(f"Model file not found: {config.model_path}")
        sys.exit(1)
    
    # Global tracker for all videos
    global_tracker = ViolationTracker(logger)
    
    # Process each video
    successful = 0
    failed = 0
    
    for video_file in video_files:
        if process_single_video(video_file, config, logger, global_tracker):
            successful += 1
        else:
            failed += 1
    
    # Export results
    if args.export_csv:
        csv_path = Path(args.export_csv)
        if not csv_path.is_absolute():
            csv_path = project_root / csv_path
        global_tracker.export_csv(csv_path)
    
    if args.export_json:
        json_path = Path(args.export_json)
        if not json_path.is_absolute():
            json_path = project_root / json_path
        global_tracker.export_json(json_path)
    
    # Summary
    stats = global_tracker.get_statistics()
    logger.info("\n" + "="*60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Videos processed: {successful}/{len(video_files)}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total violations: {stats['total_violations']}")
    logger.info(f"Unique plates: {stats['unique_plates']}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

