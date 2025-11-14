"""
Violation tracking and reporting module.
"""
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging


class ViolationTracker:
    """Track and export violations with metadata."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize violation tracker.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger("helmet_detection")
        self.violations: List[Dict] = []
    
    def add_violation(
        self,
        plate_text: str,
        frame_index: int,
        timestamp: Optional[float] = None,
        fps: Optional[float] = None,
        image_path: Optional[str] = None,
        confidence: Optional[float] = None
    ):
        """
        Add a violation to tracking.
        
        Args:
            plate_text: License plate text
            frame_index: Frame number where violation occurred
            timestamp: Video timestamp in seconds (optional)
            fps: Video FPS for timestamp calculation (optional)
            image_path: Path to saved violation image (optional)
            confidence: Detection confidence (optional)
        """
        violation = {
            "plate_text": plate_text,
            "frame_index": frame_index,
            "timestamp_seconds": timestamp or (frame_index / fps if fps else None),
            "timestamp_formatted": self._format_timestamp(timestamp, frame_index, fps),
            "image_path": image_path,
            "confidence": confidence,
            "detected_at": datetime.now().isoformat()
        }
        
        self.violations.append(violation)
        self.logger.debug(f"Tracked violation: {plate_text} at frame {frame_index}")
    
    def _format_timestamp(self, timestamp: Optional[float], frame_index: int, fps: Optional[float]) -> str:
        """Format timestamp as HH:MM:SS.mmm"""
        if timestamp is None and fps:
            timestamp = frame_index / fps
        elif timestamp is None:
            return "N/A"
        
        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = int(timestamp % 60)
        milliseconds = int((timestamp % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def get_statistics(self) -> Dict:
        """
        Get violation statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.violations:
            return {
                "total_violations": 0,
                "unique_plates": 0,
                "plates": []
            }
        
        unique_plates = set(v["plate_text"] for v in self.violations)
        plate_counts = {}
        for v in self.violations:
            plate = v["plate_text"]
            plate_counts[plate] = plate_counts.get(plate, 0) + 1
        
        return {
            "total_violations": len(self.violations),
            "unique_plates": len(unique_plates),
            "plates": [
                {
                    "plate_text": plate,
                    "count": count,
                    "percentage": round(count / len(self.violations) * 100, 2)
                }
                for plate, count in sorted(plate_counts.items(), key=lambda x: x[1], reverse=True)
            ],
            "first_violation": self.violations[0]["timestamp_formatted"] if self.violations else None,
            "last_violation": self.violations[-1]["timestamp_formatted"] if self.violations else None
        }
    
    def export_json(self, output_path: Path, include_statistics: bool = True):
        """
        Export violations to JSON file.
        
        Args:
            output_path: Path to output JSON file
            include_statistics: Include statistics in export
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "violations": self.violations
        }
        
        if include_statistics:
            export_data["statistics"] = self.get_statistics()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported {len(self.violations)} violations to {output_path}")
    
    def export_csv(self, output_path: Path):
        """
        Export violations to CSV file.
        
        Args:
            output_path: Path to output CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.violations:
            self.logger.warning("No violations to export")
            return
        
        fieldnames = [
            "plate_text",
            "frame_index",
            "timestamp_seconds",
            "timestamp_formatted",
            "image_path",
            "confidence",
            "detected_at"
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.violations)
        
        self.logger.info(f"Exported {len(self.violations)} violations to {output_path}")
    
    def export_summary_report(self, output_path: Path):
        """
        Export a human-readable summary report.
        
        Args:
            output_path: Path to output text file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = self.get_statistics()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("HELMET VIOLATION DETECTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Violations Detected: {stats['total_violations']}\n")
            f.write(f"Unique License Plates: {stats['unique_plates']}\n")
            
            if stats['first_violation']:
                f.write(f"First Violation: {stats['first_violation']}\n")
            if stats['last_violation']:
                f.write(f"Last Violation: {stats['last_violation']}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("VIOLATIONS BY LICENSE PLATE\n")
            f.write("=" * 60 + "\n\n")
            
            if stats['plates']:
                f.write(f"{'Plate':<20} {'Count':<10} {'Percentage':<10}\n")
                f.write("-" * 60 + "\n")
                for plate_info in stats['plates']:
                    f.write(f"{plate_info['plate_text']:<20} "
                           f"{plate_info['count']:<10} "
                           f"{plate_info['percentage']:<10}%\n")
            else:
                f.write("No violations detected.\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("DETAILED VIOLATIONS\n")
            f.write("=" * 60 + "\n\n")
            
            for i, violation in enumerate(self.violations, 1):
                f.write(f"Violation #{i}\n")
                f.write(f"  Plate: {violation['plate_text']}\n")
                f.write(f"  Frame: {violation['frame_index']}\n")
                f.write(f"  Timestamp: {violation['timestamp_formatted']}\n")
                if violation.get('image_path'):
                    f.write(f"  Image: {violation['image_path']}\n")
                if violation.get('confidence'):
                    f.write(f"  Confidence: {violation['confidence']:.2f}\n")
                f.write(f"  Detected: {violation['detected_at']}\n")
                f.write("\n")
        
        self.logger.info(f"Exported summary report to {output_path}")

