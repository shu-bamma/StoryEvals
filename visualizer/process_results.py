"""
Pipeline Results Data Processor
Processes character identification JSON results for visualization
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CropData:
    """Individual crop detection data"""
    crop_id: str
    bbox_norm: List[float]  # [x, y, w, h] normalized coordinates
    crop_url: str
    detector: str
    face_conf: float  # YOLO confidence
    pred_char_id: Optional[str]
    confidence: Optional[float]  # LLM confidence
    reason: Optional[str]
    quality: Optional[Dict[str, Any]]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CropData':
        return cls(
            crop_id=data["crop_id"],
            bbox_norm=data["bbox_norm"],
            crop_url=data["crop_url"],
            detector=data["detector"],
            face_conf=data["face_conf"],
            pred_char_id=data.get("pred_char_id"),
            confidence=data.get("confidence"),
            reason=data.get("reason"),
            quality=data.get("quality")
        )


@dataclass
class KeyframeData:
    """Data for a single keyframe"""
    timestamp_ms: int
    shot_id: str
    crops: List[CropData]
    
    def has_detections(self) -> bool:
        return len(self.crops) > 0
    
    def get_identified_characters(self) -> List[str]:
        """Get list of identified characters (non-Unknown)"""
        return [crop.pred_char_id for crop in self.crops 
                if crop.pred_char_id and crop.pred_char_id != "Unknown"]


@dataclass
class CharacterData:
    """Character reference data"""
    char_id: str
    name: str
    ref_image: str
    traits: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterData':
        return cls(
            char_id=data["char_id"],
            name=data["name"],
            ref_image=data["ref_image"],
            traits=data["traits"]
        )


@dataclass
class ProjectData:
    """Complete project visualization data"""
    project_id: str
    video_url: str
    character_vault: List[CharacterData]
    keyframes: List[KeyframeData]
    metadata: Dict[str, Any]
    
    def get_timeline_duration_ms(self) -> int:
        """Get total timeline duration in milliseconds"""
        if not self.keyframes:
            return 0
        return max(kf.timestamp_ms for kf in self.keyframes)
    
    def get_character_appearance_stats(self) -> Dict[str, int]:
        """Get how many times each character appears"""
        stats = {}
        for keyframe in self.keyframes:
            for char_id in keyframe.get_identified_characters():
                stats[char_id] = stats.get(char_id, 0) + 1
        return stats


class PipelineResultsProcessor:
    """Main processor for pipeline results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_project_results(self, json_path: str) -> ProjectData:
        """Load and process a single project results file"""
        self.logger.info(f"Loading project results from: {json_path}")
        
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"Results file not found: {json_path}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self._process_project_data(data, json_file.stem)
    
    def _process_project_data(self, data: Dict[str, Any], project_id: str) -> ProjectData:
        """Process raw JSON data into structured ProjectData"""
        self.logger.info(f"Processing project data for: {project_id}")
        
        # Extract character vault
        character_vault = []
        for char_data in data.get("character_vault", []):
            character_vault.append(CharacterData.from_dict(char_data))
        
        # Extract and organize keyframes
        keyframes = []
        for shot in data.get("shots", []):
            shot_id = shot.get("shot_id", "unknown")
            keyframes_ms = shot.get("keyframes_ms", [])
            crops_data = shot.get("crops", {})
            
            # Process each keyframe in this shot
            for timestamp_ms in keyframes_ms:
                timestamp_str = str(timestamp_ms)
                crops = []
                
                # Get crops for this timestamp
                if timestamp_str in crops_data:
                    for crop_data in crops_data[timestamp_str]:
                        crops.append(CropData.from_dict(crop_data))
                
                keyframes.append(KeyframeData(
                    timestamp_ms=timestamp_ms,
                    shot_id=shot_id,
                    crops=crops
                ))
        
        # Sort keyframes by timestamp
        keyframes.sort(key=lambda x: x.timestamp_ms)
        
        # Extract metadata
        metadata = {
            "video_url": data.get("video_url", ""),
            "total_keyframes": len(keyframes),
            "total_shots": len(data.get("shots", [])),
            "total_characters": len(character_vault),
            "identification_metadata": data.get("identification_metadata", {}),
            "critique_agent_result": data.get("critique_agent_result", {})
        }
        
        project_data = ProjectData(
            project_id=project_id,
            video_url=data.get("video_url", ""),
            character_vault=character_vault,
            keyframes=keyframes,
            metadata=metadata
        )
        
        self.logger.info(f"✅ Processed {len(keyframes)} keyframes, {len(character_vault)} characters")
        return project_data
    
    def validate_data_integrity(self, project_data: ProjectData) -> bool:
        """Validate that the project data is complete and valid"""
        issues = []
        
        # Check essential fields
        if not project_data.video_url:
            issues.append("Missing video_url")
        
        if not project_data.keyframes:
            issues.append("No keyframes found")
        
        if not project_data.character_vault:
            issues.append("No characters in vault")
        
        # Check keyframe data integrity
        keyframes_with_detections = sum(1 for kf in project_data.keyframes if kf.has_detections())
        if keyframes_with_detections == 0:
            issues.append("No keyframes have detections")
        
        # Check for valid bounding boxes
        invalid_bboxes = 0
        for keyframe in project_data.keyframes:
            for crop in keyframe.crops:
                bbox = crop.bbox_norm
                if len(bbox) != 4 or not all(0 <= coord <= 1 for coord in bbox):
                    invalid_bboxes += 1
        
        if invalid_bboxes > 0:
            issues.append(f"{invalid_bboxes} crops have invalid bounding boxes")
        
        if issues:
            self.logger.warning(f"Data integrity issues: {', '.join(issues)}")
            return False
        
        self.logger.info("✅ Data integrity validation passed")
        return True
    
    def get_processing_stats(self, project_data: ProjectData) -> Dict[str, Any]:
        """Generate processing statistics for the project"""
        keyframes_with_detections = [kf for kf in project_data.keyframes if kf.has_detections()]
        total_detections = sum(len(kf.crops) for kf in project_data.keyframes)
        
        # Character identification stats
        identified_crops = sum(1 for kf in project_data.keyframes 
                             for crop in kf.crops 
                             if crop.pred_char_id and crop.pred_char_id != "Unknown")
        
        # Confidence score stats
        confidence_scores = [crop.confidence for kf in project_data.keyframes 
                           for crop in kf.crops 
                           if crop.confidence is not None]
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # YOLO confidence stats
        yolo_scores = [crop.face_conf for kf in project_data.keyframes for crop in kf.crops]
        avg_yolo_confidence = sum(yolo_scores) / len(yolo_scores) if yolo_scores else 0
        
        return {
            "total_keyframes": len(project_data.keyframes),
            "keyframes_with_detections": len(keyframes_with_detections),
            "total_detections": total_detections,
            "identified_characters": identified_crops,
            "identification_rate": identified_crops / total_detections if total_detections > 0 else 0,
            "avg_llm_confidence": round(avg_confidence, 3),
            "avg_yolo_confidence": round(avg_yolo_confidence, 3),
            "character_appearances": project_data.get_character_appearance_stats(),
            "timeline_duration_ms": project_data.get_timeline_duration_ms()
        }


def main():
    """Test the processor with a sample file"""
    processor = PipelineResultsProcessor()
    
    # Test with the sample file
    sample_file = "data/logs/test_run_20/3-character_identification/PROJadRsvgvJffe8YQEf.json"
    
    try:
        project_data = processor.load_project_results(sample_file)
        
        # Validate data
        is_valid = processor.validate_data_integrity(project_data)
        print(f"Data validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        
        # Get stats
        stats = processor.get_processing_stats(project_data)
        print("\n📊 Processing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n🎬 Sample keyframes:")
        for i, keyframe in enumerate(project_data.keyframes[:5]):  # Show first 5
            print(f"  Frame {keyframe.timestamp_ms}ms ({keyframe.shot_id}): {len(keyframe.crops)} detections")
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")


if __name__ == "__main__":
    main()
