"""
Video Shot Processing Pipeline

Processes enriched character data to extract video shots with character crops using YOLO detection.
Takes enriched characters JSON as input and outputs video shots with crop data.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from config import Config
from video_shot_processor import VideoShotProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoShotProcessingPipeline:
    """Main pipeline for video shot processing and character crop extraction"""

    def __init__(self) -> None:
        self.config = Config.get_video_shot_processing_config()
        self.processor = VideoShotProcessor()

    def _validate_input_data(self, data: list[dict[str, Any]]) -> bool:
        """Validate input data structure"""

        for i, project in enumerate(data):
            required_fields = ["character_vault", "video_url", "project_id"]
            for field in required_fields:
                if field not in project:
                    logger.error(f"Project {i} missing required field: {field}")
                    return False

            if not isinstance(project["character_vault"], list):
                logger.error(f"Project {i} character_vault must be a list")
                return False

            # Validate character vault entries
            for j, char in enumerate(project["character_vault"]):
                if not isinstance(char, dict):
                    logger.error(f"Project {i}, Character {j} must be a dictionary")
                    return False

                required_char_fields = ["char_id", "name", "ref_image", "traits"]
                for field in required_char_fields:
                    if field not in char:
                        logger.error(
                            f"Project {i}, Character {j} missing required field: {field}"
                        )
                        return False

        logger.info("Input data validation passed")
        return True

    def _process_single_project(
        self, project: dict[str, Any], create_debug_video: bool = False
    ) -> dict[str, Any]:
        """Process a single project's video"""
        project_id = project["project_id"]
        video_url = project["video_url"]
        character_vault = project["character_vault"]

        logger.info(f"Processing project {project_id} with video: {video_url}")

        try:
            # Process the video using VideoShotProcessor
            result = self.processor.process_video(
                video_url, project_id, character_vault, create_debug_video
            )

            # Add additional metadata from input
            result["job_id"] = project.get("job_id")

            logger.info(f"Successfully processed project {project_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to process project {project_id}: {e}")
            # Return error result
            return {
                "project_id": project_id,
                "video_url": video_url,
                "character_vault": character_vault,
                "error": str(e),
                "number_of_shots": 0,
                "shots": [],
            }

    def process(
        self, input_data: list[dict[str, Any]], create_debug_video: bool = False
    ) -> list[dict[str, Any]]:
        """Main processing pipeline"""
        start_time = time.time()
        logger.info("Starting video shot processing pipeline")

        try:
            # Validate input data
            if not self._validate_input_data(input_data):
                raise ValueError("Input data validation failed")

            # Process each project
            results = []
            for project in input_data:
                result = self._process_single_project(project, create_debug_video)
                results.append(result)

            total_time = time.time() - start_time
            successful_projects = len([r for r in results if "error" not in r])
            logger.info(
                f"Pipeline completed: {successful_projects}/{len(results)} projects successful in {total_time:.1f}s"
            )

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def save_results(
        self, results: list[dict[str, Any]], output_path: str | None = None
    ) -> str:
        """Save results to JSON file"""
        if output_path is None:
            output_path = self.config["output_path"]

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert Shot objects to dict for JSON serialization
        serializable_results = []
        for result in results:
            if "shots" in result and result["shots"]:
                # Use VideoShotProcessor's save_results method for each project
                _ = self.processor.save_results(result, None)  # This returns the path

                # Instead, let's convert manually
                serializable_result = result.copy()
                serializable_result["shots"] = []

                for shot in result["shots"]:
                    shot_dict = {
                        "shot_id": shot.shot_id,
                        "keyframes_ms": shot.keyframes_ms,
                        "crops": {},
                    }

                    # Convert crops dict
                    for timestamp, crops_list in shot.crops.items():
                        shot_dict["crops"][str(timestamp)] = []
                        for crop in crops_list:
                            crop_dict = {
                                "crop_id": crop.crop_id,
                                "bbox_norm": crop.bbox_norm,
                                "crop_url": crop.crop_url,
                                "detector": crop.detector,
                                "face_conf": crop.face_conf,
                                "detection_confidence": crop.face_conf,  # Add new field
                                "quality": {
                                    "blur": crop.quality.blur if crop.quality else None,
                                    "pose": crop.quality.pose if crop.quality else None,
                                }
                                if crop.quality
                                else None,
                            }
                            shot_dict["crops"][str(timestamp)].append(crop_dict)

                    serializable_result["shots"].append(shot_dict)

                serializable_results.append(serializable_result)
            else:
                serializable_results.append(result)

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Video shot processing results saved to {output_file}")
        return str(output_file)
