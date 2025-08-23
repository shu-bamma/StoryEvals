import json
import logging
import time
from pathlib import Path
from typing import Any

from image_processor import ImageProcessor

from batch_processor import BatchProcessor
from character_identifier import CharacterIdentifier
from config import Config
from critique_agent import CritiqueAgent
from models import (
    Character,
    CharacterIdentificationResult,
    CharacterTrait,
    CharacterVaultEntry,
    CritiqueAgentResult,
    Crop,
    CropQuality,
    Shot,
    VideoOutput,
)

logger = logging.getLogger(__name__)


class CharacterIdentificationPipeline:
    """Main pipeline for character identification in video shots"""

    def __init__(self) -> None:
        self.config = Config.get_character_identification_config()
        self.batch_config = Config.get_batch_processing_config()
        self.image_config = Config.get_image_processing_config()

        self.identifier = CharacterIdentifier()
        self.batch_processor = BatchProcessor()
        self.image_processor = ImageProcessor()

        # Initialize critique agent for double-checking
        self.critique_agent = CritiqueAgent()
        self.enable_critique_agent = getattr(Config, "ENABLE_CRITIQUE_AGENT", True)

    def _validate_input_data(self, data: dict[str, Any]) -> bool:
        """Validate input data structure"""
        required_fields = ["character_vault"]

        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return False

        if not isinstance(data["character_vault"], list):
            logger.error("character_vault must be a list")
            return False

        # Validate character vault entries
        for i, char in enumerate(data["character_vault"]):
            if not isinstance(char, dict):
                logger.error(f"Character {i} must be a dictionary")
                return False

            required_char_fields = ["char_id", "name", "ref_image", "traits"]
            for field in required_char_fields:
                if field not in char:
                    logger.error(f"Character {i} missing required field: {field}")
                    return False

            if not isinstance(char["traits"], dict):
                logger.error(f"Character {i} traits must be a dictionary")
                return False

        # Check if shots field exists (optional for input validation)
        if "shots" in data:
            if not isinstance(data["shots"], list):
                logger.error("shots must be a list")
                return False

        logger.info("Input data validation passed")
        return True

    def _parse_character_vault(
        self, character_vault_data: list[dict[str, Any]]
    ) -> list[CharacterVaultEntry]:
        """Parse character vault data into structured objects"""
        characters = []

        for char_data in character_vault_data:
            traits_data = char_data["traits"]

            traits = CharacterTrait(
                core=traits_data.get("core", []),
                supportive=traits_data.get("supportive", []),
                volatile=traits_data.get("volatile", []),
                age_band=traits_data.get("age_band", ""),
                skin_tone=traits_data.get("skin_tone", ""),
                type=traits_data.get("type", ""),
                notes=traits_data.get("notes", []),
            )

            character = CharacterVaultEntry(
                char_id=char_data["char_id"],
                name=char_data["name"],
                ref_image=char_data["ref_image"],
                traits=traits,
            )

            characters.append(character)

        logger.info(f"Parsed {len(characters)} characters from vault")
        return characters

    def _parse_shot_data(self, shot_data: dict[str, Any]) -> Shot:
        """Parse shot data into structured Shot object"""
        crops_dict = {}

        for timestamp_str, crops_list in shot_data["crops"].items():
            timestamp = int(timestamp_str)
            crops = []

            for crop_data in crops_list:
                quality_data = crop_data.get("quality", {})
                quality = CropQuality(
                    blur=quality_data.get("blur"), pose=quality_data.get("pose")
                )

                crop = Crop(
                    crop_id=crop_data["crop_id"],
                    bbox_norm=crop_data["bbox_norm"],
                    crop_url=crop_data["crop_url"],
                    detector=crop_data["detector"],
                    face_conf=crop_data["face_conf"],
                    quality=quality,
                )

                crops.append(crop)

            crops_dict[timestamp] = crops

        shot = Shot(
            keyframes_ms=shot_data["keyframes_ms"],
            crops=crops_dict,
            shot_id=shot_data["shot_id"],
        )

        return shot

    def _create_sample_shots(
        self, character_vault: list[CharacterVaultEntry]
    ) -> list[Shot]:
        """Create sample shots data for testing (since input doesn't have shots)"""
        logger.info("Creating sample shots data for demonstration")

        # Create a sample shot with crops
        sample_crops = {
            0: [
                Crop(
                    crop_id="c_shot0001_t0000_0",
                    bbox_norm=[0.31, 0.18, 0.14, 0.22],
                    crop_url="https://example.com/shot0001_t0000_face0.jpg",
                    detector="stylized_face_v2",
                    face_conf=0.98,
                    quality=CropQuality(blur=142.3, pose={"yaw": 6.1, "pitch": -2.3}),
                )
            ],
            420: [
                Crop(
                    crop_id="c_shot0001_t0420_0",
                    bbox_norm=[0.31, 0.18, 0.14, 0.22],
                    crop_url="https://example.com/shot0001_t0420_face0.jpg",
                    detector="stylized_face_v2",
                    face_conf=0.98,
                    quality=CropQuality(blur=142.3, pose={"yaw": 6.1, "pitch": -2.3}),
                )
            ],
            840: [
                Crop(
                    crop_id="c_shot0001_t0840_0",
                    bbox_norm=[0.31, 0.18, 0.14, 0.22],
                    crop_url="https://example.com/shot0001_t0840_face0.jpg",
                    detector="stylized_face_v2",
                    face_conf=0.98,
                    quality=CropQuality(blur=142.3, pose={"yaw": 6.1, "pitch": -2.3}),
                ),
                Crop(
                    crop_id="c_shot0001_t0840_1",
                    bbox_norm=[0.52, 0.20, 0.13, 0.21],
                    crop_url="https://example.com/shot0001_t0840_face1.jpg",
                    detector="stylized_face_v2",
                    face_conf=0.95,
                    quality=CropQuality(blur=120.9, pose={"yaw": -18.4, "pitch": 1.8}),
                ),
            ],
        }

        sample_shot = Shot(
            keyframes_ms=[0, 420, 840], crops=sample_crops, shot_id="shot_0001"
        )

        return [sample_shot]

    def _process_character_identification(
        self, character_vault: list[CharacterVaultEntry], shots: list[Shot]
    ) -> list[Shot]:
        """Process character identification for all shots"""
        logger.info("Starting character identification process")

        # Get processing statistics
        stats = self.batch_processor.get_processing_stats(shots)
        logger.info(f"Processing stats: {stats}")

        # Process shots in parallel
        processed_shots = self.batch_processor.process_shots_parallel(
            shots, character_vault, self.identifier.process_batch
        )

        logger.info(
            f"Character identification completed for {len(processed_shots)} shots"
        )
        return processed_shots

    def _process_images(self, shots: list[Shot]) -> dict[str, str]:
        """Process all crop images with ID overlays"""
        logger.info("Starting image processing")

        # Extract all crops from all shots
        all_crops = []
        for shot in shots:
            for crops_list in shot.crops.values():
                all_crops.extend(crops_list)

        logger.info(f"Processing {len(all_crops)} crop images")

        # Process images in batches
        image_results = self.image_processor.process_crops_batch(all_crops)

        # Create mapping of crop_id to output path
        crop_to_path = {}
        for crop, output_path in image_results:
            if output_path:
                crop_to_path[crop.crop_id] = output_path

        successful_images = len([path for _, path in image_results if path])
        logger.info(
            f"Image processing completed: {successful_images}/{len(all_crops)} successful"
        )

        return crop_to_path

    def _create_output_structure(
        self,
        character_vault: list[CharacterVaultEntry],
        shots: list[Shot],
        original_data: dict[str, Any],
        critique_result: CritiqueAgentResult | None = None,
    ) -> CharacterIdentificationResult:
        """Create the complete output structure"""
        output = CharacterIdentificationResult(
            character_vault=character_vault,
            shots=shots,
            project_id=original_data.get("project_id"),
            job_id=original_data.get("job_id"),
            video_url=original_data.get("video_url"),
            identification_metadata={
                "total_characters": len(character_vault),
                "total_shots": len(shots),
                "total_crops": sum(
                    len(crops_list)
                    for shot in shots
                    for crops_list in shot.crops.values()
                ),
                "model_used": self.config["model"],
                "batch_size": self.batch_config["batch_size"],
                "max_parallel_shots": self.batch_config["max_parallel_shots"],
                "critique_agent_enabled": self.enable_critique_agent,
                "critique_agent_verification": critique_result is not None,
                "timestamp": time.time(),
            },
        )

        return output

    def _run_critique_agent_verification(
        self,
        character_vault: list[CharacterVaultEntry],
        shots: list[Shot],
        video_url: str | None = None,
    ) -> CritiqueAgentResult | None:
        """
        Run critique agent verification as a double-check for character identification

        Args:
            character_vault: List of characters from the vault
            shots: Processed shots with character identifications
            video_url: URL of the video being analyzed

        Returns:
            CritiqueAgentResult if verification is enabled and successful, None otherwise
        """
        if not self.enable_critique_agent:
            logger.info("Critique agent verification is disabled")
            return None

        if not video_url:
            logger.warning("No video URL provided for critique agent verification")
            return None

        try:
            logger.info("Starting critique agent verification as double-check")

            # Convert character vault to Character objects for critique agent
            characters = []
            for char_entry in character_vault:
                # Create a description from traits
                description_parts = []
                if char_entry.traits.core:
                    description_parts.append(
                        f"Core traits: {', '.join(char_entry.traits.core)}"
                    )
                if char_entry.traits.supportive:
                    description_parts.append(
                        f"Supportive traits: {', '.join(char_entry.traits.supportive)}"
                    )
                if char_entry.traits.age_band:
                    description_parts.append(f"Age: {char_entry.traits.age_band}")
                if char_entry.traits.skin_tone:
                    description_parts.append(
                        f"Skin tone: {char_entry.traits.skin_tone}"
                    )
                if char_entry.traits.type:
                    description_parts.append(f"Type: {char_entry.traits.type}")

                description = (
                    ". ".join(description_parts)
                    if description_parts
                    else "Character from video"
                )

                character = Character(
                    char_id=char_entry.char_id,
                    name=char_entry.name,
                    image=char_entry.ref_image,
                    description=description,
                )
                characters.append(character)

            # Create VideoOutput object for critique agent
            video_output = VideoOutput(
                project_id="",  # Will be filled by the calling method
                clip_url=video_url,
                prompt="",  # Not needed for critique agent
                enhanced_prompt="",  # Not needed for critique agent
                reference_image="",  # Not needed for critique agent
                thumbnail_image="",  # Not needed for critique agent
                characters=characters,
                debug={},
            )

            # Run critique agent verification
            critique_result = self.critique_agent.evaluate_video_output(video_output)

            logger.info("Critique agent verification completed successfully")
            return critique_result

        except Exception as e:
            logger.error(f"Critique agent verification failed: {e}")
            return None

    def process(
        self, input_data: dict[str, Any], process_images: bool = True
    ) -> CharacterIdentificationResult:
        """Main processing pipeline"""
        start_time = time.time()
        logger.info("Starting character identification pipeline")

        try:
            # Validate input data
            if not self._validate_input_data(input_data):
                raise ValueError("Input data validation failed")

            # Parse character vault
            character_vault = self._parse_character_vault(input_data["character_vault"])

            # Parse shots or create sample data
            if "shots" in input_data and input_data["shots"]:
                shots = [
                    self._parse_shot_data(shot_data)
                    for shot_data in input_data["shots"]
                ]
                logger.info(f"Using {len(shots)} shots from input data")
            else:
                shots = self._create_sample_shots(character_vault)
                logger.info("Created sample shots data for demonstration")

            # Process character identification
            processed_shots = self._process_character_identification(
                character_vault, shots
            )

            # Process images if requested
            image_paths = {}
            if process_images:
                image_paths = self._process_images(processed_shots)
                logger.info(
                    f"Processed {len(image_paths)} images with crop ID overlays"
                )

            # Run critique agent verification
            critique_result = None
            if self.enable_critique_agent:
                video_url = input_data.get("video_url")
                if not video_url:
                    logger.warning(
                        "No video URL provided for critique agent verification. Skipping."
                    )
                else:
                    critique_result = self._run_critique_agent_verification(
                        character_vault, processed_shots, video_url
                    )

            # Create output structure
            result = self._create_output_structure(
                character_vault, processed_shots, input_data, critique_result
            )

            # Store critique agent result if available
            if critique_result:
                result.critique_agent_result = critique_result
                logger.info("Critique agent results integrated into output")

            total_time = time.time() - start_time
            logger.info(f"Pipeline completed successfully in {total_time:.1f}s")

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def save_results(
        self, result: CharacterIdentificationResult, output_path: str | None = None
    ) -> str:
        """Save results to JSON file"""
        if output_path is None:
            output_path = Config.CHARACTER_IDENTIFICATION_OUTPUT_PATH

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary for JSON serialization
        result_dict = {
            "character_vault": [
                {
                    "char_id": char.char_id,
                    "name": char.name,
                    "ref_image": char.ref_image,
                    "traits": {
                        "core": char.traits.core,
                        "supportive": char.traits.supportive,
                        "volatile": char.traits.volatile,
                        "age_band": char.traits.age_band,
                        "skin_tone": char.traits.skin_tone,
                        "type": char.traits.type,
                        "notes": char.traits.notes,
                    },
                }
                for char in result.character_vault
            ],
            "shots": [
                {
                    "keyframes_ms": shot.keyframes_ms,
                    "crops": {
                        str(timestamp): [
                            {
                                "crop_id": crop.crop_id,
                                "bbox_norm": crop.bbox_norm,
                                "crop_url": crop.crop_url,
                                "detector": crop.detector,
                                "face_conf": crop.face_conf,
                                "quality": {
                                    "blur": crop.quality.blur if crop.quality else None,
                                    "pose": crop.quality.pose if crop.quality else None,
                                }
                                if crop.quality
                                else None,
                                "pred_char_id": crop.pred_char_id,
                                "confidence": crop.confidence,
                                "reason": crop.reason,
                            }
                            for crop in crops_list
                        ]
                        for timestamp, crops_list in shot.crops.items()
                    },
                    "shot_id": shot.shot_id,
                }
                for shot in result.shots
            ],
        }

        # Add additional metadata
        if result.project_id:
            result_dict["project_id"] = result.project_id
        if result.job_id:
            result_dict["job_id"] = result.job_id
        if result.video_url:
            result_dict["video_url"] = result.video_url
        if result.identification_metadata:
            result_dict["identification_metadata"] = result.identification_metadata

        # Add critique agent results if available
        if hasattr(result, "critique_agent_result") and result.critique_agent_result:
            critique_data = result.critique_agent_result
            result_dict["critique_agent_result"] = {
                "project_id": critique_data.project_id,
                "overall_accuracy": critique_data.overall_accuracy,
                "evaluation_notes": critique_data.evaluation_notes,
                "character_verifications": [
                    {
                        "character_name": v.character_name,
                        "character_image": v.character_image,
                        "video_clip_url": v.video_clip_url,
                        "is_present": v.is_present,
                        "confidence_score": v.confidence_score,
                        "reasoning": v.reasoning,
                        "timestamp_analysis": v.timestamp_analysis,
                    }
                    for v in critique_data.character_verifications
                ],
                "llm_metadata": critique_data.llm_metadata,
            }

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_file}")
        return str(output_file)
