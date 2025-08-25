import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from config import Config
from models import (
    CharacterIdentificationBatch,
    CharacterVaultEntry,
    Crop,
    Shot,
)

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batching of crops and parallel processing of shots"""

    def __init__(self) -> None:
        self.config = Config.get_batch_processing_config()
        self.batch_size = self.config["batch_size"]
        self.max_parallel_shots = self.config["max_parallel_shots"]

    def _create_batches_from_crops(
        self, crops: list[Crop], character_vault: list[CharacterVaultEntry]
    ) -> list[CharacterIdentificationBatch]:
        """Create batches of crops for character identification"""
        batches = []
        total_crops = len(crops)

        for i in range(0, total_crops, self.batch_size):
            batch_crops = crops[i : i + self.batch_size]
            batch_id = f"batch_{i//self.batch_size + 1}"

            batch = CharacterIdentificationBatch(
                crops=batch_crops, character_vault=character_vault, batch_id=batch_id
            )
            batches.append(batch)

            logger.debug(f"Created {batch_id} with {len(batch_crops)} crops")

        logger.info(
            f"Created {len(batches)} batches from {total_crops} crops (batch size: {self.batch_size})"
        )
        return batches

    def _extract_all_crops_from_shots(
        self, shots: list[Shot]
    ) -> list[tuple[Crop, str]]:
        """Extract all crops from shots with their shot context"""
        all_crops = []

        for shot in shots:
            for _timestamp, crops_list in shot.crops.items():
                for crop in crops_list:
                    all_crops.append((crop, shot.shot_id))

        logger.info(f"Extracted {len(all_crops)} total crops from {len(shots)} shots")
        return all_crops

    def _create_batches_from_shots(
        self, shots: list[Shot], character_vault: list[CharacterVaultEntry]
    ) -> list[CharacterIdentificationBatch]:
        """Create batches from all crops across all shots"""
        all_crops_with_context = self._extract_all_crops_from_shots(shots)
        all_crops = [crop for crop, _ in all_crops_with_context]

        return self._create_batches_from_crops(all_crops, character_vault)

    def _process_shot_sequential(
        self,
        shot: Shot,
        character_vault: list[CharacterVaultEntry],
        identifier_func: Any,
    ) -> Shot:
        """Process a single shot sequentially"""
        logger.info(f"Processing shot {shot.shot_id} sequentially")

        # Create batches for this shot
        shot_crops = []
        for _timestamp, crops_list in shot.crops.items():
            shot_crops.extend(crops_list)

        batches = self._create_batches_from_crops(shot_crops, character_vault)

        # Process each batch
        for batch in batches:
            try:
                identifier_func(batch)
                logger.debug(
                    f"Processed batch {batch.batch_id} for shot {shot.shot_id}"
                )
            except Exception as e:
                logger.error(
                    f"Error processing batch {batch.batch_id} for shot {shot.shot_id}: {e}"
                )

        return shot

    def _process_shot_parallel(
        self,
        shot: Shot,
        character_vault: list[CharacterVaultEntry],
        identifier_func: Any,
    ) -> Shot:
        """Process a single shot in parallel (internal batches)"""
        logger.info(f"Processing shot {shot.shot_id} with internal parallelization")

        # Create batches for this shot
        shot_crops = []
        for _timestamp, crops_list in shot.crops.items():
            shot_crops.extend(crops_list)

        batches = self._create_batches_from_crops(shot_crops, character_vault)

        # Process batches in parallel for this shot
        with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as executor:
            future_to_batch = {
                executor.submit(identifier_func, batch): batch for batch in batches
            }

            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    future.result()
                    logger.debug(
                        f"Processed batch {batch.batch_id} for shot {shot.shot_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing batch {batch.batch_id} for shot {shot.shot_id}: {e}"
                    )

        return shot

    def process_shots_parallel(
        self,
        shots: list[Shot],
        character_vault: list[CharacterVaultEntry],
        identifier_func: Any,
    ) -> list[Shot]:
        """Process multiple shots in parallel with progress tracking"""
        total_shots = len(shots)
        logger.info(
            f"Processing {total_shots} shots in parallel (max workers: {self.max_parallel_shots})"
        )

        start_time = time.time()
        processed_shots = []

        # Process shots in parallel
        with ThreadPoolExecutor(max_workers=self.max_parallel_shots) as executor:
            future_to_shot = {
                executor.submit(
                    self._process_shot_sequential,
                    shot,
                    character_vault,
                    identifier_func,
                ): shot
                for shot in shots
            }

            completed = 0
            for future in as_completed(future_to_shot):
                shot = future_to_shot[future]
                try:
                    processed_shot = future.result()
                    processed_shots.append(processed_shot)
                    completed += 1

                    elapsed = time.time() - start_time
                    avg_time_per_shot = elapsed / completed
                    remaining_shots = total_shots - completed
                    estimated_remaining_time = remaining_shots * avg_time_per_shot

                    logger.info(
                        f"Completed shot {shot.shot_id} ({completed}/{total_shots}) - "
                        f"Elapsed: {elapsed:.1f}s, "
                        f"ETA: {estimated_remaining_time:.1f}s"
                    )

                except Exception as e:
                    logger.error(f"Error processing shot {shot.shot_id}: {e}")
                    # Add the original shot to results even if processing failed
                    processed_shots.append(shot)
                    completed += 1

        total_time = time.time() - start_time
        logger.info(
            f"Parallel processing complete: {completed}/{total_shots} shots processed in {total_time:.1f}s"
        )

        return processed_shots

    def process_shots_sequential(
        self,
        shots: list[Shot],
        character_vault: list[CharacterVaultEntry],
        identifier_func: Any,
    ) -> list[Shot]:
        """Process shots sequentially with progress tracking"""
        total_shots = len(shots)
        logger.info(f"Processing {total_shots} shots sequentially")

        start_time = time.time()
        processed_shots = []

        for i, shot in enumerate(shots, 1):
            logger.info(f"Processing shot {i}/{total_shots}: {shot.shot_id}")

            try:
                processed_shot = self._process_shot_sequential(
                    shot, character_vault, identifier_func
                )
                processed_shots.append(processed_shot)

                elapsed = time.time() - start_time
                avg_time_per_shot = elapsed / i
                remaining_shots = total_shots - i
                estimated_remaining_time = remaining_shots * avg_time_per_shot

                logger.info(
                    f"Completed shot {shot.shot_id} ({i}/{total_shots}) - "
                    f"Elapsed: {elapsed:.1f}s, "
                    f"ETA: {estimated_remaining_time:.1f}s"
                )

            except Exception as e:
                logger.error(f"Error processing shot {shot.shot_id}: {e}")
                # Add the original shot to results even if processing failed
                processed_shots.append(shot)

        total_time = time.time() - start_time
        logger.info(
            f"Sequential processing complete: {len(processed_shots)}/{total_shots} shots processed in {total_time:.1f}s"
        )

        return processed_shots

    def get_processing_stats(self, shots: list[Shot]) -> dict[str, int]:
        """Get statistics about the shots to be processed"""
        total_crops = 0
        total_timestamps = 0

        for shot in shots:
            total_timestamps += len(shot.crops)
            for crops_list in shot.crops.values():
                total_crops += len(crops_list)

        estimated_batches = (total_crops + self.batch_size - 1) // self.batch_size

        return {
            "total_shots": len(shots),
            "total_timestamps": total_timestamps,
            "total_crops": total_crops,
            "batch_size": self.batch_size,
            "estimated_batches": estimated_batches,
            "max_parallel_shots": self.max_parallel_shots,
        }
