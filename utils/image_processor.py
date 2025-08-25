import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageDraw, ImageFont

from config import Config
from models import Crop

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles downloading and processing of crop images with ID overlays"""

    def __init__(self) -> None:
        self.font: ImageFont.ImageFont
        self.config = Config.get_image_processing_config()
        self.output_dir = Path(self.config["output_dir"])
        self.font_size = self.config["font_size"]
        self.color = self.config["color"]
        self.outline_color = self.config["outline_color"]
        self.outline_width = self.config["outline_width"]

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize font (fallback to default if custom font not available)
        try:
            self.font = ImageFont.truetype("arial.ttf", self.font_size)
        except OSError:
            try:
                self.font = ImageFont.truetype(
                    "/System/Library/Fonts/Arial.ttf", self.font_size
                )
            except OSError:
                self.font = ImageFont.load_default()
                logger.warning("Using default font - custom font not available")

    def _download_image(self, url: str, timeout: int = 30) -> Image.Image | None:
        """Download image from URL or load from local file with retry logic"""

        # Handle local file URLs
        if url.startswith("file://"):
            try:
                local_path = url.replace("file://", "")
                if os.path.exists(local_path):
                    return Image.open(local_path)
                else:
                    logger.error(f"Local file not found: {local_path}")
                    return None
            except Exception as e:
                logger.error(f"Failed to load local image {url}: {e}")
                return None

        # Handle remote URLs (existing logic)
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()

                # Open image from bytes
                image = Image.open(requests.get(url, stream=True).raw)
                return image

            except requests.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}")
                if attempt < 2:
                    time.sleep(1)  # Brief delay before retry
                continue
            except Exception as e:
                logger.error(f"Unexpected error downloading {url}: {e}")
                break

        logger.error(f"Failed to download image after 3 attempts: {url}")
        return None

    def _add_text_with_outline(
        self, image: Image.Image, text: str, position: tuple[int, int]
    ) -> Image.Image:
        """Add text with outline to image"""
        # Create a copy to avoid modifying original
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)

        # Note: text size calculation removed as it's not currently used
        # Could be added back for dynamic positioning in the future

        # Adjust position to top-left with some padding
        x, y = position
        x = max(10, x)  # Ensure text doesn't go off left edge
        y = max(10, y)  # Ensure text doesn't go off top edge

        # Draw outline (multiple passes for thicker outline)
        for dx in range(-self.outline_width, self.outline_width + 1):
            for dy in range(-self.outline_width, self.outline_width + 1):
                if dx != 0 or dy != 0:  # Skip center pixel
                    draw.text(
                        (x + dx, y + dy), text, font=self.font, fill=self.outline_color
                    )

        # Draw main text
        draw.text((x, y), text, font=self.font, fill=self.color)

        return result_image

    def _generate_filename(self, crop: Crop) -> str:
        """Generate filename for processed crop image"""
        # Extract filename from URL
        url_path = urlparse(crop.crop_url).path
        base_filename = os.path.basename(url_path)

        # Remove extension and add crop ID
        name_without_ext = os.path.splitext(base_filename)[0]
        return f"{name_without_ext}_{crop.crop_id}.jpg"

    def process_crop(self, crop: Crop) -> str | None:
        """Process a single crop: download, add overlay, save"""
        try:
            logger.info(f"Processing crop {crop.crop_id}")

            # Download image
            image = self._download_image(crop.crop_url)
            if image is None:
                logger.error(f"Failed to download image for crop {crop.crop_id}")
                return None

            # Add crop ID overlay
            processed_image = self._add_text_with_outline(
                image,
                crop.crop_id,
                (10, 10),  # Position in top-left with padding
            )

            # Generate output filename
            filename = self._generate_filename(crop)
            output_path = self.output_dir / filename

            # Save processed image
            processed_image.save(output_path, "JPEG", quality=95)

            logger.info(f"Successfully processed crop {crop.crop_id} -> {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error processing crop {crop.crop_id}: {e}")
            return None

    def process_crops(self, crops: list[Crop]) -> list[tuple[Crop, str | None]]:
        """Process multiple crops and return results"""
        results = []

        for crop in crops:
            output_path = self.process_crop(crop)
            results.append((crop, output_path))

        return results

    def process_crops_batch(self, crops: list[Crop]) -> list[tuple[Crop, str | None]]:
        """Process crops in a batch with progress tracking"""
        total_crops = len(crops)
        logger.info(f"Processing {total_crops} crops in batch")

        results = []
        for i, crop in enumerate(crops, 1):
            logger.info(f"Processing crop {i}/{total_crops}: {crop.crop_id}")
            output_path = self.process_crop(crop)
            results.append((crop, output_path))

            # Progress update every 10 crops
            if i % 10 == 0 or i == total_crops:
                logger.info(f"Progress: {i}/{total_crops} crops processed")

        successful = sum(1 for _, path in results if path is not None)
        logger.info(
            f"Batch processing complete: {successful}/{total_crops} crops successful"
        )

        return results

    def cleanup_old_images(self, max_age_hours: int = 24) -> int:
        """Clean up old processed images"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            deleted_count = 0

            for image_file in self.output_dir.glob("*.jpg"):
                file_age = current_time - image_file.stat().st_mtime
                if file_age > max_age_seconds:
                    image_file.unlink()
                    deleted_count += 1

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old processed images")

            return deleted_count

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
