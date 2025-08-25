"""
Keyframe Extractor
Downloads videos and extracts keyframes at specific timestamps
"""

import logging
import tempfile
from pathlib import Path

import cv2
import requests
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeyframeExtractor:
    """Extracts keyframes from videos at specified timestamps"""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(tempfile.gettempdir()) / "video_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def extract_keyframes(
        self, video_url: str, timestamps_ms: list[int], output_dir: str
    ) -> dict[int, str]:
        """
        Extract keyframes from video at specified timestamps

        Args:
                        video_url: URL of the video to process
            timestamps_ms: List of timestamps in milliseconds
            output_dir: Directory to save extracted keyframes

            Returns:
            Dictionary mapping timestamp_ms -> saved_image_path
        """
        self.logger.info(f"Extracting {len(timestamps_ms)} keyframes from video")

        # Download video if needed
        video_path = self._download_video_if_needed(video_url)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract frames
        extracted_frames = {}

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = (total_frames / fps) * 1000 if fps > 0 else 0

            self.logger.info(
                f"Video: {fps:.2f} FPS, {total_frames} frames, {duration_ms:.0f}ms duration"
            )

            for timestamp_ms in sorted(timestamps_ms):
                frame_path = self._extract_frame_at_timestamp(
                    cap, timestamp_ms, fps, output_path
                )
                if frame_path:
                    extracted_frames[timestamp_ms] = frame_path

        except Exception as e:
            self.logger.error(f"Error extracting keyframes: {e}")
            raise
        finally:
            if "cap" in locals():
                cap.release()

        self.logger.info(f"✅ Successfully extracted {len(extracted_frames)} keyframes")
        return extracted_frames

    def _download_video_if_needed(self, video_url: str) -> str:
        """Download video to cache if not already present"""
        # Create cache filename from URL
        video_filename = video_url.split("/")[-1]
        if not video_filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_filename += ".mp4"

        cache_path = self.cache_dir / video_filename

        # Check if already cached
        if cache_path.exists() and cache_path.stat().st_size > 0:
            self.logger.info(f"Using cached video: {cache_path}")
            return str(cache_path)

        # Download video
        self.logger.info(f"Downloading video from: {video_url}")
        try:
            response = requests.get(video_url, stream=True, timeout=300)
            response.raise_for_status()

            with open(cache_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            self.logger.info(f"✅ Video downloaded: {cache_path}")
            return str(cache_path)

        except Exception as e:
            self.logger.error(f"Failed to download video: {e}")
            if cache_path.exists():
                cache_path.unlink()  # Remove partial file
            raise

    def _extract_frame_at_timestamp(
        self, cap: cv2.VideoCapture, timestamp_ms: int, fps: float, output_dir: Path
    ) -> str | None:
        """Extract a single frame at the specified timestamp"""
        try:
            # Calculate frame number
            frame_number = int((timestamp_ms / 1000.0) * fps)

            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read frame
            ret, frame = cap.read()
            if not ret:
                self.logger.warning(
                    f"Could not read frame at {timestamp_ms}ms (frame {frame_number})"
                )
                return None

            # Save frame
            output_filename = f"frame_{timestamp_ms:06d}ms.jpg"
            output_path = output_dir / output_filename

            # Convert BGR to RGB for saving
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image.save(output_path, "JPEG", quality=95)

            self.logger.debug(f"Saved frame: {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Error extracting frame at {timestamp_ms}ms: {e}")
            return None

    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cached videos"""
        import time

        if not self.cache_dir.exists():
            return

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for video_file in self.cache_dir.glob("*"):
            if video_file.is_file():
                file_age = current_time - video_file.stat().st_mtime
                if file_age > max_age_seconds:
                    video_file.unlink()
                    self.logger.info(f"Cleaned up cached video: {video_file}")


def main():
    """Test the keyframe extractor"""
    extractor = KeyframeExtractor()

    # Test with sample data
    video_url = "https://content.dashtoon.ai/frameo-merged-videos/shot_final_merged_final_video_final_merged_faec5c098748fe83.mp4"
    timestamps = [0, 420, 840, 1260, 1680]  # First few keyframes
    output_dir = "test_keyframes"

    try:
        extracted_frames = extractor.extract_keyframes(
            video_url, timestamps, output_dir
        )

        print(f"✅ Extracted {len(extracted_frames)} frames:")
        for timestamp, path in extracted_frames.items():
            print(f"  {timestamp}ms -> {path}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
