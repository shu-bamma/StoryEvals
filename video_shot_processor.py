import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# PySceneDetect imports
try:
    from scenedetect import SceneManager, VideoManager
    from scenedetect.detectors import ContentDetector, ThresholdDetector
except ImportError as e:
    logging.error(f"PySceneDetect not available: {e}")
    raise

# YOLO imports
try:
    from ultralytics import YOLO
except ImportError as e:
    logging.error(f"Ultralytics YOLO not available: {e}")
    raise

from config import Config
from models import Crop, CropQuality, Shot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoShotProcessor:
    """Main processor for video shot boundary detection and character crop extraction"""

    def __init__(self) -> None:
        self.config = Config.get_video_shot_processing_config()
        
        # Initialize YOLO model
        self.model = YOLO(self.config["detection_model"])
        self.target_classes = set(self.config["detection_classes"])
        
        # Create temp directories
        os.makedirs(self.config["temp_video_dir"], exist_ok=True)
        os.makedirs(self.config["temp_crops_dir"], exist_ok=True)

    def download_video(self, video_url: str, project_id: str) -> str:
        """Download video to local temp storage, skip if already exists"""
        try:
            filename = f"{project_id}.mp4"
            temp_video_path = os.path.join(self.config["temp_video_dir"], filename)
            
            # Check if video already exists
            if os.path.exists(temp_video_path):
                # Verify the file is not empty/corrupted
                if os.path.getsize(temp_video_path) > 0:
                    logger.info(f"Video already exists, skipping download: {temp_video_path}")
                    return temp_video_path
                else:
                    logger.warning(f"Existing video file is empty, re-downloading: {temp_video_path}")
                    os.remove(temp_video_path)
            
            logger.info(f"Downloading video from: {video_url}")
            logger.info(f"Saving to: {temp_video_path}")
            
            urllib.request.urlretrieve(video_url, temp_video_path)
            logger.info(f"Video download complete: {temp_video_path}")
            return temp_video_path
            
        except Exception as e:
            logger.error(f"Failed to download video from {video_url}: {e}")
            raise

    def detect_shot_boundaries(self, video_path: str) -> list[dict[str, Any]]:
        """Use PySceneDetect to find shot boundaries"""
        logger.info("Starting shot boundary detection")
        
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        
        # Add detectors (from scene_detect_compare.py)
        scene_manager.add_detector(ContentDetector(threshold=self.config["shot_boundary_threshold"]))
        scene_manager.add_detector(
            ThresholdDetector(threshold=12, fade_bias=0.0, add_final_scene=True)
        )
        
        try:
            video_manager.set_downscale_factor()
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()
        finally:
            video_manager.release()
        
        # Convert scene list to shot boundaries
        boundaries = [0.0]  # Start with first frame
        for i, (start_tc, _) in enumerate(scene_list):
            if i > 0:  # Skip first scene start
                boundaries.append(start_tc.get_seconds())
        
        # Get video duration for last boundary
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        if duration > 0:
            boundaries.append(duration)
        
        # Create shot data structures
        shots = []
        for i in range(len(boundaries) - 1):
            start_ms = int(boundaries[i] * 1000)
            end_ms = int(boundaries[i + 1] * 1000)
            shot_id = f"shot_{i+1:04d}"
            
            # Generate keyframes at fixed intervals
            keyframes_ms = []
            current_ms = start_ms
            while current_ms <= end_ms:
                keyframes_ms.append(current_ms)
                current_ms += self.config["keyframe_interval_ms"]
            
            shots.append({
                "shot_id": shot_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "keyframes_ms": keyframes_ms
            })
        
        logger.info(f"Detected {len(shots)} shots with {sum(len(s['keyframes_ms']) for s in shots)} total keyframes")
        return shots

    def expand_bbox(self, box: np.ndarray, frame_shape: tuple[int, int, int], expansion: float = 0.15) -> np.ndarray:
        """Expand bounding box by specified percentage to include hair/ears"""
        x1, y1, x2, y2 = box
        h, w = frame_shape[:2]
        
        # Calculate expansion
        width = x2 - x1
        height = y2 - y1
        
        expand_w = width * expansion
        expand_h = height * expansion
        
        # Apply expansion with bounds checking
        x1 = max(0, x1 - expand_w)
        y1 = max(0, y1 - expand_h)
        x2 = min(w, x2 + expand_w)
        y2 = min(h, y2 + expand_h)
        
        return np.array([x1, y1, x2, y2])

    def extract_crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract crop from frame using bounding box"""
        x1, y1, x2, y2 = bbox.astype(int)
        return frame[y1:y2, x1:x2]

    def calculate_blur(self, image: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def apply_nms(self, boxes: np.ndarray, confidences: np.ndarray, class_ids: np.ndarray, iou_threshold: float = 0.5) -> list[int]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if len(boxes) == 0:
            return []
        
        # Convert boxes to the format expected by cv2.dnn.NMSBoxes [x, y, width, height]
        nms_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            nms_boxes.append([x1, y1, width, height])
        
        # Apply OpenCV's NMS
        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            confidences.tolist(),
            score_threshold=self.config["detection_confidence"],
            nms_threshold=iou_threshold
        )
        
        # Handle the case where indices is empty or None
        if indices is not None and len(indices) > 0:
            # OpenCV returns indices as nested arrays, flatten them
            return indices.flatten().tolist()
        else:
            return []

    def upload_crop(self, crop_image: np.ndarray, crop_id: str, project_id: str) -> str:
        """Upload crop to storage and return URL"""
        # Save locally first
        filename = f"{crop_id}.jpg"
        local_dir = os.path.join(self.config["temp_crops_dir"], project_id)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        
        # Save crop image
        cv2.imwrite(local_path, crop_image)
        
        if self.config["crop_upload_enabled"]:
            # TODO: Implement actual upload to your storage infrastructure
            # For now, return constructed URL following your pattern
            crop_url = f"{self.config['crop_base_url']}{project_id}/{filename}"
            logger.debug(f"Generated crop URL: {crop_url}")
        else:
            # Return local file path
            crop_url = f"file://{os.path.abspath(local_path)}"
        
        return crop_url

    def process_shots_with_yolo(self, video_path: str, shots: list[dict[str, Any]], project_id: str) -> list[Shot]:
        """Extract keyframes and run YOLO detection on each"""
        logger.info(f"Processing {len(shots)} shots with YOLO detection")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            logger.error("Invalid video FPS")
            cap.release()
            return []
        
        processed_shots = []
        
        for shot in shots:
            shot_crops = {}
            
            logger.info(f"Processing shot {shot['shot_id']} with {len(shot['keyframes_ms'])} keyframes")
            
            # Process each keyframe in this shot
            for keyframe_ms in shot["keyframes_ms"]:
                # Seek to keyframe
                frame_number = int((keyframe_ms / 1000.0) * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if not ret:
                    shot_crops[keyframe_ms] = []
                    logger.warning(f"Failed to read frame at {keyframe_ms}ms")
                    continue
                
                # Run YOLO detection on this frame
                results = self.model(frame, verbose=False)
                crops_for_keyframe = []
                
                crop_index = 0
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)
                        
                        # Filter for people and animals only first
                        valid_indices = []
                        valid_boxes = []
                        valid_confidences = []
                        valid_class_ids = []
                        
                        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids, strict=False)):
                            class_name = self.model.names[cls_id]
                            if class_name in self.target_classes and conf >= self.config["detection_confidence"]:
                                valid_indices.append(i)
                                valid_boxes.append(box)
                                valid_confidences.append(conf)
                                valid_class_ids.append(cls_id)
                        
                        if not valid_boxes:
                            continue
                        
                        # Apply NMS if enabled
                        if self.config.get("nms_enabled", True):
                            nms_indices = self.apply_nms(
                                np.array(valid_boxes), 
                                np.array(valid_confidences), 
                                np.array(valid_class_ids),
                                iou_threshold=self.config.get("nms_iou_threshold", 0.5)
                            )
                            # Filter to only NMS-selected detections
                            final_boxes = [valid_boxes[i] for i in nms_indices]
                            final_confidences = [valid_confidences[i] for i in nms_indices]
                            final_class_ids = [valid_class_ids[i] for i in nms_indices]
                        else:
                            # Use all valid detections if NMS is disabled
                            final_boxes = valid_boxes
                            final_confidences = valid_confidences
                            final_class_ids = valid_class_ids
                        
                        # Process final detections after NMS
                        for box, conf, cls_id in zip(final_boxes, final_confidences, final_class_ids, strict=False):
                            # Expand bbox by specified percentage
                            expanded_box = self.expand_bbox(box, frame.shape, self.config["bbox_expansion"])
                            
                            # Extract crop from frame
                            crop_image = self.extract_crop(frame, expanded_box)
                            
                            if crop_image.size == 0:
                                logger.warning(f"Empty crop extracted at {keyframe_ms}ms")
                                continue
                            
                            # Generate crop ID
                            crop_id = f"c_{shot['shot_id']}_t{keyframe_ms:04d}_{crop_index}"
                            
                            # Upload crop and get URL
                            crop_url = self.upload_crop(crop_image, crop_id, project_id)
                            
                            # Calculate quality metrics
                            blur_score = self.calculate_blur(crop_image)
                            
                            # Normalize bbox coordinates
                            h, w = frame.shape[:2]
                            bbox_norm = [
                                float(expanded_box[0] / w),  # x
                                float(expanded_box[1] / h),  # y
                                float((expanded_box[2] - expanded_box[0]) / w),  # width
                                float((expanded_box[3] - expanded_box[1]) / h)   # height
                            ]
                            
                            # Create Crop object (YOLO detection with NMS)
                            crop = Crop(
                                crop_id=crop_id,
                                bbox_norm=bbox_norm,
                                crop_url=crop_url,
                                detector=self.config["detection_model"],
                                face_conf=round(float(conf), 3),
                                quality=CropQuality(
                                    blur=round(blur_score, 1),
                                    pose=None  # YOLO doesn't provide pose data
                                )
                            )
                            
                            crops_for_keyframe.append(crop)
                            crop_index += 1
                
                shot_crops[keyframe_ms] = crops_for_keyframe
                logger.debug(f"Found {len(crops_for_keyframe)} crops at keyframe {keyframe_ms}ms")
            
            # Create final Shot object
            processed_shot = Shot(
                shot_id=shot["shot_id"],
                keyframes_ms=shot["keyframes_ms"],
                crops=shot_crops
            )
            processed_shots.append(processed_shot)
            
            total_crops = sum(len(crops) for crops in shot_crops.values())
            logger.info(f"Completed shot {shot['shot_id']} with {total_crops} total crops")
        
        cap.release()
        
        total_shots = len(processed_shots)
        total_crops = sum(
            sum(len(crops) for crops in shot.crops.values()) 
            for shot in processed_shots
        )
        logger.info(f"YOLO processing complete: {total_shots} shots, {total_crops} total crops")
        
        return processed_shots

    def create_debug_video(self, video_path: str, shots: list[dict[str, Any]], processed_shots: list[Shot], project_id: str) -> str:
        """Create a debug video with shot boundaries and YOLO detections overlaid"""
        import cv2
        
        debug_output_path = os.path.join(self.config["temp_video_dir"], f"{project_id}_debug.mp4")
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(debug_output_path, fourcc, fps, (width, height))
            
            frame_idx = 0
            current_shot_idx = 0
            
            # Create shot boundary lookup
            shot_boundaries = []
            for shot in shots:
                shot_boundaries.append({
                    'start_ms': shot['start_ms'],
                    'end_ms': shot['end_ms'],
                    'shot_id': shot['shot_id']
                })
            
            # Create crops lookup by timestamp
            crops_by_timestamp = {}
            for shot in processed_shots:
                for timestamp_ms, crops_list in shot.crops.items():
                    # Convert string timestamp to int for matching
                    crops_by_timestamp[int(timestamp_ms)] = crops_list
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time_ms = int((frame_idx / fps) * 1000)
                
                # Draw shot boundaries
                current_shot = None
                for i, shot in enumerate(shot_boundaries):
                    if shot['start_ms'] <= current_time_ms <= shot['end_ms']:
                        current_shot = shot
                        break
                
                # Draw shot info
                if current_shot:
                    cv2.putText(frame, f"Shot: {current_shot['shot_id']}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Time: {current_time_ms}ms", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw shot boundary markers
                for shot in shot_boundaries:
                    if abs(current_time_ms - shot['start_ms']) < 500:  # Within 500ms of boundary
                        cv2.line(frame, (0, 0), (width, height), (0, 0, 255), 3)
                        cv2.putText(frame, f"SHOT START: {shot['shot_id']}", 
                                   (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Draw YOLO detections - look for nearby keyframes since frame timing might not be exact
                matching_crops = []
                for keyframe_timestamp, crops_list in crops_by_timestamp.items():
                    # Allow for small timing differences (±210ms, half of keyframe interval)
                    if abs(current_time_ms - keyframe_timestamp) <= 210:
                        matching_crops.extend(crops_list)
                        # Debug info - show which keyframe we're displaying
                        cv2.putText(frame, f"Keyframe: {keyframe_timestamp}ms", 
                                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw all matching crops
                for crop in matching_crops:
                    # Convert normalized bbox to pixel coordinates
                    x_norm, y_norm, w_norm, h_norm = crop.bbox_norm
                    x = int(x_norm * width)
                    y = int(y_norm * height)
                    w = int(w_norm * width)
                    h = int(h_norm * height)
                    
                    # Draw bounding box (blue for YOLO detections)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Draw crop info
                    cv2.putText(frame, f"Conf: {crop.face_conf:.2f}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(frame, crop.crop_id, 
                               (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                out.write(frame)
                frame_idx += 1
            
            cap.release()
            out.release()
            
            logger.info(f"Debug video saved: {debug_output_path}")
            return debug_output_path
            
        except Exception as e:
            logger.error(f"Failed to create debug video: {e}")
            return ""

    def process_video(self, video_url: str, project_id: str, character_vault: list[dict[str, Any]], create_debug_video: bool = False) -> dict[str, Any]:
        """Main function that orchestrates the entire process"""
        start_time = time.time()
        logger.info(f"Starting video processing for project: {project_id}")
        
        try:
            # Step 1: Download video
            video_path = self.download_video(video_url, project_id)
            
            # Step 2: Detect shot boundaries
            shots = self.detect_shot_boundaries(video_path)
            
            # Step 3: Process shots with YOLO
            processed_shots = self.process_shots_with_yolo(video_path, shots, project_id)
            
            # Step 4: Create debug video if requested
            debug_video_path = None
            if create_debug_video:
                debug_video_path = self.create_debug_video(video_path, shots, processed_shots, project_id)
            
            # Step 5: Create final output structure
            result = {
                "project_id": project_id,
                "video_url": video_url,
                "character_vault": character_vault,  # Pass through
                "number_of_shots": len(processed_shots),
                "shots": processed_shots
            }
            
            if debug_video_path:
                result["debug_video_path"] = debug_video_path
            
            # Cleanup original video (but keep debug video)
            try:
                os.remove(video_path)
                logger.info(f"Cleaned up temporary video: {video_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up video file {video_path}: {e}")
            
            processing_time = time.time() - start_time
            logger.info(f"Video processing complete for {project_id} in {processing_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing video for project {project_id}: {e}")
            raise

    def save_results(self, result: dict[str, Any], output_path: str | None = None) -> str:
        """Save video processing results to JSON file"""
        if output_path is None:
            output_path = self.config["output_path"]
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert Shot objects to dict for JSON serialization
        serializable_result = result.copy()
        serializable_result["shots"] = []
        
        for shot in result["shots"]:
            shot_dict = {
                "shot_id": shot.shot_id,
                "keyframes_ms": shot.keyframes_ms,
                "crops": {}
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
                        "face_conf": crop.face_conf,  # Keep for backward compatibility
                        "detection_confidence": crop.face_conf,  # Add new field
                        "quality": {
                            "blur": crop.quality.blur if crop.quality else None,
                            "pose": crop.quality.pose if crop.quality else None,  # Fix missing pose
                        } if crop.quality else None
                    }
                    shot_dict["crops"][str(timestamp)].append(crop_dict)
            
            serializable_result["shots"].append(shot_dict)
        
        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Video processing results saved to {output_file}")
        return str(output_file)
