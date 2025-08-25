#!/usr/bin/env python3
"""
scene_detect_compare.py

Detects scene/shot cuts using PySceneDetect AND runs YOLO object detection.
Creates multiple annotated videos and CSV files with comprehensive analysis.
Supports both single video processing and batch processing from CSV files.

FEATURES:
    - Scene/shot cut detection using PySceneDetect detectors
    - YOLO object detection with configurable sampling rate
    - Multiple output videos: cuts-only, YOLO-only, and combined
    - CSV exports for both scene cuts and object detections
    - Organized output structure in evals/{project_id}/ directories
    - Batch processing from CSV files with project IDs and video URLs
    - Automatic video downloading from URLs

USAGE:
    # Single video processing
    python3 scene_detect_compare.py --video input.mp4 [--yolo-fps 2.0] [--yolo-model yolo11n.pt] [--generate-json]

    # Batch processing from CSV
    python3 scene_detect_compare.py --csv projects.csv [--yolo-fps 2.0] [--yolo-model yolo11n.pt] [--generate-json]

CSV FORMAT:
    The CSV file should contain project IDs and video URLs:
    Project IDs,Full Video URL
    PROJvg5wy5R8zFQ9kh6K,https://content.dashtoon.ai/frameo-merged-videos/shot_final_merged_final_video_final_merged_b5de2af54f28a8d4.mp4

EXAMPLES:
    # Basic usage with single video
    python3 scene_detect_compare.py --video movie.mp4

    # Batch processing from CSV file
    python3 scene_detect_compare.py --csv projects.csv --generate-json

    # Higher YOLO sampling rate with larger model
    python3 scene_detect_compare.py --csv projects.csv --yolo-fps 5.0 --yolo-model yolo11s.pt

    # Skip YOLO and only do scene detection
    python3 scene_detect_compare.py --csv projects.csv --skip-yolo

REQUIREMENTS:
    - Python packages: opencv-python, scenedetect, ultralytics, numpy

INSTALL:
    pip install opencv-python scenedetect ultralytics numpy

OUTPUT FILES:
    evals/{project_id}/
    ├── {project_id}_cuts.csv              # Scene cut timestamps and shot lengths
    ├── {project_id}_detections.csv        # YOLO detection results with bounding boxes
    ├── {project_id}_annotated_cuts.mp4    # Video with scene cut overlays only
    ├── {project_id}_annotated_yolo.mp4    # Video with YOLO bounding boxes only
    ├── {project_id}_annotated_all.mp4     # Video with both scene cuts and YOLO detections
    └── analysis.json                      # Structured JSON analysis (with --generate-json)

SCENE DETECTION:
    - Uses PySceneDetect detectors for comprehensive scene detection:
      * ContentDetector: Hard cuts
      * ThresholdDetector: Fades in/out
      * AdaptiveDetector: Fast-motion robustness (optional)
      * HistogramDetector: Subtle cuts (optional)
      * HashDetector: Perceptual changes (optional)

YOLO DETECTION:
    - Uses Ultralytics YOLO11 models for object detection
    - Filters to only detect people and animals (person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe)
    - Samples frames at configurable FPS rate (default: 2.0 FPS)
    - Supports all YOLO11 model variants (n, s, m, l, x)
    - Exports detections with timestamps, class names, confidence, and bounding boxes
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from typing import Any

import cv2

# PySceneDetect imports
try:
    from scenedetect import SceneManager, VideoManager
    from scenedetect.detectors import (
        ContentDetector,
        ThresholdDetector,
    )
except Exception as e:
    print(
        "[ERROR] PySceneDetect not available or failed to import.", e, file=sys.stderr
    )
    print("Install with: pip install scenedetect", file=sys.stderr)
    sys.exit(1)

# YOLO imports
try:
    from ultralytics import YOLO
except Exception as e:
    print(
        "[ERROR] Ultralytics YOLO not available or failed to import.",
        e,
        file=sys.stderr,
    )
    print("Install with: pip install ultralytics", file=sys.stderr)
    sys.exit(1)


# -----------------------------
# Utility
# -----------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_video(url: str, output_path: str) -> bool:
    """
    Download video from URL to local path.

    Args:
        url: Video URL to download
        output_path: Local path to save video

    Returns:
        True if download successful, False otherwise
    """
    try:
        print(f"[DOWNLOAD] Downloading video from: {url}")
        print(f"[DOWNLOAD] Saving to: {output_path}")

        # Ensure output directory exists
        ensure_dir(os.path.dirname(output_path))

        # Download with progress indication
        def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                if block_num % 50 == 0:  # Print every ~50 blocks
                    print(f"[DOWNLOAD] Progress: {percent}%")

        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print(f"[DOWNLOAD] Download complete: {output_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to download video from {url}: {e}")
        return False


def read_csv_input(csv_path: str) -> list[tuple[str, str]]:
    """
    Read CSV file with project IDs and video URLs.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of (project_id, video_url) tuples
    """
    projects = []

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            # Try to detect if first row is header
            f.seek(0)  # Just seek without reading sample

            reader = csv.reader(f)
            first_row = next(reader, None)

            # Skip header if it looks like headers (contains "project" or "url" case-insensitive)
            if first_row and any(
                "project" in cell.lower() or "url" in cell.lower() for cell in first_row
            ):
                print(f"[CSV] Detected header row, skipping: {first_row}")
            else:
                # First row is data, add it back
                if first_row and len(first_row) >= 2:
                    projects.append((first_row[0].strip(), first_row[1].strip()))

            # Read remaining rows
            for row in reader:
                if len(row) >= 2:
                    project_id = row[0].strip()
                    video_url = row[1].strip()
                    if project_id and video_url:
                        projects.append((project_id, video_url))

    except Exception as e:
        print(f"[ERROR] Failed to read CSV file {csv_path}: {e}")
        sys.exit(1)

    print(f"[CSV] Read {len(projects)} projects from {csv_path}")
    return projects


def write_cuts_csv(
    out_path: str, cut_times_s: list[float], video_fps: float, video_frames: int
) -> None:
    """Save cut timestamps and derived shot lengths to CSV."""
    # Derive shot boundaries: assume first scene starts at 0.0
    # Shot boundaries = [0.0] + cut_times + [video_duration]
    duration = video_frames / video_fps if (video_fps and video_frames) else None
    shots = [0.0] + sorted(cut_times_s)
    if duration is not None:
        shots.append(duration)

    rows = []
    for i in range(len(shots) - 1):
        start = shots[i]
        end = shots[i + 1]
        rows.append(
            {
                "shot_index": i,
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "length_s": round(max(0.0, end - start), 3),
            }
        )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["shot_index", "start_s", "end_s", "length_s"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def probe_video_meta(video_path: str) -> tuple[float, int, int, int]:
    """Return (fps, frame_count, width, height)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, frame_count, width, height


def write_detections_csv(out_path: str, detections: list[dict[str, Any]]) -> None:
    """Save YOLO detection results to CSV."""
    if not detections:
        # Create empty CSV with headers
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_number",
                    "timestamp_s",
                    "class_name",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ],
            )
            writer.writeheader()
        return

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_number",
                "timestamp_s",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer.writeheader()
        for detection in detections:
            writer.writerow(detection)


def generate_analysis_json(
    video_basename: str,
    cuts: list[float],
    detections: list[dict[str, Any]],
    fps: float,
    frame_count: int,
    output_dir: str,
) -> str:
    """
    Generate structured JSON analysis from scene cuts and YOLO detections.

    Args:
        video_basename: Base name of the video file
        cuts: List of cut times in seconds
        detections: List of YOLO detection dictionaries
        fps: Video frame rate
        frame_count: Total number of frames
        output_dir: Output directory path

    Returns:
        Path to generated JSON file
    """
    # Build shot boundaries from cuts (same logic as write_cuts_csv)
    duration = frame_count / fps if (fps and frame_count) else None
    shots = [0.0] + sorted(cuts)
    if duration is not None:
        shots.append(duration)

    # Create shot intervals
    shot_intervals = []
    for i in range(len(shots) - 1):
        start = shots[i]
        end = shots[i + 1]
        shot_intervals.append({"shot_index": i, "start_s": start, "end_s": end})

    # Group detections by shot intervals
    shot_detections: dict[int, list[dict[str, Any]]] = {
        int(shot["shot_index"]): [] for shot in shot_intervals
    }

    for detection in detections:
        timestamp = detection["timestamp_s"]

        # Find which shot this detection belongs to
        for shot in shot_intervals:
            if shot["start_s"] <= timestamp < shot["end_s"]:
                # Convert detection to JSON format with coordinates array
                json_detection = {
                    "frame_number": detection["frame_number"],
                    "timestamp_s": detection["timestamp_s"],
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "coordinates": [
                        detection["x1"],
                        detection["y1"],
                        detection["x2"],
                        detection["y2"],
                    ],
                }
                shot_detections[int(shot["shot_index"])].append(json_detection)
                break

    # Build final JSON structure
    shots_dict = {}
    for shot in shot_intervals:
        shot_index = shot["shot_index"]
        shots_dict[str(shot_index)] = {
            "cut": [shot["start_s"], shot["end_s"]],
            "detections": shot_detections[int(shot_index)],
        }

    json_data = {"annotated_video_path": video_basename, "shots": shots_dict}

    # Write JSON file
    json_path = os.path.join(output_dir, "analysis.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Print summary statistics
    total_shots = len(shots_dict)
    total_detections = sum(
        len(shot["detections"])
        for shot in shots_dict.values()
        if isinstance(shot, dict)
        and "detections" in shot
        and isinstance(shot["detections"], list)
    )
    shots_with_detections = sum(
        1
        for shot in shots_dict.values()
        if isinstance(shot, dict)
        and "detections" in shot
        and isinstance(shot["detections"], list)
        and len(shot["detections"]) > 0
    )

    print(f"[JSON] Generated analysis: {json_path}")
    print(
        f"[JSON] Summary: {total_shots} shots, {total_detections} detections, {shots_with_detections} shots with detections"
    )

    if detections:
        class_counts: dict[str, int] = {}
        for detection in detections:
            class_name = detection["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        print(f"[JSON] Detection classes: {dict(sorted(class_counts.items()))}")

    return json_path


# -----------------------------
# YOLO detection
# -----------------------------


def detect_objects_yolo(
    video_path: str, sample_fps: float = 2.0, model_name: str = "yolo11n.pt"
) -> list[dict[str, Any]]:
    """
    Run YOLO object detection on sampled frames from video.
    Filters to only include people and animals.

    Args:
        video_path: Path to input video
        sample_fps: FPS rate for sampling frames (default 2.0)
        model_name: YOLO model to use (default yolo11n.pt)

    Returns:
        List of detection dictionaries with frame info and bounding boxes (people & animals only)
    """
    print(f"[YOLO] Loading model: {model_name}")
    model = YOLO(model_name)

    # COCO dataset classes for people and animals
    # Based on YOLO11 COCO class names
    target_classes = {
        "person",  # People
        "bird",  # Animals
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    }
    print(f"[YOLO] Filtering for classes: {sorted(target_classes)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Calculate frame sampling interval
    frame_interval = max(1, int(video_fps / sample_fps))

    print(f"[YOLO] Video FPS: {video_fps:.1f}, Sample FPS: {sample_fps:.1f}")
    print(
        f"[YOLO] Processing every {frame_interval} frames ({total_frames // frame_interval} total samples)"
    )

    detections = []
    frame_idx = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process frames at sampling interval
        if frame_idx % frame_interval == 0:
            timestamp_s = frame_idx / video_fps

            # Run YOLO detection
            results = model(frame, verbose=False)

            # Extract detections from results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for box, conf, cls_id in zip(
                        boxes, confidences, class_ids, strict=False
                    ):
                        class_name = model.names[cls_id]

                        # Only include people and animals
                        if class_name in target_classes:
                            detection = {
                                "frame_number": frame_idx,
                                "timestamp_s": round(timestamp_s, 3),
                                "class_name": class_name,
                                "confidence": round(float(conf), 3),
                                "x1": round(float(box[0]), 1),
                                "y1": round(float(box[1]), 1),
                                "x2": round(float(box[2]), 1),
                                "y2": round(float(box[3]), 1),
                            }
                            detections.append(detection)

            processed_count += 1
            if processed_count % 50 == 0:
                print(f"[YOLO] Processed {processed_count} frames...")

        frame_idx += 1

    cap.release()
    print(
        f"[YOLO] Detection complete. Found {len(detections)} people/animals in {processed_count} frames."
    )

    # Show breakdown by class
    if detections:
        class_counts: dict[str, int] = {}
        for detection in detections:
            class_name = detection["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        print(f"[YOLO] Class breakdown: {dict(sorted(class_counts.items()))}")

    return detections


# -----------------------------
# PySceneDetect detection
# -----------------------------


def detect_cuts_pyscenedetect(video_path: str) -> list[float]:
    """
    Use PySceneDetect with all detectors for comprehensive scene detection.
    Returns boundary times (seconds) excluding the very first frame at t=0.
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()

    print("[INFO] Using all detectors for comprehensive scene detection...")

    # Hard cuts
    scene_manager.add_detector(ContentDetector(threshold=27.0))

    # # Fades in/out
    scene_manager.add_detector(
        ThresholdDetector(threshold=12, fade_bias=0.0, add_final_scene=True)
    )

    # # Fast-motion robustness --> not needed for most cases
    # scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=3.0, window_width=3, min_content_val=15.0))

    # # Subtle cuts / luminance-stable changes
    # scene_manager.add_detector(HistogramDetector(threshold=0.08, bins=128))

    # # Perceptual changes / low-motion jump cuts
    # scene_manager.add_detector(HashDetector(threshold=0.38, size=16, lowpass=2))

    try:
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()  # union of all detector events
    finally:
        video_manager.release()

    # Each scene is (start_timecode, end_timecode)
    # We want the boundary times (start of each scene except the first)
    boundaries = []
    for i, (start_tc, _) in enumerate(scene_list):
        if i == 0:
            continue
        boundaries.append(start_tc.get_seconds())
    return sorted(boundaries)


# -----------------------------
# Annotation
# -----------------------------


def annotate_video_with_cuts(
    video_path: str, cuts_s: list[float], out_path: str, label_prefix: str = "CUT"
) -> None:
    """
    Annotate the video with a large "CUT #k" overlay on frames nearest to each cut time.
    Shows annotation for multiple frames around each cut for better visibility.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Use H264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("[WARN] H264 codec failed, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Map cut times to frame ranges (show annotation for multiple frames)
    annotation_frames = {}
    annotation_duration_frames = max(1, int(fps * 0.5))  # Show for 0.5 seconds

    for idx, t in enumerate(sorted(cuts_s), start=1):
        center_frame = max(0, int(round(t * fps)))
        # Show annotation for frames around the cut
        for offset in range(
            -annotation_duration_frames // 2, annotation_duration_frames // 2 + 1
        ):
            frame_num = center_frame + offset
            if frame_num >= 0:
                annotation_frames[frame_num] = idx

    print(
        f"[DEBUG] Will annotate {len(annotation_frames)} frames for {len(cuts_s)} cuts"
    )
    print(f"[DEBUG] Cut times: {cuts_s}")
    print(
        f"[DEBUG] Sample annotation frames: {list(list(annotation_frames.keys())[:10])}"
    )

    # Draw parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.5, width / 640.0 * 3.0)  # Increased font size
    thickness = max(3, int(width / 200))  # Increased thickness
    text_color = (255, 255, 255)  # white
    stroke_color = (0, 0, 255)  # red

    frame_idx = 0
    annotations_drawn = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in annotation_frames:
            k = annotation_frames[frame_idx]
            text = f"{label_prefix} #{k}"
            annotations_drawn += 1

            # Draw semi-transparent banner (larger)
            overlay = frame.copy()
            banner_h = int(0.25 * height)  # Increased banner height
            cv2.rectangle(overlay, (0, 0), (width, banner_h), (0, 0, 0), -1)
            alpha = 0.7  # More opaque
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Put large text centered
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = max(20, (width - tw) // 2)
            y = int(banner_h * 0.6)

            # Stroke (thicker for better visibility)
            cv2.putText(
                frame,
                text,
                (x, y),
                font,
                font_scale,
                stroke_color,
                thickness + 6,
                cv2.LINE_AA,
            )
            # Fill
            cv2.putText(
                frame,
                text,
                (x, y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        out.write(frame)
        frame_idx += 1

    # After OpenCV processing is complete
    cap.release()
    out.release()

    # Create temporary video file without audio
    temp_video = out_path.replace(".mp4", "_temp_no_audio.mp4")
    os.rename(out_path, temp_video)

    # Use FFmpeg to combine the processed video with original audio
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                temp_video,
                "-i",
                video_path,
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-y",
                out_path,
            ],
            check=True,
            capture_output=True,
        )

        # Clean up temp file
        os.remove(temp_video)
        print(f"[INFO] Audio preserved in output: {out_path}")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Failed to add audio: {e}")
        print("[WARN] Output video will have no audio")
        os.rename(temp_video, out_path)

    print(f"[DEBUG] Processed {frame_idx} frames, drew {annotations_drawn} annotations")


def annotate_video_with_yolo(
    video_path: str, detections: list[dict[str, Any]], out_path: str
) -> None:
    """
    Annotate the video with YOLO object detection bounding boxes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Use H264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("[WARN] H264 codec failed, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Group detections by frame number for efficient lookup
    detections_by_frame: dict[int, list[dict[str, Any]]] = {}
    for detection in detections:
        frame_num = detection["frame_number"]
        if frame_num not in detections_by_frame:
            detections_by_frame[frame_num] = []
        detections_by_frame[frame_num].append(detection)

    print(
        f"[DEBUG] Will annotate {len(detections_by_frame)} frames with YOLO detections"
    )

    frame_idx = 0
    annotations_drawn = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in detections_by_frame:
            frame_detections = detections_by_frame[frame_idx]
            annotations_drawn += len(frame_detections)

            # Draw bounding boxes and labels
            for detection in frame_detections:
                x1, y1, x2, y2 = (
                    int(detection["x1"]),
                    int(detection["y1"]),
                    int(detection["x2"]),
                    int(detection["y2"]),
                )
                confidence = detection["confidence"]
                class_name = detection["class_name"]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

                # Draw label with confidence
                label = f"{class_name}: {confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                # Draw label background
                cv2.rectangle(
                    frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1
                )

                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )

        out.write(frame)
        frame_idx += 1

    # After OpenCV processing is complete
    cap.release()
    out.release()

    # Create temporary video file without audio
    temp_video = out_path.replace(".mp4", "_temp_no_audio.mp4")
    os.rename(out_path, temp_video)

    # Use FFmpeg to combine the processed video with original audio
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                temp_video,
                "-i",
                video_path,
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-y",
                out_path,
            ],
            check=True,
            capture_output=True,
        )

        # Clean up temp file
        os.remove(temp_video)
        print(f"[INFO] Audio preserved in output: {out_path}")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Failed to add audio: {e}")
        print("[WARN] Output video will have no audio")
        os.rename(temp_video, out_path)

    print(
        f"[DEBUG] Processed {frame_idx} frames, drew {annotations_drawn} YOLO annotations"
    )


def annotate_video_combined(
    video_path: str,
    cuts_s: list[float],
    detections: list[dict[str, Any]],
    out_path: str,
) -> None:
    """
    Annotate the video with both scene cuts and YOLO object detections.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Use H264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("[WARN] H264 codec failed, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Map cut times to frame ranges (show annotation for multiple frames)
    cut_annotation_frames = {}
    annotation_duration_frames = max(1, int(fps * 0.5))  # Show for 0.5 seconds

    for idx, t in enumerate(sorted(cuts_s), start=1):
        center_frame = max(0, int(round(t * fps)))
        # Show annotation for frames around the cut
        for offset in range(
            -annotation_duration_frames // 2, annotation_duration_frames // 2 + 1
        ):
            frame_num = center_frame + offset
            if frame_num >= 0:
                cut_annotation_frames[frame_num] = idx

    # Group detections by frame number for efficient lookup
    detections_by_frame: dict[int, list[dict[str, Any]]] = {}
    for detection in detections:
        frame_num = detection["frame_number"]
        if frame_num not in detections_by_frame:
            detections_by_frame[frame_num] = []
        detections_by_frame[frame_num].append(detection)

    print(
        f"[DEBUG] Will annotate {len(cut_annotation_frames)} frames for cuts and {len(detections_by_frame)} frames for YOLO"
    )

    # Draw parameters for cuts
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.5, width / 640.0 * 3.0)  # Increased font size
    thickness = max(3, int(width / 200))  # Increased thickness
    text_color = (255, 255, 255)  # white
    stroke_color = (0, 0, 255)  # red

    frame_idx = 0
    cut_annotations_drawn = 0
    yolo_annotations_drawn = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw YOLO detections first (so cut annotations appear on top)
        if frame_idx in detections_by_frame:
            frame_detections = detections_by_frame[frame_idx]
            yolo_annotations_drawn += len(frame_detections)

            # Draw bounding boxes and labels
            for detection in frame_detections:
                x1, y1, x2, y2 = (
                    int(detection["x1"]),
                    int(detection["y1"]),
                    int(detection["x2"]),
                    int(detection["y2"]),
                )
                confidence = detection["confidence"]
                class_name = detection["class_name"]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

                # Draw label with confidence
                label = f"{class_name}: {confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                # Draw label background
                cv2.rectangle(
                    frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1
                )

                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )

        # Draw scene cut annotations on top
        if frame_idx in cut_annotation_frames:
            k = cut_annotation_frames[frame_idx]
            text = f"CUT #{k}"
            cut_annotations_drawn += 1

            # Draw semi-transparent banner (larger)
            overlay = frame.copy()
            banner_h = int(0.25 * height)  # Increased banner height
            cv2.rectangle(overlay, (0, 0), (width, banner_h), (0, 0, 0), -1)
            alpha = 0.7  # More opaque
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Put large text centered
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = max(20, (width - tw) // 2)
            y = int(banner_h * 0.6)

            # Stroke (thicker for better visibility)
            cv2.putText(
                frame,
                text,
                (x, y),
                font,
                font_scale,
                stroke_color,
                thickness + 6,
                cv2.LINE_AA,
            )
            # Fill
            cv2.putText(
                frame,
                text,
                (x, y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        out.write(frame)
        frame_idx += 1

    # After OpenCV processing is complete
    cap.release()
    out.release()

    # Create temporary video file without audio
    temp_video = out_path.replace(".mp4", "_temp_no_audio.mp4")
    os.rename(out_path, temp_video)

    # Use FFmpeg to combine the processed video with original audio
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                temp_video,
                "-i",
                video_path,
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-y",
                out_path,
            ],
            check=True,
            capture_output=True,
        )

        # Clean up temp file
        os.remove(temp_video)
        print(f"[INFO] Audio preserved in output: {out_path}")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Failed to add audio: {e}")
        print("[WARN] Output video will have no audio")
        os.rename(temp_video, out_path)

    print(
        f"[DEBUG] Processed {frame_idx} frames, drew {cut_annotations_drawn} cut annotations and {yolo_annotations_drawn} YOLO annotations"
    )


# -----------------------------
# Main
# -----------------------------


def process_single_project(
    project_id: str,
    video_url: str,
    yolo_fps: float,
    yolo_model: str,
    skip_yolo: bool,
    generate_json: bool,
) -> bool:
    """
    Process a single project: download video, run detection, save results.

    Args:
        project_id: Project identifier for output directory
        video_url: URL to download video from
        yolo_fps: FPS for YOLO sampling
        yolo_model: YOLO model to use
        skip_yolo: Whether to skip YOLO detection
        generate_json: Whether to generate JSON analysis

    Returns:
        True if processing successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"[PROJECT] Processing: {project_id}")
    print(f"[PROJECT] Video URL: {video_url}")
    print(f"{'='*60}")

    try:
        # Create output directory structure: evals/{project_id}/
        output_dir = os.path.join("evals", project_id)
        ensure_dir(output_dir)
        print(f"[INFO] Output directory: {output_dir}")

        # Download video to temporary location
        video_filename = f"{project_id}.mp4"
        temp_video_path = os.path.join(output_dir, video_filename)

        if not download_video(video_url, temp_video_path):
            print(f"[ERROR] Failed to download video for project {project_id}")
            return False

        # Probe video for meta / CSV stats
        fps, frame_count, width, height = probe_video_meta(temp_video_path)
        print(
            f"[INFO] Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {frame_count}"
        )

        # PySceneDetect with all detectors
        print("\n[PySceneDetect] Detecting cuts with all detectors...")
        cuts = detect_cuts_pyscenedetect(temp_video_path)
        print(f"[PySceneDetect] Found {len(cuts)} cuts.")

        # Save scene cut results
        cuts_csv_out = os.path.join(output_dir, f"{project_id}_cuts.csv")
        write_cuts_csv(cuts_csv_out, cuts, fps, frame_count)
        print(f"[PySceneDetect] Saved cuts CSV: {cuts_csv_out}")

        # Create annotated video with scene cuts only
        cuts_video_out = os.path.join(output_dir, f"{project_id}_annotated_cuts.mp4")
        print(f"[PySceneDetect] Writing cuts-only annotated video -> {cuts_video_out}")
        annotate_video_with_cuts(
            temp_video_path, cuts, cuts_video_out, label_prefix="CUT"
        )

        # YOLO detection (unless skipped)
        detections = []
        if not skip_yolo:
            print(f"\n[YOLO] Starting object detection (sampling at {yolo_fps} FPS)...")
            detections = detect_objects_yolo(temp_video_path, yolo_fps, yolo_model)

            # Save YOLO detection results
            detections_csv_out = os.path.join(
                output_dir, f"{project_id}_detections.csv"
            )
            write_detections_csv(detections_csv_out, detections)
            print(f"[YOLO] Saved detections CSV: {detections_csv_out}")

            # Create annotated video with YOLO detections only
            yolo_video_out = os.path.join(
                output_dir, f"{project_id}_annotated_yolo.mp4"
            )
            print(f"[YOLO] Writing YOLO-only annotated video -> {yolo_video_out}")
            annotate_video_with_yolo(temp_video_path, detections, yolo_video_out)

            # Create combined annotated video with both cuts and YOLO
            combined_video_out = os.path.join(
                output_dir, f"{project_id}_annotated_all.mp4"
            )
            print(
                f"[COMBINED] Writing combined annotated video -> {combined_video_out}"
            )
            annotate_video_combined(
                temp_video_path, cuts, detections, combined_video_out
            )

        # Generate JSON analysis if requested
        json_path = None
        if generate_json:
            print("\n[JSON] Generating structured analysis...")
            json_path = generate_analysis_json(
                project_id, cuts, detections, fps, frame_count, output_dir
            )

        # Clean up downloaded video file
        try:
            os.remove(temp_video_path)
            print(f"[CLEANUP] Removed temporary video: {temp_video_path}")
        except Exception as e:
            print(f"[WARN] Could not remove temporary video {temp_video_path}: {e}")

        print(f"\n[PROJECT COMPLETE] {project_id} - All outputs saved to: {output_dir}")
        print("Generated files:")
        print(f" - {cuts_csv_out}")
        print(f" - {cuts_video_out}")
        if not skip_yolo:
            print(f" - {detections_csv_out}")
            print(f" - {yolo_video_out}")
            print(f" - {combined_video_out}")
        if json_path:
            print(f" - {json_path}")

        print(f"\nDetection Summary for {project_id}:")
        print(f" - Scene cuts: {len(cuts)}")
        if not skip_yolo:
            print(f" - YOLO detections: {len(detections)}")
            if detections:
                # Count unique classes
                unique_classes = {d["class_name"] for d in detections}
                print(
                    f" - Unique object classes: {len(unique_classes)} ({', '.join(sorted(unique_classes))})"
                )
        else:
            print(" - YOLO detection: SKIPPED")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to process project {project_id}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect scene/shot cuts using PySceneDetect and YOLO object detection with annotated outputs."
    )

    # Input options: either single video or CSV with multiple projects
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video", help="Path to input video file (e.g., input.mp4)"
    )
    input_group.add_argument(
        "--csv", help="Path to CSV file with project IDs and video URLs"
    )

    ap.add_argument(
        "--yolo-fps",
        type=float,
        default=2.0,
        help="FPS for YOLO frame sampling (default: 2.0)",
    )
    ap.add_argument(
        "--yolo-model",
        default="yolo11n.pt",
        help="YOLO model to use (default: yolo11n.pt)",
    )
    ap.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO detection and only do scene detection",
    )
    ap.add_argument(
        "--generate-json",
        action="store_true",
        help="Generate structured JSON analysis file",
    )
    args = ap.parse_args()

    if args.csv:
        # CSV mode: process multiple projects
        projects = read_csv_input(args.csv)

        if not projects:
            print("[ERROR] No valid projects found in CSV file")
            sys.exit(1)

        print(f"\n[BATCH] Starting batch processing of {len(projects)} projects...")

        successful = 0
        failed = 0

        for i, (project_id, video_url) in enumerate(projects, 1):
            print(f"\n[BATCH] Processing project {i}/{len(projects)}")

            if process_single_project(
                project_id,
                video_url,
                args.yolo_fps,
                args.yolo_model,
                args.skip_yolo,
                args.generate_json,
            ):
                successful += 1
            else:
                failed += 1

        print(f"\n{'='*60}")
        print(f"[BATCH COMPLETE] Processed {len(projects)} projects")
        print(f"[BATCH COMPLETE] Successful: {successful}, Failed: {failed}")
        print(f"{'='*60}")

    else:
        # Single video mode (original behavior)
        # Probe video for meta / CSV stats
        fps, frame_count, width, height = probe_video_meta(args.video)
        print(f"[INFO] Video: {args.video}")
        print(
            f"[INFO] Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {frame_count}"
        )

        # Extract base name from input video (without extension)
        video_basename = os.path.splitext(os.path.basename(args.video))[0]

        # Create output directory structure: evals/{video_basename}/
        output_dir = os.path.join("evals", video_basename)
        ensure_dir(output_dir)
        print(f"[INFO] Output directory: {output_dir}")

        # PySceneDetect with all detectors
        print("\n[PySceneDetect] Detecting cuts with all detectors...")
        cuts = detect_cuts_pyscenedetect(args.video)
        print(f"[PySceneDetect] Found {len(cuts)} cuts.")

        # Save scene cut results
        cuts_csv_out = os.path.join(output_dir, f"{video_basename}_cuts.csv")
        write_cuts_csv(cuts_csv_out, cuts, fps, frame_count)
        print(f"[PySceneDetect] Saved cuts CSV: {cuts_csv_out}")

        # Create annotated video with scene cuts only
        cuts_video_out = os.path.join(
            output_dir, f"{video_basename}_annotated_cuts.mp4"
        )
        print(f"[PySceneDetect] Writing cuts-only annotated video -> {cuts_video_out}")
        annotate_video_with_cuts(args.video, cuts, cuts_video_out, label_prefix="CUT")

        # YOLO detection (unless skipped)
        detections = []
        if not args.skip_yolo:
            print(
                f"\n[YOLO] Starting object detection (sampling at {args.yolo_fps} FPS)..."
            )
            detections = detect_objects_yolo(args.video, args.yolo_fps, args.yolo_model)

            # Save YOLO detection results
            detections_csv_out = os.path.join(
                output_dir, f"{video_basename}_detections.csv"
            )
            write_detections_csv(detections_csv_out, detections)
            print(f"[YOLO] Saved detections CSV: {detections_csv_out}")

            # Create annotated video with YOLO detections only
            yolo_video_out = os.path.join(
                output_dir, f"{video_basename}_annotated_yolo.mp4"
            )
            print(f"[YOLO] Writing YOLO-only annotated video -> {yolo_video_out}")
            annotate_video_with_yolo(args.video, detections, yolo_video_out)

            # Create combined annotated video with both cuts and YOLO
            combined_video_out = os.path.join(
                output_dir, f"{video_basename}_annotated_all.mp4"
            )
            print(
                f"[COMBINED] Writing combined annotated video -> {combined_video_out}"
            )
            annotate_video_combined(args.video, cuts, detections, combined_video_out)

        # Generate JSON analysis if requested
        json_path = None
        if args.generate_json:
            print("\n[JSON] Generating structured analysis...")
            json_path = generate_analysis_json(
                video_basename, cuts, detections, fps, frame_count, output_dir
            )

        print(f"\n[DONE] All outputs saved to: {output_dir}")
        print("Generated files:")
        print(f" - {cuts_csv_out}")
        print(f" - {cuts_video_out}")
        if not args.skip_yolo:
            print(f" - {detections_csv_out}")
            print(f" - {yolo_video_out}")
            print(f" - {combined_video_out}")
        if json_path:
            print(f" - {json_path}")

        print("\nDetection Summary:")
        print(f" - Scene cuts: {len(cuts)}")
        if not args.skip_yolo:
            print(f" - YOLO detections: {len(detections)}")
            if detections:
                # Count unique classes
                unique_classes = {d["class_name"] for d in detections}
                print(
                    f" - Unique object classes: {len(unique_classes)} ({', '.join(sorted(unique_classes))})"
                )
        else:
            print(" - YOLO detection: SKIPPED")

        if json_path:
            print(" - JSON analysis: Generated")


if __name__ == "__main__":
    main()
