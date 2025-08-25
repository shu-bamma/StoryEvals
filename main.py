import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from character_enricher import CharacterEnricher
from character_identification_pipeline import CharacterIdentificationPipeline
from critique_agent import CritiqueAgent
from models import Character
from video_shot_processing_pipeline import VideoShotProcessingPipeline


def setup_logging(run_dir: str) -> logging.Logger:
    """Setup logging for pipeline execution"""
    log_file = os.path.join(run_dir, "pipeline_log.txt")

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Logger
    logger = logging.getLogger("StoryEvalsPipeline")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def setup_run_directories(run_name: str) -> str:
    """Create directory structure for pipeline run"""
    run_dir = f"data/logs/{run_name}"

    # Create directories for all steps
    Path(f"{run_dir}/1-character_enrichment").mkdir(parents=True, exist_ok=True)
    Path(f"{run_dir}/2-video_shots_with_crops").mkdir(parents=True, exist_ok=True)
    Path(f"{run_dir}/3-character_identification").mkdir(parents=True, exist_ok=True)
    Path(f"{run_dir}/4-critique_results").mkdir(parents=True, exist_ok=True)

    return run_dir


def check_existing_step_results(
    run_name: str, projects: list[dict[str, Any]]
) -> dict[str, bool]:
    """Check which steps have already been completed for the given run
    Returns:
        Dictionary mapping step names to completion status
    """
    run_dir = f"data/logs/{run_name}"

    step_status = {
        "character_enrichment": False,
        "video_shot_processing": False,
        "character_identification": False,
        "critique_agent": False,
    }

    # Check step 1: Character Enrichment
    step1_dir = f"{run_dir}/1-character_enrichment"
    if os.path.exists(step1_dir):
        # Check if we have results for all projects
        project_ids = [
            p.get("project_id", f"project_{i}") for i, p in enumerate(projects, 1)
        ]
        step1_files = [f for f in os.listdir(step1_dir) if f.endswith(".json")]
        step1_complete = all(f"{pid}.json" in step1_files for pid in project_ids)
        step_status["character_enrichment"] = step1_complete

    # Check step 2: Video Shot Processing
    step2_dir = f"{run_dir}/2-video_shots_with_crops"
    if os.path.exists(step2_dir):
        project_ids = [
            p.get("project_id", f"project_{i}") for i, p in enumerate(projects, 1)
        ]
        step2_files = [f for f in os.listdir(step2_dir) if f.endswith(".json")]
        step2_complete = all(f"{pid}.json" in step2_files for pid in project_ids)
        step_status["video_shot_processing"] = step2_complete

    # Check step 3: Character Identification
    step3_dir = f"{run_dir}/3-character_identification"
    if os.path.exists(step3_dir):
        project_ids = [
            p.get("project_id", f"project_{i}") for i, p in enumerate(projects, 1)
        ]
        step3_files = [f for f in os.listdir(step3_dir) if f.endswith(".json")]
        step3_complete = all(f"{pid}.json" in step3_files for pid in project_ids)
        step_status["character_identification"] = step3_complete

    # Check step 4: Critique Agent
    step4_dir = f"{run_dir}/4-critique_results"
    if os.path.exists(step4_dir):
        project_ids = [
            p.get("project_id", f"project_{i}") for i, p in enumerate(projects, 1)
        ]
        step4_files = [f for f in os.listdir(step4_dir) if f.endswith(".json")]
        step4_complete = all(f"{pid}.json" in step4_files for pid in project_ids)
        step_status["critique_agent"] = step4_complete

    return step_status


def load_projects_from_vault(num_projects: int | None = None) -> list[dict[str, Any]]:
    """Load projects from character vault data
    Args:
        num_projects: Number of projects to load. If None, loads all projects.
    """
    input_path = (
        "data/cleaned_data/3-character_vault_with_full_video_with_clips_by_proj_id.json"
    )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Master project file not found: {input_path}")

    with open(input_path) as f:
        all_projects: list[dict[str, Any]] = json.load(f)

    if num_projects is None:
        # Process all projects
        selected_projects = all_projects
    else:
        # Take first N projects
        selected_projects = (
            all_projects[:num_projects] if num_projects > 0 else all_projects
        )

    return selected_projects


def run_character_enrichment_step(
    projects: list[dict[str, Any]], run_dir: str, logger: logging.Logger
) -> dict[str, Any]:
    """Step 1: Enrich characters using CharacterEnricher"""
    logger.info(f"=== STEP 1: Character Enrichment ({len(projects)} projects) ===")
    start_time = time.time()

    enricher = CharacterEnricher()
    stats: dict[str, Any] = {
        "total": len(projects),
        "successful": 0,
        "failed": 0,
        "errors": [],
    }

    for i, project in enumerate(projects, 1):
        project_id = project.get("project_id", f"project_{i}")
        logger.info(f"Processing project {i}/{len(projects)}: {project_id}")

        try:
            # Enrich characters for this project
            enriched_project: list[dict[str, Any]] | None = (
                enricher.enrich_from_vault_data([project])
            )

            if enriched_project and len(enriched_project) > 0:
                # Save individual project result
                output_path = f"{run_dir}/1-character_enrichment/{project_id}.json"
                with open(output_path, "w") as f:
                    json.dump(enriched_project[0], f, indent=2, ensure_ascii=False)
                stats["successful"] += 1
                logger.info(f"✅ {project_id}: Character enrichment completed")
            else:
                raise ValueError("No enriched data returned")

        except Exception as e:
            stats["failed"] += 1
            error_msg = f"❌ {project_id}: Character enrichment failed - {str(e)}"
            logger.error(error_msg)
            if isinstance(stats["errors"], list):
                stats["errors"].append({"project_id": project_id, "error": str(e)})

    duration = time.time() - start_time
    stats["duration_s"] = round(duration, 2)

    logger.info(
        f"Character Enrichment Complete: {stats['successful']}/{stats['total']} successful ({duration:.1f}s)"
    )
    return stats


def run_video_shot_processing_step(
    projects: list[dict[str, Any]],
    run_dir: str,
    logger: logging.Logger,
    create_debug_video: bool = False,
) -> dict[str, Any]:
    """Step 2: Process projects through video shot processing pipeline"""
    logger.info(f"=== STEP 2: Video Shot Processing ({len(projects)} projects) ===")
    start_time = time.time()

    pipeline = VideoShotProcessingPipeline()
    stats: dict[str, Any] = {
        "total": len(projects),
        "successful": 0,
        "failed": 0,
        "errors": [],
    }

    for i, project in enumerate(projects, 1):
        project_id = project.get("project_id", f"project_{i}")
        logger.info(f"Processing project {i}/{len(projects)}: {project_id}")

        try:
            # Load enriched character data
            enriched_char_path = f"{run_dir}/1-character_enrichment/{project_id}.json"

            if not os.path.exists(enriched_char_path):
                raise FileNotFoundError(
                    f"Character enrichment result not found: {enriched_char_path}"
                )

            with open(enriched_char_path) as f:
                enriched_project_data: dict[str, Any] = json.load(f)

            # Process single project
            result = pipeline._process_single_project(
                enriched_project_data, create_debug_video
            )

            # Save individual project result using pipeline's save method (handles serialization)
            output_path = f"{run_dir}/2-video_shots_with_crops/{project_id}.json"
            pipeline.save_results([result], output_path)  # Use pipeline's method

            stats["successful"] += 1
            logger.info(f"✅ {project_id}: Video processing completed")

        except Exception as e:
            stats["failed"] += 1
            error_msg = f"❌ {project_id}: Video processing failed - {str(e)}"
            logger.error(error_msg)
            if isinstance(stats["errors"], list):
                stats["errors"].append({"project_id": project_id, "error": str(e)})

    duration = time.time() - start_time
    stats["duration_s"] = round(duration, 2)

    logger.info(
        f"Video Shot Processing Complete: {stats['successful']}/{stats['total']} successful ({duration:.1f}s)"
    )
    return stats


def run_character_identification_step(
    projects: list[dict[str, Any]], run_dir: str, logger: logging.Logger
) -> dict[str, Any]:
    """Step 3: Process projects through character identification pipeline"""
    logger.info(f"=== STEP 3: Character Identification ({len(projects)} projects) ===")
    start_time = time.time()

    pipeline = CharacterIdentificationPipeline()
    stats: dict[str, Any] = {
        "total": len(projects),
        "successful": 0,
        "failed": 0,
        "errors": [],
    }

    for i, project in enumerate(projects, 1):
        project_id = project.get("project_id", f"project_{i}")
        logger.info(f"Processing project {i}/{len(projects)}: {project_id}")

        try:
            # Load video shot processing result
            video_shot_path = f"{run_dir}/2-video_shots_with_crops/{project_id}.json"

            if not os.path.exists(video_shot_path):
                raise FileNotFoundError(
                    f"Video shot processing result not found: {video_shot_path}"
                )

            with open(video_shot_path) as f:
                video_shot_data = json.load(f)

            # Fix: Extract single project from array
            if isinstance(video_shot_data, list) and len(video_shot_data) > 0:
                single_project_data = video_shot_data[
                    0
                ]  # Extract first (and only) project
            else:
                single_project_data = video_shot_data

            # Process through character identification
            result = pipeline.process(single_project_data)

            # Save individual project result
            output_path = f"{run_dir}/3-character_identification/{project_id}.json"
            pipeline.save_results(result, output_path)

            stats["successful"] += 1
            logger.info(f"✅ {project_id}: Character identification completed")

        except Exception as e:
            stats["failed"] += 1
            error_msg = f"❌ {project_id}: Character identification failed - {str(e)}"
            logger.error(error_msg)
            if isinstance(stats["errors"], list):
                stats["errors"].append({"project_id": project_id, "error": str(e)})

    duration = time.time() - start_time
    stats["duration_s"] = round(duration, 2)

    logger.info(
        f"Character Identification Complete: {stats['successful']}/{stats['total']} successful ({duration:.1f}s)"
    )
    return stats


def run_critique_agent_step(
    projects: list[dict[str, Any]], run_dir: str, logger: logging.Logger
) -> dict[str, Any]:
    """Step 4: Process projects through critique agent"""
    logger.info(f"=== STEP 4: Critique Agent ({len(projects)} projects) ===")
    start_time = time.time()

    critique_agent = CritiqueAgent()
    stats: dict[str, Any] = {
        "total": len(projects),
        "successful": 0,
        "failed": 0,
        "errors": [],
    }

    for i, project in enumerate(projects, 1):
        project_id = project.get("project_id", f"project_{i}")
        logger.info(f"Processing project {i}/{len(projects)}: {project_id}")

        try:
            # Load character identification result
            char_id_path = f"{run_dir}/3-character_identification/{project_id}.json"

            if not os.path.exists(char_id_path):
                raise FileNotFoundError(
                    f"Character identification result not found: {char_id_path}"
                )

            with open(char_id_path) as f:
                char_id_data = json.load(f)

            # Convert dictionary to VideoOutput object (fix the attribute error)
            from models import (  # Use CharacterVaultEntry instead of Character
                VideoOutput,
            )

            # Create VideoOutput object with proper attributes
            video_output = VideoOutput(
                project_id=char_id_data.get("project_id", project_id),
                clip_url=char_id_data.get("video_url", ""),
                prompt="",  # Required field
                enhanced_prompt="",  # Required field
                reference_image="",  # Required field
                thumbnail_image="",  # Required field
                characters=[
                    Character(
                        char_id=char.get("char_id", ""),
                        name=char.get("name", ""),
                        image=char.get("ref_image", ""),
                        description=None,
                    )
                    for char in char_id_data.get("character_vault", [])
                ],
                debug=None,
            )

            # Process through critique agent
            critique_result = critique_agent.evaluate_video_output(video_output)

            # Save individual project result
            output_path = f"{run_dir}/4-critique_results/{project_id}.json"
            with open(output_path, "w") as f:
                json.dump(critique_result.__dict__, f, indent=2, ensure_ascii=False)

            stats["successful"] += 1
            logger.info(f"✅ {project_id}: Critique evaluation completed")

        except Exception as e:
            stats["failed"] += 1
            error_msg = f"❌ {project_id}: Critique evaluation failed - {str(e)}"
            logger.error(error_msg)
            if isinstance(stats["errors"], list):
                stats["errors"].append({"project_id": project_id, "error": str(e)})

    duration = time.time() - start_time
    stats["duration_s"] = round(duration, 2)

    logger.info(
        f"Critique Agent Complete: {stats['successful']}/{stats['total']} successful ({duration:.1f}s)"
    )
    return stats


def save_pipeline_summary(
    run_name: str,
    run_dir: str,
    all_stats: dict[str, dict],
    total_duration: float,
    num_projects: int,
) -> None:
    """Save overall pipeline execution summary"""
    summary = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "total_projects": num_projects,
        "steps": all_stats,
        "total_duration_s": round(total_duration, 2),
        "final_success_rate": min(
            stats.get("successful", 0) / max(stats.get("total", 1), 1)
            for stats in all_stats.values()
        )
        if all_stats
        else 0.0,
    }

    summary_path = f"{run_dir}/pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="StoryEvals Full Pipeline")
    parser.add_argument("--run_name", required=True, help="Name for this pipeline run")
    parser.add_argument(
        "--num_projects",
        type=int,
        default=None,
        help="Number of projects to process (default: all projects)",
    )
    parser.add_argument(
        "--debug_video",
        action="store_true",
        help="Create debug videos for video shot processing",
    )
    parser.add_argument(
        "--skip_critique", action="store_true", help="Skip critique agent step"
    )

    args = parser.parse_args()

    # Setup
    run_dir = setup_run_directories(args.run_name)
    logger = setup_logging(run_dir)

    logger.info(f"🚀 Starting StoryEvals Pipeline: {args.run_name}")

    pipeline_start_time = time.time()
    all_stats = {}

    try:
        # Load projects from character vault
        logger.info("📖 Loading projects from character vault...")
        projects = load_projects_from_vault(args.num_projects)

        if args.num_projects is None:
            logger.info(f"✅ Processing ALL {len(projects)} projects")
        else:
            logger.info(
                f"✅ Processing {len(projects)} projects (limited by --num_projects {args.num_projects})"
            )

        # Check existing step results to resume from where we left off
        logger.info("🔍 Checking existing step results...")
        step_status = check_existing_step_results(args.run_name, projects)

        logger.info("📊 Step completion status:")
        for step, completed in step_status.items():
            status_icon = "✅" if completed else "⏳"
            logger.info(
                f"  {status_icon} {step}: {'Completed' if completed else 'Pending'}"
            )

        # Step 1: Character Enrichment (skip if already completed)
        if not step_status["character_enrichment"]:
            logger.info("🔄 Starting Step 1: Character Enrichment...")
            enrichment_stats = run_character_enrichment_step(projects, run_dir, logger)
            all_stats["character_enrichment"] = enrichment_stats
        else:
            logger.info(
                "⏭️  Step 1: Character Enrichment already completed, skipping..."
            )
            all_stats["character_enrichment"] = {
                "total": len(projects),
                "successful": len(projects),
                "failed": 0,
                "errors": [],
                "duration_s": 0,
                "skipped": True,
            }

        # Step 2: Video Shot Processing (skip if already completed)
        if not step_status["video_shot_processing"]:
            logger.info("🔄 Starting Step 2: Video Shot Processing...")
            video_stats = run_video_shot_processing_step(
                projects, run_dir, logger, args.debug_video
            )
            all_stats["video_shot_processing"] = video_stats
        else:
            logger.info(
                "⏭️  Step 2: Video Shot Processing already completed, skipping..."
            )
            all_stats["video_shot_processing"] = {
                "total": len(projects),
                "successful": len(projects),
                "failed": 0,
                "errors": [],
                "duration_s": 0,
                "skipped": True,
            }

        # Step 3: Character Identification (skip if already completed)
        if not step_status["character_identification"]:
            logger.info("🔄 Starting Step 3: Character Identification...")
            char_stats = run_character_identification_step(projects, run_dir, logger)
            all_stats["character_identification"] = char_stats
        else:
            logger.info(
                "⏭️  Step 3: Character Identification already completed, skipping..."
            )
            all_stats["character_identification"] = {
                "total": len(projects),
                "successful": len(projects),
                "failed": 0,
                "errors": [],
                "duration_s": 0,
                "skipped": True,
            }

        # Step 4: Critique Agent (optional, skip if already completed)
        if not args.skip_critique:
            if not step_status["critique_agent"]:
                logger.info("🔄 Starting Step 4: Critique Agent...")
                critique_stats = run_critique_agent_step(projects, run_dir, logger)
                all_stats["critique_agent"] = critique_stats
            else:
                logger.info("⏭️  Step 4: Critique Agent already completed, skipping...")
                all_stats["critique_agent"] = {
                    "total": len(projects),
                    "successful": len(projects),
                    "failed": 0,
                    "errors": [],
                    "duration_s": 0,
                    "skipped": True,
                }
        else:
            logger.info("⏭️  Skipping critique agent step (--skip_critique flag)")

        # Save summary
        total_duration = time.time() - pipeline_start_time
        save_pipeline_summary(
            args.run_name, run_dir, all_stats, total_duration, len(projects)
        )

        logger.info(f"🎉 Pipeline Complete! Total time: {total_duration:.1f}s")
        logger.info(f"📁 Results saved in: {run_dir}")

    except Exception as e:
        logger.error(f"💥 Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
