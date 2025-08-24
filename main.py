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
from config import Config
from critique_agent import CritiqueAgent
from video_shot_processing_pipeline import VideoShotProcessingPipeline


def setup_logging(run_dir: str) -> logging.Logger:
    """Setup logging for pipeline execution"""
    log_file = os.path.join(run_dir, "pipeline_log.txt")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Logger
    logger = logging.getLogger('StoryEvalsPipeline')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_run_directories(run_name: str) -> str:
    """Create directory structure for pipeline run"""
    run_dir = f"data/logs/{run_name}"
    
    # Create directories
    Path(f"{run_dir}/2-video_shots_with_crops").mkdir(parents=True, exist_ok=True)
    Path(f"{run_dir}/3-character_identification").mkdir(parents=True, exist_ok=True)
    Path(f"{run_dir}/4-critique_results").mkdir(parents=True, exist_ok=True)
    
    return run_dir


def load_enriched_projects(num_projects: int | None = None) -> list[dict[str, Any]]:
    """Load projects from enriched characters data
    
    Args:
        num_projects: Number of projects to load. If None, loads all projects.
    """
    input_path = "data/logs/1-enriched_characters.json"
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Enriched characters file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        all_projects = json.load(f)
    
    if num_projects is None:
        # Process all projects
        selected_projects = all_projects
    else:
        # Take first N projects
        selected_projects = all_projects[:num_projects] if num_projects > 0 else all_projects
    
    return selected_projects


def run_video_shot_processing_step(
    projects: list[dict[str, Any]], 
    run_dir: str, 
    logger: logging.Logger,
    create_debug_video: bool = False
) -> dict[str, Any]:
    """Step 1: Process projects through video shot processing pipeline"""
    logger.info(f"=== STEP 1: Video Shot Processing ({len(projects)} projects) ===")
    start_time = time.time()
    
    pipeline = VideoShotProcessingPipeline()
    stats = {"total": len(projects), "successful": 0, "failed": 0, "errors": []}
    
    for i, project in enumerate(projects, 1):
        project_id = project.get("project_id", f"project_{i}")
        logger.info(f"Processing project {i}/{len(projects)}: {project_id}")
        
        try:
            # Process single project
            result = pipeline._process_single_project(project, create_debug_video)
            
            # Save individual project result using pipeline's save method (handles serialization)
            output_path = f"{run_dir}/2-video_shots_with_crops/{project_id}.json"
            pipeline.save_results([result], output_path)  # Use pipeline's method
            
            stats["successful"] += 1
            logger.info(f"✅ {project_id}: Video processing completed")
            
        except Exception as e:
            stats["failed"] += 1
            error_msg = f"❌ {project_id}: Video processing failed - {str(e)}"
            logger.error(error_msg)
            stats["errors"].append({"project_id": project_id, "error": str(e)})
    
    duration = time.time() - start_time
    stats["duration_s"] = round(duration, 2)
    
    logger.info(f"Video Shot Processing Complete: {stats['successful']}/{stats['total']} successful ({duration:.1f}s)")
    return stats


def run_character_identification_step(
    projects: list[dict[str, Any]], 
    run_dir: str, 
    logger: logging.Logger
) -> dict[str, Any]:
    """Step 2: Process projects through character identification pipeline"""
    logger.info(f"=== STEP 2: Character Identification ({len(projects)} projects) ===")
    start_time = time.time()
    
    pipeline = CharacterIdentificationPipeline()
    stats = {"total": len(projects), "successful": 0, "failed": 0, "errors": []}
    
    for i, project in enumerate(projects, 1):
        project_id = project.get("project_id", f"project_{i}")
        logger.info(f"Processing project {i}/{len(projects)}: {project_id}")
        
        try:
            # Load video shot processing result
            video_shot_path = f"{run_dir}/2-video_shots_with_crops/{project_id}.json"
            
            if not os.path.exists(video_shot_path):
                raise FileNotFoundError(f"Video shot processing result not found: {video_shot_path}")
            
            with open(video_shot_path, 'r') as f:
                video_shot_data = json.load(f)
            
            # Fix: Extract single project from array
            if isinstance(video_shot_data, list) and len(video_shot_data) > 0:
                single_project_data = video_shot_data[0]  # Extract first (and only) project
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
            stats["errors"].append({"project_id": project_id, "error": str(e)})
    
    duration = time.time() - start_time
    stats["duration_s"] = round(duration, 2)
    
    logger.info(f"Character Identification Complete: {stats['successful']}/{stats['total']} successful ({duration:.1f}s)")
    return stats


def run_critique_agent_step(
    projects: list[dict[str, Any]], 
    run_dir: str, 
    logger: logging.Logger
) -> dict[str, Any]:
    """Step 3: Process projects through critique agent"""
    logger.info(f"=== STEP 3: Critique Agent ({len(projects)} projects) ===")
    start_time = time.time()
    
    critique_agent = CritiqueAgent()
    stats = {"total": len(projects), "successful": 0, "failed": 0, "errors": []}
    
    for i, project in enumerate(projects, 1):
        project_id = project.get("project_id", f"project_{i}")
        logger.info(f"Processing project {i}/{len(projects)}: {project_id}")
        
        try:
            # Load character identification result
            char_id_path = f"{run_dir}/3-character_identification/{project_id}.json"
            
            if not os.path.exists(char_id_path):
                raise FileNotFoundError(f"Character identification result not found: {char_id_path}")
            
            with open(char_id_path, 'r') as f:
                char_id_data = json.load(f)
            
            # Convert dictionary to VideoOutput object (fix the attribute error)
            from models import VideoOutput, CharacterVaultEntry, Shot  # Use CharacterVaultEntry instead of Character
            
            # Create VideoOutput object with proper attributes
            video_output = VideoOutput(
                project_id=char_id_data.get("project_id", project_id),
                video_url=char_id_data.get("video_url", ""),
                character_vault=[CharacterVaultEntry(**char) for char in char_id_data.get("character_vault", [])],  # Use CharacterVaultEntry
                shots=[Shot(**shot) for shot in char_id_data.get("shots", [])]
            )
            
            # Process through critique agent
            critique_result = critique_agent.evaluate_video_output(video_output)
            
            # Save individual project result
            output_path = f"{run_dir}/4-critique_results/{project_id}.json"
            with open(output_path, 'w') as f:
                json.dump(critique_result.__dict__, f, indent=2, ensure_ascii=False)
            
            stats["successful"] += 1
            logger.info(f"✅ {project_id}: Critique evaluation completed")
            
        except Exception as e:
            stats["failed"] += 1
            error_msg = f"❌ {project_id}: Critique evaluation failed - {str(e)}"
            logger.error(error_msg)
            stats["errors"].append({"project_id": project_id, "error": str(e)})
    
    duration = time.time() - start_time
    stats["duration_s"] = round(duration, 2)
    
    logger.info(f"Critique Agent Complete: {stats['successful']}/{stats['total']} successful ({duration:.1f}s)")
    return stats


def save_pipeline_summary(
    run_name: str, 
    run_dir: str, 
    all_stats: dict[str, dict], 
    total_duration: float, 
    num_projects: int
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
        ) if all_stats else 0.0
    }
    
    summary_path = f"{run_dir}/pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="StoryEvals Full Pipeline")
    parser.add_argument("--run_name", required=True, help="Name for this pipeline run")
    parser.add_argument("--num_projects", type=int, default=None, help="Number of projects to process (default: all projects)")
    parser.add_argument("--debug_video", action="store_true", help="Create debug videos for video shot processing")
    parser.add_argument("--skip_critique", action="store_true", help="Skip critique agent step")
    
    args = parser.parse_args()
    
    # Setup
    run_dir = setup_run_directories(args.run_name)
    logger = setup_logging(run_dir)
    
    logger.info(f"🚀 Starting StoryEvals Pipeline: {args.run_name}")
    
    pipeline_start_time = time.time()
    all_stats = {}
    
    try:
        # Load enriched projects
        logger.info("📖 Loading enriched character data...")
        projects = load_enriched_projects(args.num_projects)
        
        if args.num_projects is None:
            logger.info(f"✅ Processing ALL {len(projects)} projects")
        else:
            logger.info(f"✅ Processing {len(projects)} projects (limited by --num_projects {args.num_projects})")
        
        # Step 1: Video Shot Processing
        video_stats = run_video_shot_processing_step(
            projects, run_dir, logger, args.debug_video
        )
        all_stats["video_shot_processing"] = video_stats
        
        # Step 2: Character Identification
        char_stats = run_character_identification_step(
            projects, run_dir, logger
        )
        all_stats["character_identification"] = char_stats
        
        # Step 3: Critique Agent (optional)
        if not args.skip_critique:
            critique_stats = run_critique_agent_step(
                projects, run_dir, logger
            )
            all_stats["critique_agent"] = critique_stats
        else:
            logger.info("⏭️  Skipping critique agent step")
        
        # Save summary
        total_duration = time.time() - pipeline_start_time
        save_pipeline_summary(args.run_name, run_dir, all_stats, total_duration, len(projects))
        
        logger.info(f"🎉 Pipeline Complete! Total time: {total_duration:.1f}s")
        logger.info(f"📁 Results saved in: {run_dir}")
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
