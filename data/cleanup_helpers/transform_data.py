import csv
import json
import random
from datetime import datetime
from typing import Any


def generate_job_id() -> str:
    """Generate a job ID with format job_YYYY_MM_DD_XXX"""
    now = datetime.now()
    random_suffix = random.randint(1, 999)
    return f"job_{now.year}_{now.month:02d}_{now.day:02d}_{random_suffix:03d}"


def get_full_video_url(project_id: str, csv_data: list[dict[str, str]]) -> str | None:
    """Get the full video URL for a project ID from the CSV data"""
    for row in csv_data:
        if row["Project IDs"] == project_id:
            return row["Full Video URL"]
    return None


def transform_data() -> list[dict[str, Any]]:
    # Load project IDs
    with open("data/cleaned_data/0-project_ids.json") as f:
        project_ids = json.load(f)

    # Load story evals results
    with open("data/cleaned_data/1-clips_by_proj_id.json") as f:
        story_evals = json.load(f)

    # Load CSV data
    csv_data = []
    with open("data/proj_id_full_vid.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_data.append(row)

    transformed_data = []

    for project_id in project_ids:
        # Get clips for this project
        if project_id in story_evals.get("projects", {}):
            clips = story_evals["projects"][project_id]

            # Get full video URL
            full_video_url = get_full_video_url(project_id, csv_data)

            # Skip if no full video URL available
            if (
                not full_video_url
                or full_video_url == "NA"
                or full_video_url.strip() == ""
            ):
                continue

            # Transform clips
            transformed_clips = []
            for i, clip in enumerate(clips):
                # Generate random timing values
                start_time = random.randint(1000, 5000)
                end_time = start_time + random.randint(2000, 4000)

                transformed_clip = {
                    "clip_index": i,
                    "start": start_time,
                    "end": end_time,
                    "url": clip.get("ClipUrl", ""),
                    "image_prompt": clip.get("Prompt", ""),
                    "enhanced_prompt": clip.get("EnhancedPrompt", ""),
                    "reference_image": clip.get("ReferenceImage", ""),
                    "characters": clip.get("Characters", []),
                }
                transformed_clips.append(transformed_clip)

            # Create the main object
            transformed_object = {
                "job_id": generate_job_id(),
                "video_url": full_video_url,
                "project_id": project_id,
                "clips": transformed_clips,
            }

            transformed_data.append(transformed_object)

    return transformed_data


def main() -> None:
    try:
        transformed_data = transform_data()

        # Save to file
        with open("data/cleaned_data/transformed_data.json", "w") as f:
            json.dump(transformed_data, f, indent=2)

        print(f"Successfully transformed {len(transformed_data)} projects")
        print("Data saved to data/cleaned_data/transformed_data.json")

        # Print first few entries as example
        if transformed_data:
            print("\nExample transformed object:")
            print(json.dumps(transformed_data[0], indent=2))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
