#!/usr/bin/env python3
"""
Script to transform 2-full_video_with_clips_by_proj_id by adding a character_vault key
at the top of each object containing the union of all unique characters.
"""

import json
from typing import Any


def extract_unique_characters(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract unique characters from all clips in a project.
    Returns a list of unique characters based on their Name.
    """
    unique_characters = {}

    for clip in clips:
        if "characters" in clip:
            for character in clip["characters"]:
                if "Name" in character:
                    char_name = character["Name"]
                    if char_name not in unique_characters:
                        unique_characters[char_name] = character

    # Convert back to list and sort by name for consistency
    return sorted(unique_characters.values(), key=lambda x: x["Name"])


def transform_data_with_character_vault(
    data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Transform the data by adding character_vault at the top of each object.
    """
    transformed_data = []

    for project in data:
        # Create a new project object with character_vault at the top
        new_project: dict[str, Any] = {}

        # Add character_vault first
        new_project["character_vault"] = extract_unique_characters(
            project.get("clips", [])
        )

        # Add all other keys from the original project
        for key, value in project.items():
            new_project[key] = value

        transformed_data.append(new_project)

    return transformed_data


def main() -> None:
    """Main function to read, transform, and save the data."""
    input_file = "data/cleaned_data/2-full_video_with_clips_by_proj_id.json"
    output_file = (
        "data/cleaned_data/3-character_vault_with_full_video_with_clips_by_proj_id.json"
    )

    try:
        # Read the input file
        print(f"Reading {input_file}...")
        with open(input_file, encoding="utf-8") as f:
            data = json.load(f)

        print(f"Found {len(data)} projects to transform.")

        # Transform the data
        print("Transforming data...")
        transformed_data = transform_data_with_character_vault(data)

        # Save the transformed data
        print(f"Saving transformed data to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transformed_data, f, indent=2, ensure_ascii=False)

        print("Transformation completed successfully!")

        # Print summary for each project
        for i, project in enumerate(transformed_data):
            project_id = project.get("project_id", f"Project_{i}")
            character_count = len(project.get("character_vault", []))
            print(f"  {project_id}: {character_count} unique characters")

            # Print character names
            for char in project.get("character_vault", []):
                print(f"    - {char.get('Name', 'Unknown')}")

    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
