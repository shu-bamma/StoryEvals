import json
import time
from typing import Any

import pandas as pd
import streamlit as st


def load_data(file_path: str) -> list[dict[str, Any]]:
    """Load the transformed data from JSON file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return []


def save_data(data: list[dict[str, Any]], file_path: str) -> None:
    """Save the updated data back to the original JSON file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        st.success("Data saved successfully!")
    except Exception as e:
        st.error(f"Error saving data: {e}")


def update_clip_timings(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Update clip_index, start, and end times based on new order."""
    updated_clips = []
    current_time = 0

    for i, clip in enumerate(clips):
        # Calculate duration of the clip
        duration = clip["end"] - clip["start"]

        # Update clip properties
        updated_clip = clip.copy()
        updated_clip["clip_index"] = i
        updated_clip["start"] = current_time
        updated_clip["end"] = current_time + duration

        updated_clips.append(updated_clip)
        current_time += duration

    return updated_clips


def format_timestamp(seconds: int) -> str:
    """Convert seconds to T00:00:00 format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"T{hours:02d}:{minutes:02d}:{secs:02d}"


def display_clip_info(clip: dict[str, Any], index: int) -> None:
    """Display information about a single clip."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.write(f"**Clip {index}**")
        st.write(f"**Start:** {format_timestamp(clip['start'])}")
        st.write(f"**End:** {format_timestamp(clip['end'])}")
        st.write(f"**Duration:** {format_timestamp(clip['end'] - clip['start'])}")

    with col2:
        if "image_prompt" in clip:
            st.text_area(
                "Image Prompt", clip["image_prompt"], height=100, key=f"prompt_{index}"
            )

        if "characters" in clip:
            st.write("**Characters:**")
            for char in clip["characters"]:
                st.write(f"- {char.get('Name', 'Unknown')}")

    with col3:
        if "reference_image" in clip:
            st.image(
                clip["reference_image"],
                caption="Reference Image",
                use_column_width=True,
            )


def reorder_clips_with_inputs(
    clips: list[dict[str, Any]],
    project_data: dict[str, Any],
    data: list[dict[str, Any]],
    selected_index: int,
) -> None:
    """Allow reordering clips by typing new indices."""
    st.write("**Reorder Clips:**")
    st.write("Type the new index (0-based) for each clip, then click 'Apply New Order'")

    # Create input fields for new indices
    new_indices = []

    # Use enumerate to get both index and clip
    for i, clip in enumerate(clips):
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.write(
                f"**Clip {i}** ({format_timestamp(clip['start'])}-{format_timestamp(clip['end'])})"
            )

        with col2:
            new_index = st.number_input(
                "New Index",
                min_value=0,
                max_value=len(clips) - 1,
                value=i,
                key=f"index_{i}",
                label_visibility="collapsed",
            )
            new_indices.append(new_index)

        with col3:
            if st.button("🗑️", key=f"delete_{i}", type="secondary"):
                # Delete this specific clip by its content, not by index
                clip_to_delete = clip
                # Only remove from one list since project_data is a reference to data[selected_index]
                project_data["clips"].remove(clip_to_delete)

                # Save the changes immediately
                save_data(
                    data, "data/cleaned_data/2-full_video_with_clips_by_proj_id.json"
                )
                st.success("Clip deleted and saved!")
                time.sleep(1)
                st.rerun()

    # Apply reordering button
    if st.button("🔄 Apply New Order", type="primary"):
        # Check for duplicate indices
        if len(set(new_indices)) != len(new_indices):
            st.error("Duplicate indices detected! Each clip must have a unique index.")
            return

        # Create new order based on input indices
        new_order = []
        for target_index in range(len(clips)):
            # Find which clip should go at this target index
            for i, desired_index in enumerate(new_indices):
                if desired_index == target_index:
                    new_order.append(clips[i])
                    break

        # Update timings
        updated_clips = update_clip_timings(new_order)

        # Update both project_data and main data
        project_data["clips"] = updated_clips
        data[selected_index]["clips"] = updated_clips

        # Save directly to file
        save_data(data, "data/cleaned_data/2-full_video_with_clips_by_proj_id.json")

        st.success("Clips reordered and saved! Timings updated automatically.")
        time.sleep(1)
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Video Clip Manager",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🎬 Video Clip Manager")
    st.markdown("Manage and reorder video clips with manual reordering and deletion")

    # Load data
    data = load_data("data/cleaned_data/2-full_video_with_clips_by_proj_id.json")

    if not data:
        st.error("No data loaded. Please check the file path.")
        return

    # Sidebar for project selection
    st.sidebar.header("Project Selection")
    project_options = [f"{item['job_id']} - {item['project_id']}" for item in data]
    selected_project = st.sidebar.selectbox("Select Project", project_options, index=0)

    # Get selected project data
    selected_index = project_options.index(selected_project)
    project_data = data[selected_index]

    # Display project info
    st.header(f"Project: {project_data['job_id']}")
    st.write(f"**Project ID:** {project_data['project_id']}")
    st.write(f"**Video URL:** {project_data['video_url']}")
    st.write(f"**Total Clips:** {len(project_data['clips'])}")

    # Calculate and display total duration
    total_duration = sum(clip["end"] - clip["start"] for clip in project_data["clips"])
    st.write(f"**Total Duration:** {format_timestamp(total_duration)}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Clip Management & Details")

        # Toggle for reorder controls
        show_reorder = st.toggle("🔀 Show Reorder Controls", value=False)

        if show_reorder:
            st.info(
                "Type the new index (0-based) for each clip, then click 'Apply New Order' to reorder. Use 🗑️ to delete clips."
            )

            # Reorder clips using input fields
            reorder_clips_with_inputs(
                project_data["clips"], project_data, data, selected_index
            )

        # Display individual clips with details
        st.subheader("Clip Details")
        for i, clip in enumerate(project_data["clips"]):
            with st.expander(
                f"Clip {i}: {format_timestamp(clip['start'])}-{format_timestamp(clip['end'])}",
                expanded=False,
            ):
                display_clip_info(clip, i)

    with col2:
        st.subheader("Actions")

        # Summary
        st.subheader("Summary")
        total_duration = sum(
            clip["end"] - clip["start"] for clip in project_data["clips"]
        )
        st.write(f"**Total Duration:** {format_timestamp(total_duration)}")
        st.write(f"**Remaining Clips:** {len(project_data['clips'])}")

        # Show current clip order
        st.subheader("Current Order")
        for i, clip in enumerate(project_data["clips"]):
            st.write(
                f"{i+1}. Clip {clip['clip_index']} ({format_timestamp(clip['start'])}-{format_timestamp(clip['end'])})"
            )

    # Display all clips in a table format
    st.subheader("All Clips Table")
    clips_df = pd.DataFrame(project_data["clips"])
    if not clips_df.empty:
        # Select relevant columns for display
        display_columns = ["clip_index", "start", "end", "url"]
        if "image_prompt" in clips_df.columns:
            display_columns.append("image_prompt")

        st.dataframe(clips_df[display_columns], use_container_width=True)


if __name__ == "__main__":
    main()
