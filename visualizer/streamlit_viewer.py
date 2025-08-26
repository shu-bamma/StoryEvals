"""
Streamlit Pipeline Results Viewer
Interactive visualization of character identification pipeline results
"""

import logging
import os
import tempfile
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from extract_keyframes import KeyframeExtractor
from PIL import Image, ImageDraw, ImageFont
from plotly.subplots import make_subplots
from process_results import (
    CropData,
    KeyframeData,
    PipelineResultsProcessor,
    ProjectData,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamlitViewer:
    """Streamlit-based pipeline results viewer"""

    def __init__(self):
        self.processor = PipelineResultsProcessor()
        self.extractor = KeyframeExtractor()
        self._init_session_state()

    def _init_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'project_data': None,
            'extracted_frames': {},
            'current_timestamp_ms': 0,
            'keyframes_extracted': False,
            'current_project_id': None,
            'extraction_in_progress': False,
            'keyframe_lookup': {},  # For efficient keyframe access
            'timeline_chart_key': 0,  # For forcing chart updates
            'last_slider_value': None,  # Track slider changes
            'chart_click_source': False,  # Track if change came from chart click
            # Multi-project support
            'projects': {},  # Dict[project_id, ProjectData]
            'project_tabs': [],  # List of project_ids for tab ordering
            'active_project_id': None,  # Currently selected project
            'folder_loaded': False,
            'projects_extracted_frames': {},  # Dict[project_id, Dict[timestamp, path]]
            'projects_extraction_status': {},  # Dict[project_id, bool]
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @property
    def project_data(self) -> ProjectData | None:
        """Compatibility property - returns currently active project"""
        if not st.session_state.active_project_id:
            return st.session_state.project_data  # Fallback to single project mode
        return st.session_state.projects.get(st.session_state.active_project_id)

    def _get_current_extracted_frames(self) -> dict[int, str]:
        """Get extracted frames for current project"""
        if st.session_state.active_project_id:
            return st.session_state.projects_extracted_frames.get(st.session_state.active_project_id, {})
        return st.session_state.extracted_frames  # Fallback to single project mode

    def _set_current_extracted_frames(self, frames: dict[int, str]):
        """Set extracted frames for current project"""
        if st.session_state.active_project_id:
            st.session_state.projects_extracted_frames[st.session_state.active_project_id] = frames
        else:
            st.session_state.extracted_frames = frames

    def _get_current_extraction_status(self) -> bool:
        """Get extraction status for current project"""
        if st.session_state.active_project_id:
            return st.session_state.projects_extraction_status.get(st.session_state.active_project_id, False)
        return st.session_state.keyframes_extracted

    def _set_current_extraction_status(self, status: bool):
        """Set extraction status for current project"""
        if st.session_state.active_project_id:
            st.session_state.projects_extraction_status[st.session_state.active_project_id] = status
        else:
            st.session_state.keyframes_extracted = status

    @staticmethod
    def get_detection_colors():
        """Get consistent detection colors for bounding boxes and UI"""
        return [
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#FFFF00",  # Yellow
            "#FF00FF",  # Magenta
            "#00FFFF",  # Cyan
            "#FFA500",  # Orange
            "#800080",  # Purple
            "#FF69B4",  # Hot Pink
            "#32CD32"   # Lime Green
        ]

    def run(self):
        """Main Streamlit app"""
        st.set_page_config(
            page_title="Pipeline Results Viewer",
            page_icon="🎬",
            layout="wide"
        )

        st.title("🎬 Character Identification Pipeline Viewer")
        st.markdown("---")

        # Sidebar for file selection and controls
        with st.sidebar:
            self._render_sidebar()

        # Main content
        if self.project_data is not None:
            self._render_main_content()
        else:
            st.info("👈 Please select a project results file from the sidebar to begin")

    def _render_sidebar(self):
        """Render the sidebar with file selection and controls"""
        st.header("📂 Project Selection")

        # Multi-project folder loading
        self._render_folder_selection()

        # Project tabs if multiple projects loaded
        if st.session_state.project_tabs:
            self._render_project_tabs()

        # Separator
        st.markdown("---")

        # Single file uploader (existing functionality)
        st.subheader("Or load single file:")
        uploaded_file = st.file_uploader(
            "Upload Character Identification Results JSON",
            type=['json'],
            help="Select a JSON file from the pipeline results"
        )

        if uploaded_file is not None:
            file_content = uploaded_file.getvalue().decode('utf-8')
            file_hash = hash(file_content)
            project_id = f"upload_{file_hash}"

            if st.session_state.current_project_id != project_id:
                self._load_project(uploaded_file, project_id, is_upload=True)

        # Alternative: Browse local files (existing functionality)
        st.subheader("Or browse local files:")

        # Look for results in common locations
        common_paths = [
            "data/logs",
            "data/logs/test_run_20/3-character_identification"
        ]

        available_files = self._get_available_files(common_paths)

        if available_files:
            selected_file = st.selectbox(
                "Select from available files:",
                [""] + available_files,
                format_func=lambda x: Path(x).name if x else "Choose a file..."
            )

            if selected_file and st.session_state.current_project_id != selected_file:
                self._load_project(selected_file, selected_file, is_upload=False)
        else:
            st.info("No result files found in common locations")

        # Project info
        if self.project_data is not None:
            self._render_project_info()

    def _get_available_files(self, common_paths: list[str]) -> list[str]:
        """Get list of available JSON files"""
        available_files = []
        for path in common_paths:
            if Path(path).exists():
                for json_file in Path(path).glob("**/*.json"):
                    available_files.append(str(json_file))
        return available_files

    def _render_project_info(self):
        """Render project information"""
        project_data = self.project_data  # Use property

        st.markdown("---")
        st.subheader("📊 Project Info")

        st.write(f"**Project ID:** `{project_data.project_id}`")
        st.write(f"**Total Keyframes:** {len(project_data.keyframes)}")
        st.write(f"**Characters:** {len(project_data.character_vault)}")
        st.write(f"**Duration:** {project_data.get_timeline_duration_ms():,} ms")

        # Show extraction status using new methods
        current_extracted = self._get_current_extraction_status()
        if current_extracted:
            st.success("✅ Keyframes extracted and ready")
        elif st.session_state.extraction_in_progress:
            st.info("🔄 Extracting keyframes...")
        else:
            st.warning("⚠️ Keyframes not extracted yet")

        # Statistics
        self._render_statistics(project_data)

    def _render_statistics(self, project_data: ProjectData):
        """Render project statistics"""
        stats = self.processor.get_processing_stats(project_data)

        st.markdown("**Detection Stats:**")
        st.write(f"• Keyframes with detections: {stats['keyframes_with_detections']}")
        st.write(f"• Total detections: {stats['total_detections']}")
        st.write(f"• Identification rate: {stats['identification_rate']:.1%}")
        st.write(f"• Avg YOLO confidence: {stats['avg_yolo_confidence']:.3f}")
        st.write(f"• Avg LLM confidence: {stats['avg_llm_confidence']:.3f}")

        # Character appearances
        if stats['character_appearances']:
            st.markdown("**Character Appearances:**")
            for char_id, count in stats['character_appearances'].items():
                st.write(f"• {char_id}: {count}")

    def _load_project(self, source, project_id: str, is_upload: bool):
        """Unified method to load project from either upload or file path"""
        try:
            # Handle file source
            if is_upload:
                temp_path = self._save_uploaded_file(source)
                file_path = temp_path
            else:
                file_path = source

            # Load and validate project data
            project_data = self.processor.load_project_results(file_path)

            if not self.processor.validate_data_integrity(project_data):
                st.error("❌ Data validation failed")
                return

            # Reset state for new project
            self._reset_project_state(project_data, project_id)

            st.success(f"✅ Loaded project: {project_data.project_id}")

            # Auto-extract keyframes
            self._auto_extract_keyframes()

            # Cleanup temp file if upload
            if is_upload:
                os.unlink(temp_path)

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

    def _save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to temporary location"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(uploaded_file.getvalue().decode('utf-8'))
            return f.name

    def _reset_project_state(self, project_data: ProjectData, project_id: str):
        """Reset all session state for new project"""
        st.session_state.project_data = project_data
        st.session_state.current_project_id = project_id
        st.session_state.keyframes_extracted = False
        st.session_state.extraction_in_progress = False
        st.session_state.extracted_frames = {}
        st.session_state.timeline_chart_key = 0
        st.session_state.last_slider_value = None
        st.session_state.chart_click_source = False

        # Create keyframe lookup dictionary for efficiency
        st.session_state.keyframe_lookup = {
            kf.timestamp_ms: kf for kf in project_data.keyframes
        }

        # Reset timestamp to first keyframe
        if project_data.keyframes:
            st.session_state.current_timestamp_ms = project_data.keyframes[0].timestamp_ms
            st.session_state.last_slider_value = project_data.keyframes[0].timestamp_ms

    def _auto_extract_keyframes(self):
        """Automatically extract keyframes when project is loaded (updated for multi-project)"""
        project_data = self.project_data  # Use property

        if not project_data or not project_data.video_url:
            if project_data is None:
                st.error("No project data available")
            else:
                st.error("No video URL found in project data")
            return

        # Prevent duplicate extraction
        current_extracted = self._get_current_extraction_status()
        if st.session_state.extraction_in_progress or current_extracted:
            return

        st.session_state.extraction_in_progress = True

        with st.spinner("🎞️ Downloading video and extracting keyframes..."):
            try:
                all_timestamps = [kf.timestamp_ms for kf in project_data.keyframes]
                temp_dir = tempfile.mkdtemp(prefix="keyframes_")

                extracted_frames = self.extractor.extract_keyframes(
                    project_data.video_url,
                    all_timestamps,
                    temp_dir
                )

                # Use new methods to set extraction data
                self._set_current_extracted_frames(extracted_frames)
                self._set_current_extraction_status(True)
                st.session_state.extraction_in_progress = False
                st.success(f"✅ Extracted {len(extracted_frames)} keyframes")

                st.rerun()

            except Exception as e:
                st.error(f"❌ Error extracting keyframes: {e}")
                self._set_current_extraction_status(False)
                st.session_state.extraction_in_progress = False

    def _render_main_content(self):
        """Render the main content area"""
        # Use property instead of direct session state access
        if self.project_data is None:
            if st.session_state.project_tabs:
                st.info("👈 Please select a project from the sidebar tabs")
            else:
                st.info("👈 Please select a project or load a folder from the sidebar to begin")
            return

        project_data = self.project_data

        # Only show navigation if keyframes are extracted
        current_extracted = self._get_current_extraction_status()
        if not current_extracted:
            if st.session_state.extraction_in_progress:
                st.info("🔄 Please wait while keyframes are being extracted...")
            else:
                st.info("🔄 Keyframes will be extracted automatically...")
            return

        # Timeline overview
        st.subheader("📊 Timeline Overview")
        st.caption("💡 Click on any bar or point in the charts below to jump to that keyframe")
        self._render_timeline_histogram(project_data)

        # Keyframe navigation
        if project_data.keyframes:
            self._render_keyframe_navigation(project_data)

        # Character vault
        st.markdown("---")
        self._render_character_vault(project_data.character_vault)

    def _render_timeline_histogram(self, project_data: ProjectData):
        """Render improved timeline visualization with separate plots for better clarity"""

        # Prepare data for visualization
        timestamps = []
        detection_counts = []
        unknown_counts = []
        character_data = {}  # Dict[char_id, List[Dict[timestamp, confidence]]]

        # Get ALL characters from character vault (not just those that appear in keyframes)
        all_characters = set()

        # First, add all characters from the character vault
        for char in project_data.character_vault:
            all_characters.add(char.char_id)

        # Also add any characters that appear in keyframes but might not be in vault
        for kf in project_data.keyframes:
            for crop in kf.crops:
                if crop.pred_char_id and crop.pred_char_id != "Unknown":
                    all_characters.add(crop.pred_char_id)

        # Sort characters for consistent ordering
        all_characters = sorted(all_characters)

        # Initialize character data structures
        for char_id in all_characters:
            character_data[char_id] = []

        # Process keyframes to collect data
        for kf in project_data.keyframes:
            timestamps.append(kf.timestamp_ms)
            detection_counts.append(len(kf.crops))

            # Count unknowns in this keyframe
            unknown_count = sum(1 for crop in kf.crops
                              if not crop.pred_char_id or crop.pred_char_id == "Unknown")
            unknown_counts.append(unknown_count)

            # Collect character confidence data for this keyframe
            for char_id in all_characters:
                # Find all crops for this character in this keyframe
                char_crops = [crop for crop in kf.crops
                             if crop.pred_char_id == char_id]

                if char_crops:
                    # Use highest confidence if multiple detections of same character
                    max_confidence = max(crop.confidence or 0.0 for crop in char_crops)
                    character_data[char_id].append({
                        'timestamp': kf.timestamp_ms,
                        'confidence': max_confidence,
                        'count': len(char_crops)
                    })
                else:
                    # No detection of this character in this keyframe
                    character_data[char_id].append({
                        'timestamp': kf.timestamp_ms,
                        'confidence': 0.0,
                        'count': 0
                    })

        # Get current timestamp for highlighting
        current_timestamp = st.session_state.current_timestamp_ms

        # Calculate number of subplots needed
        num_character_plots = len(all_characters)
        total_plots = 2 + num_character_plots  # Detection count + Unknown count + character plots

        if total_plots == 0:
            st.warning("No data to visualize")
            return

        # Create subplot titles with better formatting
        subplot_titles = [
            "<b>Total YOLO Detections per Keyframe</b>",
            "<b>Unknown/Unidentified Detections per Keyframe</b>"
        ]
        subplot_titles.extend([f"<b>{char_id} - LLM Confidence Scores</b>" for char_id in all_characters])

        # FIXED: Create consistent row heights with more space for titles
        if num_character_plots > 0:
            # Fixed heights: top 2 plots get 15% each, character plots split remaining 70%
            detection_plot_height = 0.15
            unknown_plot_height = 0.15
            character_plot_height = 0.70 / num_character_plots
            row_heights = [detection_plot_height, unknown_plot_height] + [character_plot_height] * num_character_plots
        else:
            # If no characters, just use equal heights for detection plots
            row_heights = [0.5, 0.5]

        # Create subplots with FIXED height ratios and better spacing
        fig = make_subplots(
            rows=total_plots, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.06,  # Increased spacing for better title visibility
            row_heights=row_heights
        )

        # Color scheme
        detection_color = '#2E86C1'  # Blue
        unknown_color = '#E74C3C'   # Red
        character_colors = px.colors.qualitative.Set1[:len(all_characters)]

        # 1. Total Detection Count Plot
        bar_colors = ['gold' if ts == current_timestamp else detection_color for ts in timestamps]
        bar_opacities = [1.0 if ts == current_timestamp else 0.7 for ts in timestamps]

        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=detection_counts,
                name="Total Detections",
                marker={
                    "color": bar_colors,
                    "opacity": bar_opacities,
                    "line": {
                        "color": ['orange' if ts == current_timestamp else 'steelblue' for ts in timestamps],
                        "width": 1
                    }
                },
                hovertemplate="<b>%{x:,} ms</b><br>Total Detections: %{y}<br><i>Click to jump to this keyframe</i><extra></extra>",
                showlegend=False
            ),
            row=1, col=1
        )

        # 2. Unknown Count Plot
        unknown_bar_colors = ['gold' if ts == current_timestamp else unknown_color for ts in timestamps]
        unknown_bar_opacities = [1.0 if ts == current_timestamp else 0.7 for ts in timestamps]

        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=unknown_counts,
                name="Unknown Detections",
                marker={
                    "color": unknown_bar_colors,
                    "opacity": unknown_bar_opacities,
                    "line": {
                        "color": ['orange' if ts == current_timestamp else '#C0392B' for ts in timestamps],
                        "width": 1
                    }
                },
                hovertemplate="<b>%{x:,} ms</b><br>Unknown Detections: %{y}<br><i>Click to jump to this keyframe</i><extra></extra>",
                showlegend=False
            ),
            row=2, col=1
        )

        # 3. Character Confidence Plots
        for i, char_id in enumerate(all_characters):
            char_color = character_colors[i % len(character_colors)]
            char_timestamps = [point['timestamp'] for point in character_data[char_id]]
            char_confidences = [point['confidence'] for point in character_data[char_id]]
            char_counts = [point['count'] for point in character_data[char_id]]

            # Create line plot for confidence scores
            # Only show points where confidence > 0 (character was detected)
            visible_timestamps = []
            visible_confidences = []
            visible_counts = []

            for j, conf in enumerate(char_confidences):
                if conf > 0:  # Only show when character was actually detected
                    visible_timestamps.append(char_timestamps[j])
                    visible_confidences.append(conf)
                    visible_counts.append(char_counts[j])

            # Always add trace for each character (even if no appearances)
            if visible_timestamps:  # Character appears somewhere
                # Highlight current timestamp
                point_colors = []
                point_sizes = []
                for ts in visible_timestamps:
                    if ts == current_timestamp:
                        point_colors.append('gold')
                        point_sizes.append(12)
                    else:
                        point_colors.append(char_color)
                        point_sizes.append(8)

                fig.add_trace(
                    go.Scatter(
                        x=visible_timestamps,
                        y=visible_confidences,
                        mode='markers+lines',
                        name=char_id,
                        line={"color": char_color, "width": 2},
                        marker={
                            "color": point_colors,
                            "size": point_sizes,
                            "line": {"width": 2, "color": 'white'}
                        },
                        hovertemplate="<b>%{x:,} ms</b><br>" +
                                    f"Character: {char_id}<br>" +
                                    "LLM Confidence: %{y:.3f}<br>" +
                                    "Detections: %{customdata}<br>" +
                                    "<i>Click to jump to this keyframe</i><extra></extra>",
                        customdata=visible_counts,
                        showlegend=True
                    ),
                    row=3 + i, col=1
                )
            else:
                # Character never appears - show empty plot with message
                fig.add_annotation(
                    x=0.5, y=0.5,
                    xref=f"x{3+i} domain", yref=f"y{3+i} domain",
                    text=f"No appearances of {char_id}",
                    showarrow=False,
                    font={"color": "gray", "size": 12},
                    row=3 + i, col=1
                )

                # Add empty trace to maintain subplot structure and legend
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode='markers',
                        name=f"{char_id} (No appearances)",
                        marker={"color": char_color},
                        showlegend=True
                    ),
                    row=3 + i, col=1
                )

        # Add vertical line to show current position across all subplots
        for row in range(1, total_plots + 1):
            fig.add_vline(
                x=current_timestamp,
                line_dash="dash",
                line_color="gold",
                line_width=2,
                opacity=0.6,
                row=row, col=1
            )

        # Calculate timeline range for consistent x-axis alignment
        if timestamps:
            timeline_min = min(timestamps)
            timeline_max = max(timestamps)
            # Add 5% padding on each side
            timeline_range = timeline_max - timeline_min
            x_range = [
                timeline_min - (timeline_range * 0.05),
                timeline_max + (timeline_range * 0.05)
            ]
        else:
            x_range = [0, 1000]

        # Update layout with FIXED height calculation and better title spacing
        base_height = 140  # Increased base height for better title visibility
        total_height = base_height * total_plots

        fig.update_layout(
            height=total_height,
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1
            },
            margin={"t": 120, "b": 60, "l": 60, "r": 60},  # Increased top margin for titles
            title={
                "text": "<b>📊 Timeline Analysis - Click any point to navigate</b>",
                "font": {"size": 16},
                "x": 0.5,
                "xanchor": 'center'
            }
        )

        # FIXED: Update x-axes with consistent range and alignment
        for row in range(1, total_plots + 1):
            fig.update_xaxes(
                title_text="<b>Timeline (ms)</b>" if row == total_plots else "",
                range=x_range,  # Fixed range for all subplots
                matches='x',    # Link all x-axes together
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                row=row, col=1
            )

        # Update y-axes with appropriate labels and CONSISTENT ranges
        fig.update_yaxes(
            title_text="<b>Count</b>",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="<b>Count</b>",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            row=2, col=1
        )

        for i, _char_id in enumerate(all_characters):
            fig.update_yaxes(
                title_text="<b>Confidence</b>",
                range=[0, 1.05],  # Fixed range for confidence scores
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                row=3 + i, col=1
            )

        # Update subplot title formatting for better visibility
        for i in fig['layout']['annotations']:
            i['font'] = {"size": 14, "color": 'white'}  # Larger, white titles
            i['bgcolor'] = 'rgba(0,0,0,0.8)'  # Dark background for titles
            i['bordercolor'] = 'white'
            i['borderwidth'] = 1

        # Render the chart with event capture
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key=f"timeline_chart_{st.session_state.timeline_chart_key}"
        )

        # Handle the selection event properly (same as before)
        if event and "selection" in event and event["selection"]["points"]:
            clicked_point = event["selection"]["points"][0]
            clicked_timestamp = int(clicked_point["x"])

            # Validate timestamp and update if different
            if (clicked_timestamp in st.session_state.keyframe_lookup and
                clicked_timestamp != st.session_state.current_timestamp_ms):

                # Mark this as a chart click to prevent slider feedback loop
                st.session_state.chart_click_source = True
                st.session_state.current_timestamp_ms = clicked_timestamp
                st.session_state.last_slider_value = clicked_timestamp
                st.session_state.timeline_chart_key += 1  # Force chart redraw
                st.rerun()

        # Enhanced summary stats
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            total_detections = sum(detection_counts)
            st.metric("Total Detections", total_detections)

        with col2:
            total_unknowns = sum(unknown_counts)
            st.metric("Unknown Detections", total_unknowns)

        with col3:
            total_identified = total_detections - total_unknowns
            st.metric("Identified Characters", total_identified)

        with col4:
            identification_rate = (total_identified / total_detections * 100) if total_detections > 0 else 0
            st.metric("Identification Rate", f"{identification_rate:.1f}%")

        with col5:
            active_characters = len([char for char in all_characters
                                   if any(point['confidence'] > 0 for point in character_data[char])])
            st.metric("Active Characters", f"{active_characters}/{len(all_characters)}")

    def _render_keyframe_navigation(self, project_data: ProjectData):
        """Render keyframe navigation section"""
        st.subheader("🎞️ Keyframe Navigation")

        # Get all timestamps for the slider
        all_timestamps = [kf.timestamp_ms for kf in project_data.keyframes]

        # Create timestamp slider that snaps to actual keyframe timestamps
        new_timestamp = st.select_slider(
            "Navigate through keyframes:",
            options=all_timestamps,
            value=st.session_state.current_timestamp_ms,
            format_func=lambda x: f"{x:,} ms"
        )

        # Check if slider value changed (and it wasn't from a chart click)
        if (new_timestamp != st.session_state.last_slider_value and
            not st.session_state.chart_click_source):

            st.session_state.current_timestamp_ms = new_timestamp
            st.session_state.last_slider_value = new_timestamp
            st.session_state.timeline_chart_key += 1  # Force chart redraw
            st.rerun()

        # Reset chart click flag after processing
        if st.session_state.chart_click_source:
            st.session_state.chart_click_source = False

        # Update last slider value
        st.session_state.last_slider_value = new_timestamp

        # Get current keyframe using efficient lookup
        current_keyframe = st.session_state.keyframe_lookup.get(st.session_state.current_timestamp_ms)

        if current_keyframe is None:
            st.error("Could not find keyframe for selected timestamp")
            return

        # Show current position info
        current_idx = all_timestamps.index(st.session_state.current_timestamp_ms)
        total_frames = len(all_timestamps)
        st.caption(f"Keyframe {current_idx + 1} of {total_frames} • Timeline position: {st.session_state.current_timestamp_ms:,} ms")

        col1, col2 = st.columns([2, 1])

        with col1:
            self._render_keyframe_viewer(current_keyframe)

        with col2:
            self._render_keyframe_details(current_keyframe)

    def _render_keyframe_viewer(self, keyframe: KeyframeData):
        """Render the keyframe image with bounding boxes"""
        st.subheader(f"Frame at {keyframe.timestamp_ms:,}ms ({keyframe.shot_id})")

        # Check if we have extracted frames using new method
        current_frames = self._get_current_extracted_frames()
        if keyframe.timestamp_ms not in current_frames:
            st.warning("⚠️ Keyframe not available.")
            return

        frame_path = current_frames[keyframe.timestamp_ms]

        try:
            image = Image.open(frame_path)

            # Draw bounding boxes if there are detections
            if keyframe.crops:
                annotated_image = self._draw_bounding_boxes(image, keyframe.crops)
                st.image(annotated_image, use_column_width=True)
                st.write(f"**Detections:** {len(keyframe.crops)}")
            else:
                st.image(image, use_column_width=True)
                st.write("**No detections in this frame**")

        except Exception as e:
            st.error(f"Error loading image: {e}")

    def _draw_bounding_boxes(self, image: Image.Image, crops: list[CropData]) -> Image.Image:
        """Draw bounding boxes and labels on the image with detection numbers"""
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        # Try to use a better font, fallback to default
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
            number_font = ImageFont.truetype("Arial.ttf", 20)  # Larger font for numbers
        except OSError:
            font = ImageFont.load_default()
            number_font = ImageFont.load_default()

        # Use consistent detection colors
        colors = self.get_detection_colors()

        img_width, img_height = image.size

        for i, crop in enumerate(crops):
            # Convert normalized bbox to pixel coordinates
            x, y, w, h = crop.bbox_norm
            x1 = int(x * img_width)
            y1 = int(y * img_height)
            x2 = int((x + w) * img_width)
            y2 = int((y + h) * img_height)

            color = colors[i % len(colors)]

            # Draw bounding box with thicker line
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

            # Create main label
            char_id = crop.pred_char_id or "Unknown"
            yolo_conf = crop.face_conf
            llm_conf = crop.confidence or 0.0

            main_label = f"{char_id}\nYOLO: {yolo_conf:.3f}\nLLM: {llm_conf:.3f}"

            # Draw main label background and text
            label_bbox = draw.textbbox((x1, y1 - 80), main_label, font=font)
            draw.rectangle(label_bbox, fill=color, outline=color)
            draw.text((x1, y1 - 80), main_label, fill="white", font=font)

            # Draw detection number in top-left corner of bounding box
            detection_number = f"#{i+1}"
            number_bbox = draw.textbbox((x1, y1), detection_number, font=number_font)

            # Add padding to number background
            number_bg = [
                number_bbox[0] - 5,
                number_bbox[1] - 2,
                number_bbox[2] + 5,
                number_bbox[3] + 2
            ]

            # Draw number background (white circle for visibility)
            draw.rectangle(number_bg, fill="white", outline=color, width=2)
            draw.text((x1, y1), detection_number, fill=color, font=number_font)

        return annotated

    def _render_keyframe_details(self, keyframe: KeyframeData):
        """Render detailed information about the current keyframe with color coding"""
        st.subheader("📋 Detection Details")

        if not keyframe.crops:
            st.write("No detections in this frame")
            return

        # Get consistent colors
        colors = self.get_detection_colors()

        for i, crop in enumerate(keyframe.crops):
            color = colors[i % len(colors)]

            # Convert hex color to RGB for container styling
            color_rgb = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))

            # Create colored container using markdown with custom CSS
            st.markdown(
                f"""
                <div style="
                    border-left: 5px solid {color};
                    background-color: rgba{color_rgb + (0.1,)};
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                ">
                    <strong style="color: {color};">Detection #{i+1}: {crop.pred_char_id or 'Unknown'}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Create columns for organized display
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Crop ID:** `{crop.crop_id}`")
                st.write(f"**Detector:** {crop.detector}")
                st.write(f"**YOLO Confidence:** {crop.face_conf:.3f}")

                # Bounding box info
                x, y, w, h = crop.bbox_norm
                st.write(f"**Bounding Box:** ({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})")

            with col2:
                if crop.pred_char_id:
                    st.write(f"**Predicted Character:** {crop.pred_char_id}")
                    st.write(f"**LLM Confidence:** {crop.confidence or 0.0:.3f}")
                else:
                    st.write("**Predicted Character:** Unknown")
                    st.write("**LLM Confidence:** 0.000")

                # Quality metrics if available
                if crop.quality:
                    st.write("**Quality Metrics:**")
                    for key, value in crop.quality.items():
                        st.write(f"• {key}: {value}")

                # Show reasoning if available
                if crop.reason:
                    st.write("**Reasoning:**")
                    st.write(crop.reason)
                elif not crop.pred_char_id or crop.pred_char_id == "Unknown":
                    st.write("**Reasoning:**")
                    st.write("The image crops were not provided, so no traits could be analyzed.")

                # Add separator between detections
                if i < len(keyframe.crops) - 1:
                    st.markdown("---")

    def _render_character_vault(self, characters: list):
        """Render the character vault section"""
        st.subheader("👥 Character Vault")

        if not characters:
            st.write("No characters in vault")
            return

        # Create columns for character cards
        cols = st.columns(min(len(characters), 3))

        for i, character in enumerate(characters):
            with cols[i % 3]:
                with st.container():
                    st.write(f"**{character.name}**")
                    st.write(f"ID: `{character.char_id}`")

                    # Try to display reference image
                    if character.ref_image:
                        try:
                            st.image(character.ref_image, width=150)
                        except Exception:
                            st.write(f"[Reference Image]({character.ref_image})")

                    # Show key traits
                    if character.traits:
                        with st.expander("View Traits"):
                            if 'core' in character.traits:
                                st.write("**Core traits:**")
                                for trait in character.traits['core']:
                                    st.write(f"• {trait}")

                            if 'age_band' in character.traits:
                                st.write(f"**Age:** {character.traits['age_band']}")

                            if 'skin_tone' in character.traits:
                                st.write(f"**Skin tone:** {character.traits['skin_tone']}")

    def _render_folder_selection(self):
        """Render folder selection for multi-project loading"""
        st.subheader("📁 Load Project Folder")

        # Common folder paths
        common_folders = [
            "data/logs/test_run_20/3-character_identification",
            "data/logs/full_run_11/3-character_identification",
            "data/logs/test_run_1/3-character_identification",
            "data/logs/test_run_7/3-character_identification"
        ]

        # Check which folders exist
        existing_folders = [folder for folder in common_folders if Path(folder).exists()]

        if existing_folders:
            selected_folder = st.selectbox(
                "Select folder containing projects:",
                [""] + existing_folders,
                format_func=lambda x: x if x else "Choose a folder..."
            )

            if selected_folder and st.button("Load All Projects from Folder"):
                self._load_projects_from_folder(selected_folder)

        # Manual folder path input
        manual_folder = st.text_input(
            "Or enter custom folder path:",
            placeholder="data/logs/your_run/3-character_identification"
        )

        if manual_folder and st.button("Load Custom Folder"):
            self._load_projects_from_folder(manual_folder)

    def _render_project_tabs(self):
        """Render project selection tabs"""
        st.markdown("---")
        st.subheader("🎬 Loaded Projects")

        # Create display names for projects
        project_names = []
        for project_id in st.session_state.project_tabs:
            project_data = st.session_state.projects[project_id]
            display_name = f"{project_data.project_id}"
            if len(display_name) > 20:
                display_name = display_name[:17] + "..."
            project_names.append(display_name)

        # Find current selection index
        current_idx = 0
        if st.session_state.active_project_id in st.session_state.project_tabs:
            current_idx = st.session_state.project_tabs.index(st.session_state.active_project_id)

        # Project selector
        selected_idx = st.radio(
            "Select project:",
            range(len(st.session_state.project_tabs)),
            index=current_idx,
            format_func=lambda x: project_names[x],
            key="project_selector"
        )

        # Switch project if selection changed
        new_active_id = st.session_state.project_tabs[selected_idx]
        if new_active_id != st.session_state.active_project_id:
            self._switch_to_project(new_active_id)

        # Show project count
        st.caption(f"📊 {len(st.session_state.project_tabs)} projects loaded")

    def _load_projects_from_folder(self, folder_path: str):
        """Load all project JSON files from a folder"""
        folder = Path(folder_path)

        if not folder.exists():
            st.error(f"❌ Folder not found: {folder_path}")
            return

        # Find all JSON files
        json_files = list(folder.glob("*.json"))

        if not json_files:
            st.error(f"❌ No JSON files found in: {folder_path}")
            return

        # Load projects with progress bar
        loaded_projects = {}
        failed_projects = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, json_file in enumerate(json_files):
            try:
                status_text.text(f"Loading {json_file.name}...")
                progress_bar.progress((i + 1) / len(json_files))

                # Load project data
                project_data = self.processor.load_project_results(str(json_file))

                if self.processor.validate_data_integrity(project_data):
                    loaded_projects[project_data.project_id] = project_data
                else:
                    failed_projects.append(json_file.name)

            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
                failed_projects.append(json_file.name)

        # Update session state
        if loaded_projects:
            st.session_state.projects = loaded_projects
            st.session_state.project_tabs = list(loaded_projects.keys())
            st.session_state.folder_loaded = True

            # Set first project as active
            if st.session_state.project_tabs:
                first_project_id = st.session_state.project_tabs[0]
                self._switch_to_project(first_project_id)

            # Clear single project mode
            st.session_state.project_data = None
            st.session_state.current_project_id = None

            success_msg = f"✅ Loaded {len(loaded_projects)} projects from {folder_path}"
            if failed_projects:
                success_msg += f"\n⚠️ Failed to load: {', '.join(failed_projects)}"
            st.success(success_msg)

            # Auto-extract keyframes for all projects
            self._auto_extract_all_keyframes()

        else:
            st.error(f"❌ No valid projects found in {folder_path}")

        # Cleanup progress indicators
        progress_bar.empty()
        status_text.empty()

        # Force rerun to show new projects
        st.rerun()

    def _switch_to_project(self, project_id: str):
        """Switch to a different project"""
        if project_id not in st.session_state.projects:
            st.error(f"Project {project_id} not found")
            return

        # Update active project
        st.session_state.active_project_id = project_id
        project_data = st.session_state.projects[project_id]

        # Reset navigation state for new project
        st.session_state.timeline_chart_key += 1
        st.session_state.last_slider_value = None
        st.session_state.chart_click_source = False

        # Create keyframe lookup dictionary for efficiency
        st.session_state.keyframe_lookup = {
            kf.timestamp_ms: kf for kf in project_data.keyframes
        }

        # Reset timestamp to first keyframe
        if project_data.keyframes:
            st.session_state.current_timestamp_ms = project_data.keyframes[0].timestamp_ms
            st.session_state.last_slider_value = project_data.keyframes[0].timestamp_ms

        logger.info(f"Switched to project: {project_id}")

    def _auto_extract_all_keyframes(self):
        """Auto-extract keyframes for all loaded projects"""
        if not st.session_state.projects:
            return

        st.info("🎞️ Starting keyframe extraction for all projects...")

        for project_id, project_data in st.session_state.projects.items():
            if not project_data.video_url:
                logger.warning(f"No video URL for project {project_id}")
                continue

            # Skip if already extracted
            if st.session_state.projects_extraction_status.get(project_id, False):
                continue

            try:
                with st.spinner(f"Extracting keyframes for {project_id}..."):
                    all_timestamps = [kf.timestamp_ms for kf in project_data.keyframes]
                    temp_dir = tempfile.mkdtemp(prefix=f"keyframes_{project_id}_")

                    extracted_frames = self.extractor.extract_keyframes(
                        project_data.video_url,
                        all_timestamps,
                        temp_dir
                    )

                    st.session_state.projects_extracted_frames[project_id] = extracted_frames
                    st.session_state.projects_extraction_status[project_id] = True

                    logger.info(f"Extracted {len(extracted_frames)} keyframes for {project_id}")

            except Exception as e:
                logger.error(f"Failed to extract keyframes for {project_id}: {e}")
                st.session_state.projects_extraction_status[project_id] = False

        st.success("✅ Keyframe extraction completed for all projects")


def main():
    """Main entry point"""
    viewer = StreamlitViewer()
    viewer.run()


if __name__ == "__main__":
    main()
