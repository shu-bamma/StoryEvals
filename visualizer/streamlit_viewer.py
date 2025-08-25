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
from PIL import Image, ImageDraw, ImageFont
from plotly.subplots import make_subplots

from .extract_keyframes import KeyframeExtractor
from .process_results import (
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
            "project_data": None,
            "extracted_frames": {},
            "current_timestamp_ms": 0,
            "keyframes_extracted": False,
            "current_project_id": None,
            "extraction_in_progress": False,
            "keyframe_lookup": {},  # For efficient keyframe access
            "timeline_chart_key": 0,  # For forcing chart updates
            "last_slider_value": None,  # Track slider changes
            "chart_click_source": False,  # Track if change came from chart click
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

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
            "#32CD32",  # Lime Green
        ]

    def run(self):
        """Main Streamlit app"""
        st.set_page_config(
            page_title="Pipeline Results Viewer", page_icon="🎬", layout="wide"
        )

        st.title("🎬 Character Identification Pipeline Viewer")
        st.markdown("---")

        # Sidebar for file selection and controls
        with st.sidebar:
            self._render_sidebar()

        # Main content
        if st.session_state.project_data is not None:
            self._render_main_content()
        else:
            st.info("👈 Please select a project results file from the sidebar to begin")

    def _render_sidebar(self):
        """Render the sidebar with file selection and controls"""
        st.header("📂 Project Selection")

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Character Identification Results JSON",
            type=["json"],
            help="Select a JSON file from the pipeline results",
        )

        if uploaded_file is not None:
            file_content = uploaded_file.getvalue().decode("utf-8")
            file_hash = hash(file_content)
            project_id = f"upload_{file_hash}"

            if st.session_state.current_project_id != project_id:
                self._load_project(uploaded_file, project_id, is_upload=True)

        # Alternative: Browse local files
        st.subheader("Or browse local files:")

        # Look for results in common locations
        common_paths = ["data/logs", "data/logs/test_run_20/3-character_identification"]

        available_files = self._get_available_files(common_paths)

        if available_files:
            selected_file = st.selectbox(
                "Select from available files:",
                [""] + available_files,
                format_func=lambda x: Path(x).name if x else "Choose a file...",
            )

            if selected_file and st.session_state.current_project_id != selected_file:
                self._load_project(selected_file, selected_file, is_upload=False)
        else:
            st.info("No result files found in common locations")

        # Project info
        if st.session_state.project_data is not None:
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
        project_data = st.session_state.project_data

        st.markdown("---")
        st.subheader("📊 Project Info")

        st.write(f"**Project ID:** `{project_data.project_id}`")
        st.write(f"**Total Keyframes:** {len(project_data.keyframes)}")
        st.write(f"**Characters:** {len(project_data.character_vault)}")
        st.write(f"**Duration:** {project_data.get_timeline_duration_ms():,} ms")

        # Show extraction status
        if st.session_state.keyframes_extracted:
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
        if stats["character_appearances"]:
            st.markdown("**Character Appearances:**")
            for char_id, count in stats["character_appearances"].items():
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(uploaded_file.getvalue().decode("utf-8"))
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
            st.session_state.current_timestamp_ms = project_data.keyframes[
                0
            ].timestamp_ms
            st.session_state.last_slider_value = project_data.keyframes[0].timestamp_ms

    def _auto_extract_keyframes(self):
        """Automatically extract keyframes when project is loaded"""
        project_data = st.session_state.project_data

        if not project_data.video_url:
            st.error("No video URL found in project data")
            return

        # Prevent duplicate extraction
        if (
            st.session_state.extraction_in_progress
            or st.session_state.keyframes_extracted
        ):
            return

        st.session_state.extraction_in_progress = True

        with st.spinner("🎞️ Downloading video and extracting keyframes..."):
            try:
                all_timestamps = [kf.timestamp_ms for kf in project_data.keyframes]
                temp_dir = tempfile.mkdtemp(prefix="keyframes_")

                extracted_frames = self.extractor.extract_keyframes(
                    project_data.video_url, all_timestamps, temp_dir
                )

                st.session_state.extracted_frames = extracted_frames
                st.session_state.keyframes_extracted = True
                st.session_state.extraction_in_progress = False
                st.success(f"✅ Extracted {len(extracted_frames)} keyframes")

                st.rerun()

            except Exception as e:
                st.error(f"❌ Error extracting keyframes: {e}")
                st.session_state.keyframes_extracted = False
                st.session_state.extraction_in_progress = False

    def _render_main_content(self):
        """Render the main content area"""
        project_data = st.session_state.project_data

        # Only show navigation if keyframes are extracted
        if not st.session_state.keyframes_extracted:
            if st.session_state.extraction_in_progress:
                st.info("🔄 Please wait while keyframes are being extracted...")
            else:
                st.info("🔄 Keyframes will be extracted automatically...")
            return

        # Timeline overview
        st.subheader("📊 Timeline Overview")
        st.caption(
            "💡 Click on any bar or point in the charts below to jump to that keyframe"
        )
        self._render_timeline_histogram(project_data)

        # Keyframe navigation
        if project_data.keyframes:
            self._render_keyframe_navigation(project_data)

        # Character vault
        st.markdown("---")
        self._render_character_vault(project_data.character_vault)

    def _render_timeline_histogram(self, project_data: ProjectData):
        """Render timeline histogram showing detections and character matches"""

        # Prepare data for visualization
        timestamps = []
        detection_counts = []
        character_points = []

        # Get unique characters and assign colors
        all_characters = set()
        for kf in project_data.keyframes:
            for crop in kf.crops:
                if crop.pred_char_id and crop.pred_char_id != "Unknown":
                    all_characters.add(crop.pred_char_id)

        character_colors = px.colors.qualitative.Set3[: len(all_characters)]
        char_color_map = dict(zip(all_characters, character_colors, strict=False))
        char_color_map["Unknown"] = "#808080"  # Gray for unknown

        # Process keyframes
        for kf in project_data.keyframes:
            timestamps.append(kf.timestamp_ms)
            detection_counts.append(len(kf.crops))

            # Add character identification points
            for crop in kf.crops:
                char_id = crop.pred_char_id or "Unknown"
                character_points.append(
                    {
                        "timestamp": kf.timestamp_ms,
                        "character": char_id,
                        "yolo_conf": crop.face_conf,
                        "llm_conf": crop.confidence or 0.0,
                        "color": char_color_map.get(char_id, "#808080"),
                    }
                )

        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "YOLO Detection Count per Keyframe",
                "Character Identification Timeline",
            ),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.6],
        )

        # Get current timestamp for highlighting
        current_timestamp = st.session_state.current_timestamp_ms

        # Add histogram for detection counts with highlighting
        bar_colors = [
            "gold" if ts == current_timestamp else "lightblue" for ts in timestamps
        ]
        bar_opacities = [1.0 if ts == current_timestamp else 0.7 for ts in timestamps]

        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=detection_counts,
                name="YOLO Detections",
                marker={
                    "color": bar_colors,
                    "opacity": bar_opacities,
                    "line": {
                        "color": [
                            "orange" if ts == current_timestamp else "steelblue"
                            for ts in timestamps
                        ],
                        "width": 2,
                    },
                },
                hovertemplate="<b>%{x:,} ms</b><br>Detections: %{y}<br><i>Click to jump to this keyframe</i><extra></extra>",
                customdata=timestamps,  # Store timestamps for click detection
            ),
            row=1,
            col=1,
        )

        # Add character identification points with highlighting
        for char_id in all_characters:
            char_points = [p for p in character_points if p["character"] == char_id]
            if char_points:
                # Separate current timestamp points for highlighting
                regular_points = [
                    p for p in char_points if p["timestamp"] != current_timestamp
                ]
                current_points = [
                    p for p in char_points if p["timestamp"] == current_timestamp
                ]

                # Add regular points
                if regular_points:
                    fig.add_trace(
                        go.Scatter(
                            x=[p["timestamp"] for p in regular_points],
                            y=[1] * len(regular_points),
                            mode="markers",
                            name=f"{char_id}",
                            marker={
                                "color": char_color_map[char_id],
                                "size": 8,
                                "line": {"width": 1, "color": "white"},
                            },
                            hovertemplate="<b>%{x:,} ms</b><br>"
                            + f"Character: {char_id}<br>"
                            + "YOLO: %{customdata[0]:.3f}<br>"
                            + "LLM: %{customdata[1]:.3f}<br>"
                            + "<i>Click to jump to this keyframe</i><extra></extra>",
                            customdata=[
                                [p["yolo_conf"], p["llm_conf"]] for p in regular_points
                            ],
                            showlegend=True,
                        ),
                        row=2,
                        col=1,
                    )

                # Add highlighted current points
                if current_points:
                    fig.add_trace(
                        go.Scatter(
                            x=[p["timestamp"] for p in current_points],
                            y=[1] * len(current_points),
                            mode="markers",
                            name=f"{char_id} (Current)",
                            marker={
                                "color": char_color_map[char_id],
                                "size": 15,  # Larger size
                                "line": {"width": 3, "color": "gold"},  # Gold border
                                "symbol": "star",  # Star shape for highlighting
                            },
                            hovertemplate="<b>%{x:,} ms</b><br>"
                            + f"Character: {char_id} (CURRENT)<br>"
                            + "YOLO: %{customdata[0]:.3f}<br>"
                            + "LLM: %{customdata[1]:.3f}<extra></extra>",
                            customdata=[
                                [p["yolo_conf"], p["llm_conf"]] for p in current_points
                            ],
                            showlegend=False,  # Don't clutter legend
                        ),
                        row=2,
                        col=1,
                    )

        # Add Unknown detections with highlighting
        unknown_points = [p for p in character_points if p["character"] == "Unknown"]
        if unknown_points:
            regular_unknown = [
                p for p in unknown_points if p["timestamp"] != current_timestamp
            ]
            current_unknown = [
                p for p in unknown_points if p["timestamp"] == current_timestamp
            ]

            # Regular unknown points
            if regular_unknown:
                fig.add_trace(
                    go.Scatter(
                        x=[p["timestamp"] for p in regular_unknown],
                        y=[0.5] * len(regular_unknown),
                        mode="markers",
                        name="Unknown",
                        marker={
                            "color": "#808080",
                            "size": 6,
                            "symbol": "x",
                            "line": {"width": 1, "color": "white"},
                        },
                        hovertemplate="<b>%{x:,} ms</b><br>"
                        + "Character: Unknown<br>"
                        + "YOLO: %{customdata[0]:.3f}<br>"
                        + "<i>Click to jump to this keyframe</i><extra></extra>",
                        customdata=[[p["yolo_conf"], 0] for p in regular_unknown],
                    ),
                    row=2,
                    col=1,
                )

            # Highlighted current unknown points
            if current_unknown:
                fig.add_trace(
                    go.Scatter(
                        x=[p["timestamp"] for p in current_unknown],
                        y=[0.5] * len(current_unknown),
                        mode="markers",
                        name="Unknown (Current)",
                        marker={
                            "color": "#808080",
                            "size": 12,  # Larger size
                            "symbol": "x",
                            "line": {"width": 3, "color": "gold"},  # Gold border
                        },
                        hovertemplate="<b>%{x:,} ms</b><br>"
                        + "Character: Unknown (CURRENT)<br>"
                        + "YOLO: %{customdata[0]:.3f}<extra></extra>",
                        customdata=[[p["yolo_conf"], 0] for p in current_unknown],
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

        # Add vertical line to show current position
        fig.add_vline(
            x=current_timestamp,
            line_dash="dash",
            line_color="gold",
            line_width=2,
            opacity=0.8,
        )

        # Update layout for better click detection
        fig.update_layout(
            height=500,
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
            margin={"t": 80},
        )

        # Update x-axes
        fig.update_xaxes(title_text="Timeline (ms)", row=1, col=1)
        fig.update_xaxes(title_text="Timeline (ms)", row=2, col=1)

        # Update y-axes
        fig.update_yaxes(title_text="Detection Count", row=1, col=1)
        fig.update_yaxes(
            title_text="Characters",
            showticklabels=False,
            range=[-0.2, 1.5],
            row=2,
            col=1,
        )

        # THE CORRECT WAY: Use st.plotly_chart with event capture
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key=f"timeline_chart_{st.session_state.timeline_chart_key}",
        )

        # Handle the selection event properly
        if event and "selection" in event and event["selection"]["points"]:
            clicked_point = event["selection"]["points"][0]
            clicked_timestamp = int(clicked_point["x"])

            # Validate timestamp and update if different
            if (
                clicked_timestamp in st.session_state.keyframe_lookup
                and clicked_timestamp != st.session_state.current_timestamp_ms
            ):
                # Mark this as a chart click to prevent slider feedback loop
                st.session_state.chart_click_source = True
                st.session_state.current_timestamp_ms = clicked_timestamp
                st.session_state.last_slider_value = clicked_timestamp
                st.session_state.timeline_chart_key += 1  # Force chart redraw
                st.rerun()

        # Add summary stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_detections = sum(detection_counts)
            st.metric("Total YOLO Detections", total_detections)

        with col2:
            identified_chars = len(
                [p for p in character_points if p["character"] != "Unknown"]
            )
            st.metric("Character Identifications", identified_chars)

        with col3:
            avg_detections = (
                sum(detection_counts) / len(detection_counts) if detection_counts else 0
            )
            st.metric("Avg Detections/Frame", f"{avg_detections:.1f}")

        with col4:
            keyframes_with_chars = len(
                {
                    p["timestamp"]
                    for p in character_points
                    if p["character"] != "Unknown"
                }
            )
            st.metric("Keyframes with Characters", keyframes_with_chars)

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
            format_func=lambda x: f"{x:,} ms",
        )

        # Check if slider value changed (and it wasn't from a chart click)
        if (
            new_timestamp != st.session_state.last_slider_value
            and not st.session_state.chart_click_source
        ):
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
        current_keyframe = st.session_state.keyframe_lookup.get(
            st.session_state.current_timestamp_ms
        )

        if current_keyframe is None:
            st.error("Could not find keyframe for selected timestamp")
            return

        # Show current position info
        current_idx = all_timestamps.index(st.session_state.current_timestamp_ms)
        total_frames = len(all_timestamps)
        st.caption(
            f"Keyframe {current_idx + 1} of {total_frames} • Timeline position: {st.session_state.current_timestamp_ms:,} ms"
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            self._render_keyframe_viewer(current_keyframe)

        with col2:
            self._render_keyframe_details(current_keyframe)

    def _render_keyframe_viewer(self, keyframe: KeyframeData):
        """Render the keyframe image with bounding boxes"""
        st.subheader(f"Frame at {keyframe.timestamp_ms:,}ms ({keyframe.shot_id})")

        # Check if we have extracted frames
        if keyframe.timestamp_ms not in st.session_state.extracted_frames:
            st.warning("⚠️ Keyframe not available.")
            return

        frame_path = st.session_state.extracted_frames[keyframe.timestamp_ms]

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

    def _draw_bounding_boxes(
        self, image: Image.Image, crops: list[CropData]
    ) -> Image.Image:
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
                number_bbox[3] + 2,
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
            color_rgb = tuple(int(color[j : j + 2], 16) for j in (1, 3, 5))

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
                unsafe_allow_html=True,
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
                    st.write(
                        "The image crops were not provided, so no traits could be analyzed."
                    )

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
                            if "core" in character.traits:
                                st.write("**Core traits:**")
                                for trait in character.traits["core"]:
                                    st.write(f"• {trait}")

                            if "age_band" in character.traits:
                                st.write(f"**Age:** {character.traits['age_band']}")

                            if "skin_tone" in character.traits:
                                st.write(
                                    f"**Skin tone:** {character.traits['skin_tone']}"
                                )


def main():
    """Main entry point"""
    viewer = StreamlitViewer()
    viewer.run()


if __name__ == "__main__":
    main()
