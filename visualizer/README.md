# Pipeline Results Viewer

Interactive Streamlit-based viewer for character identification pipeline results.

## Features

- 🎞️ **Keyframe Navigation**: Slider-based timeline navigation through all keyframes
- 🎯 **Bounding Box Visualization**: YOLO detections overlaid on keyframes with confidence scores
- 🧠 **Character Identification**: Shows predicted characters with LLM confidence and reasoning
- 👥 **Character Vault**: Display reference images and traits for all characters
- 📊 **Timeline Histogram**: Visual overview of detection patterns and character matches
- 🎨 **Color-Coded Detections**: Matching colors between bounding boxes and detection details
- 📂 **File Browser**: Easy loading from common result locations

## Usage

### Quick Start
```bash
# From the visualizer directory
python run_viewer.py
```

### Manual Launch
```bash
# From the project root
streamlit run visualizer/streamlit_viewer.py
```

### Loading Results

1. **Upload JSON**: Use the file uploader in the sidebar
2. **Browse Local**: Select from detected result files in `data/logs/`
3. **Automatic Processing**: Video download and keyframe extraction happens automatically

## File Structure

```
visualizer/
├── streamlit_viewer.py     # Main Streamlit application
├── process_results.py      # Data processing and validation
├── extract_keyframes.py    # Video keyframe extraction
├── run_viewer.py          # Simple launcher script
└── README.md              # This file
```

## Input Format

Expects character identification results JSON files with structure:
```json
{
  "character_vault": [...],
  "video_url": "...",
  "shots": [
    {
      "shot_id": "...",
      "keyframes_ms": [...],
      "crops": {
        "timestamp": [
          {
            "crop_id": "...",
            "bbox_norm": [x, y, w, h],
            "face_conf": 0.95,
            "pred_char_id": "...",
            "confidence": 0.8,
            "reason": "..."
          }
        ]
      }
    }
  ]
}
```

## Navigation

- **Timeline Slider**: Navigate through keyframes chronologically
- **Detection Details**: Expand to see YOLO confidence, LLM reasoning, and quality metrics
- **Character Cards**: View reference images and trait information
- **Statistics Panel**: Overview of processing results and detection rates
```

Let me also create a simple test script to verify everything works:

```python:visualizer/test_viewer.py
"""
Test script for the pipeline results viewer
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from visualizer.process_results import PipelineResultsProcessor


def test_data_processing():
    """Test the data processing functionality"""
    print("🧪 Testing Pipeline Results Processor...")

    processor = PipelineResultsProcessor()

    # Test with sample file
    sample_files = [
        "data/logs/test_run_20/3-character_identification/PROJadRsvgvJffe8YQEf.json",
        "data/logs/test_run_20/3-character_identification/PROJvg5wy5R8zFQ9kh6K.json"
    ]

    for sample_file in sample_files:
        if Path(sample_file).exists():
            print(f"\n📂 Testing with: {sample_file}")

            try:
                # Load project data
                project_data = processor.load_project_results(sample_file)
                print(f"✅ Loaded project: {project_data.project_id}")

                # Validate data
                is_valid = processor.validate_data_integrity(project_data)
                print(f"✅ Data validation: {'PASSED' if is_valid else 'FAILED'}")

                # Get stats
                stats = processor.get_processing_stats(project_data)
                print(f"📊 Statistics:")
                print(f"  • Total keyframes: {stats['total_keyframes']}")
                print(f"  • Keyframes with detections: {stats['keyframes_with_detections']}")
                print(f"  • Total detections: {stats['total_detections']}")
                print(f"  • Identification rate: {stats['identification_rate']:.1%}")
                print(f"  • Avg YOLO confidence: {stats['avg_yolo_confidence']:.3f}")
                print(f"  • Timeline duration: {stats['timeline_duration_ms']:,} ms")

                # Show character vault
                print(f"👥 Characters in vault: {len(project_data.character_vault)}")
                for char in project_data.character_vault:
                    print(f"  • {char.name} ({char.char_id})")

                # Show sample keyframes
                print(f"🎞️ Sample keyframes:")
                for i, keyframe in enumerate(project_data.keyframes[:3]):
                    print(f"  • {keyframe.timestamp_ms}ms ({keyframe.shot_id}): {len(keyframe.crops)} detections")

            except Exception as e:
                print(f"❌ Error processing {sample_file}: {e}")
        else:
            print(f"⚠️ Sample file not found: {sample_file}")

    print("\n✅ Data processing test complete!")


def test_viewer_components():
    """Test viewer component imports"""
    print("\n🧪 Testing Viewer Components...")

    try:
        from visualizer.streamlit_viewer import StreamlitViewer
        print("✅ StreamlitViewer import successful")

        from visualizer.extract_keyframes import KeyframeExtractor
        print("✅ KeyframeExtractor import successful")

        # Test extractor initialization
        extractor = KeyframeExtractor()
        print("✅ KeyframeExtractor initialization successful")

    except Exception as e:
        print(f"❌ Component test failed: {e}")

    print("✅ Component test complete!")


def main():
    """Run all tests"""
    print("🎬 Pipeline Results Viewer - Test Suite")
    print("=" * 50)

    test_data_processing()
    test_viewer_components()

    print("\n" + "=" * 50)
    print("🎉 All tests complete!")
    print("\n💡 To launch the viewer:")
    print("   python visualizer/run_viewer.py")
    print("   or")
    print("   streamlit run visualizer/streamlit_viewer.py")


if __name__ == "__main__":
    main()
```

Perfect! Now let's test the implementation:
```

```
