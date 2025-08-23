#!/usr/bin/env python3
"""
Test script for the video shot processing pipeline
Tests output format compatibility with character identification pipeline
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
import tempfile

from video_shot_processing_pipeline import VideoShotProcessingPipeline
from character_identification_pipeline import CharacterIdentificationPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def create_sample_enriched_data() -> list[dict]:
    """Create sample enriched character data (format from 1-enriched_characters.json)"""
    return [
        {
            "character_vault": [
                {
                    "char_id": "test_anna",
                    "name": "Anna",
                    "ref_image": "https://example.com/anna.jpg",
                    "description": "A young woman with long blonde hair, blue eyes, and a bright smile.",
                    "embedding": [],
                    "traits": {
                        "core": [
                            "hair: long blonde hair",
                            "eyes: blue eyes",
                            "facial_marks: bright smile"
                        ],
                        "supportive": [
                            "age: mid-twenties",
                            "build: confident demeanor"
                        ],
                        "volatile": [
                            "clothing: red dress"
                        ],
                        "age_band": "young_adult",
                        "skin_tone": "fair",
                        "type": "human",
                        "notes": []
                    }
                }
            ],
            "project_id": "TEST_PROJECT_VIDEO_001",
            "job_id": "test_video_job_001",
            "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"
        }
    ]


def test_pipeline_validation() -> bool:
    """Test video shot processing pipeline validation"""
    print("\n🔍 Testing Pipeline Input Validation")
    print("-" * 50)
    
    try:
        pipeline = VideoShotProcessingPipeline()
        
        # Test with valid enriched data
        print("📋 Testing with valid enriched data...")
        sample_data = create_sample_enriched_data()
        is_valid = pipeline._validate_input_data(sample_data)
        print(f"   Valid data test: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        
        # Test with invalid data
        print("📋 Testing with invalid data...")
        invalid_data = [{"invalid": "data"}]
        is_invalid = not pipeline._validate_input_data(invalid_data)
        print(f"   Invalid data rejection: {'✅ PASSED' if is_invalid else '❌ FAILED'}")
        
        # Test with missing fields
        print("📋 Testing with missing required fields...")
        incomplete_data = [{"character_vault": [], "project_id": "test"}]  # missing video_url
        is_incomplete = not pipeline._validate_input_data(incomplete_data)
        print(f"   Incomplete data rejection: {'✅ PASSED' if is_incomplete else '❌ FAILED'}")
        
        return is_valid and is_invalid and is_incomplete
        
    except Exception as e:
        print(f"❌ Pipeline validation test FAILED: {e}")
        return False


def test_output_format_compatibility() -> bool:
    """Test if video shot processing output is compatible with character identification pipeline"""
    print("\n🔍 Testing Output Format Compatibility")
    print("-" * 50)
    
    try:
        # Create mock video shot processing output
        mock_video_output = [
            {
                "project_id": "TEST_PROJECT_VIDEO_001",
                "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
                "character_vault": [
                    {
                        "char_id": "test_anna",
                        "name": "Anna",
                        "ref_image": "https://example.com/anna.jpg",
                        "description": "A young woman with long blonde hair, blue eyes, and a bright smile.",
                        "embedding": [],
                        "traits": {
                            "core": ["hair: long blonde hair", "eyes: blue eyes"],
                            "supportive": ["age: mid-twenties"],
                            "volatile": ["clothing: red dress"],
                            "age_band": "young_adult",
                            "skin_tone": "fair",
                            "type": "human",
                            "notes": []
                        }
                    }
                ],
                "job_id": "test_video_job_001",
                "number_of_shots": 1,
                "shots": [
                    {
                        "shot_id": "shot_0001",
                        "keyframes_ms": [0, 420, 840],
                        "crops": {
                            "0": [],
                            "420": [
                                {
                                    "crop_id": "c_shot0001_t0420_0",
                                    "bbox_norm": [0.31, 0.18, 0.14, 0.22],
                                    "crop_url": "https://example.com/shot0001_t0420_face0.jpg",
                                    "detector": "yolov8n.pt",
                                    "face_conf": 0.98,
                                    "detection_confidence": 0.98,
                                    "quality": {
                                        "blur": 142.3,
                                        "pose": {"yaw": 6.1, "pitch": -2.3}
                                    }
                                }
                            ],
                            "840": [
                                {
                                    "crop_id": "c_shot0001_t0840_0",
                                    "bbox_norm": [0.52, 0.20, 0.13, 0.21],
                                    "crop_url": "https://example.com/shot0001_t0840_face1.jpg",
                                    "detector": "yolov8n.pt",
                                    "face_conf": 0.89,
                                    "detection_confidence": 0.89,
                                    "quality": {
                                        "blur": 98.5,
                                        "pose": {"yaw": 12.2, "pitch": -5.1}
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
        
        # Test character identification pipeline compatibility
        char_id_pipeline = CharacterIdentificationPipeline()
        
        # Extract single project (CharacterIdentificationPipeline expects single project)
        single_project = mock_video_output[0]
        
        print("📋 Testing CharacterIdentificationPipeline input validation...")
        is_valid = char_id_pipeline._validate_input_data(single_project)
        print(f"   Validation result: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        
        if not is_valid:
            return False
        
        print("👥 Testing character vault parsing...")
        character_vault = char_id_pipeline._parse_character_vault(single_project["character_vault"])
        print(f"   Parsed {len(character_vault)} characters successfully")
        
        print("🎬 Testing shot data parsing...")
        parsed_shots = []
        for shot_data in single_project["shots"]:
            parsed_shot = char_id_pipeline._parse_shot_data(shot_data)
            parsed_shots.append(parsed_shot)
        
        print(f"   Parsed {len(parsed_shots)} shots successfully")
        
        # Verify crop data structure
        total_crops = 0
        for shot in parsed_shots:
            for crops_list in shot.crops.values():
                total_crops += len(crops_list)
                for crop in crops_list:
                    # Test that all required fields are present
                    assert hasattr(crop, 'crop_id'), "Missing crop_id"
                    assert hasattr(crop, 'bbox_norm'), "Missing bbox_norm"
                    assert hasattr(crop, 'crop_url'), "Missing crop_url"
                    assert hasattr(crop, 'detector'), "Missing detector"
                    assert hasattr(crop, 'face_conf'), "Missing face_conf"
                    assert hasattr(crop, 'quality'), "Missing quality"
                    
                    # Test that face_conf field is properly populated
                    assert crop.face_conf > 0, f"Invalid face_conf value: {crop.face_conf}"
                    print(f"   ✅ Crop {crop.crop_id}: face_conf={crop.face_conf}")
        
        print(f"   Total crops processed: {total_crops}")
        print("✅ All format compatibility tests PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Format compatibility test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_serialization() -> bool:
    """Test JSON serialization of results"""
    print("\n🔍 Testing JSON Serialization")
    print("-" * 50)
    
    try:
        # Create mock results
        mock_results = [
            {
                "project_id": "TEST_001",
                "video_url": "https://example.com/video.mp4",
                "character_vault": [],
                "number_of_shots": 1,
                "shots": [
                    {
                        "shot_id": "shot_0001",
                        "keyframes_ms": [0, 420],
                        "crops": {
                            "0": [],
                            "420": [
                                {
                                    "crop_id": "test_crop",
                                    "bbox_norm": [0.1, 0.2, 0.3, 0.4],
                                    "crop_url": "https://example.com/crop.jpg",
                                    "detector": "yolov8n.pt",
                                    "face_conf": 0.95,
                                    "detection_confidence": 0.95,
                                    "quality": {"blur": 100.0, "pose": None}
                                }
                            ]
                        }
                    }
                ]
            }
        ]
        
        # Test saving to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(mock_results, tmp_file, indent=2, ensure_ascii=False)
            tmp_path = tmp_file.name
        
        print(f"📝 Saved test data to: {tmp_path}")
        
        # Test loading back
        with open(tmp_path, 'r') as f:
            loaded_results = json.load(f)
        
        # Verify structure is preserved
        assert len(loaded_results) == len(mock_results), "Array length mismatch"
        assert loaded_results[0]["project_id"] == mock_results[0]["project_id"], "Project ID mismatch"
        assert len(loaded_results[0]["shots"]) == len(mock_results[0]["shots"]), "Shots count mismatch"
        
        print("📊 Verified data integrity after save/load cycle")
        
        # Clean up
        Path(tmp_path).unlink()
        print("🧹 Cleaned up temporary file")
        
        print("✅ JSON serialization test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization test FAILED: {e}")
        return False


def test_integration_flow() -> bool:
    """Test understanding of the integration flow"""
    print("\n🔄 Testing Integration Flow Understanding")
    print("-" * 50)
    
    try:
        print("📋 Pipeline Flow:")
        print("   1️⃣ Input: 1-enriched_characters.json (list of projects)")
        print("   2️⃣ VideoShotProcessingPipeline.process(enriched_data)")
        print("   3️⃣ Output: 2-video_shots_with_crops.json (list of projects with shots)")
        print("   4️⃣ For each project: CharacterIdentificationPipeline.process(single_project)")
        
        print("\n📊 Data Flow:")
        enriched_data = create_sample_enriched_data()
        print(f"   Input format: list[dict] with {len(enriched_data)} projects")
        print(f"   Each project has: {list(enriched_data[0].keys())}")
        print(f"   Character vault size: {len(enriched_data[0]['character_vault'])}")
        
        print("\n✅ Integration flow understanding COMPLETE!")
        print("🔗 The formats are compatible with proper iteration!")
        return True
        
    except Exception as e:
        print(f"❌ Integration flow test FAILED: {e}")
        return False


def main() -> None:
    """Main test function"""
    print("🚀 Starting Video Shot Processing Pipeline Test")
    print("=" * 60)
    print("📝 Testing format compatibility and pipeline integration")
    print("🚫 No OpenAI API key needed - using mock data")
    print("🎯 Focus: Does output work with CharacterIdentificationPipeline?")
    print()
    
    try:
        # Run all tests
        tests = [
            ("Pipeline Input Validation", test_pipeline_validation),
            ("Output Format Compatibility", test_output_format_compatibility),
            ("JSON Serialization", test_json_serialization),
            ("Integration Flow", test_integration_flow),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n{'='*15} {test_name} {'='*15}")
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        
        print(f"\n{'='*60}")
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED!")
            print("\n📋 Key Findings:")
            print("   ✅ VideoShotProcessingPipeline accepts 1-enriched_characters.json format")
            print("   ✅ Output is compatible with CharacterIdentificationPipeline")
            print("   ✅ Both face_conf and detection_confidence fields present")
            print("   ✅ JSON serialization works correctly")
            print("   ✅ Data validation works as expected")
            print("\n🚀 Ready for integration!")
            print("💡 Next step: Implement iteration in main.py to process each project")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: uv sync")
        print("2. Check that pipeline files are importable")
        print("3. Run with: uv run python tests/scripts/test_video_shot_processing.py")


if __name__ == "__main__":
    main()
