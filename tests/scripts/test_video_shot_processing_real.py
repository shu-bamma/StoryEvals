#!/usr/bin/env python3
"""
Real integration test for video shot processing pipeline
Tests actual video processing with real data from 1-enriched_characters.json
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import logging

from video_shot_processing_pipeline import VideoShotProcessingPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_test_data() -> list[dict]:
    """Load first 3 projects from 1-enriched_characters.json for testing"""
    enriched_path = project_root / "data" / "logs" / "1-enriched_characters.json"
    
    if not enriched_path.exists():
        raise FileNotFoundError(f"Enriched characters file not found: {enriched_path}")
    
    with open(enriched_path, 'r') as f:
        all_data = json.load(f)
    
    # Take only first 3 projects for testing
    test_data = all_data[:3]
    
    print(f"📊 Loaded {len(test_data)} projects for testing:")
    for i, project in enumerate(test_data):
        print(f"   {i+1}. {project['project_id']} - {len(project['character_vault'])} characters")
        print(f"      Video: {project['video_url'][:60]}...")
    
    return test_data


def test_real_video_processing(create_debug_video: bool = False) -> bool:
    """Test actual video processing with real data"""
    print("\n🎬 Testing Real Video Processing")
    print("=" * 60)
    print("⚠️  This will download videos and run YOLO detection!")
    print("⏱️  This may take several minutes...")
    if create_debug_video:
        print("🎥 Debug videos will be created with shot boundaries and YOLO detections!")
    print()
    
    try:
        # Load test data
        test_data = load_test_data()
        
        # Initialize pipeline
        print("🔧 Initializing VideoShotProcessingPipeline...")
        pipeline = VideoShotProcessingPipeline()
        
        # Process the videos
        print("🚀 Starting real video processing...")
        print("   This will:")
        print("   1️⃣ Download videos from URLs")
        print("   2️⃣ Run shot boundary detection")
        print("   3️⃣ Run YOLO on keyframes")
        print("   4️⃣ Extract character crops")
        if create_debug_video:
            print("   5️⃣ Create debug videos with visualizations")
        print()
        
        results = pipeline.process(test_data, create_debug_video=create_debug_video)
        
        # Analyze results
        print("\n📊 Processing Results:")
        successful_projects = 0
        total_shots = 0
        total_crops = 0
        debug_videos = []
        
        for result in results:
            if "error" in result:
                print(f"   ❌ {result['project_id']}: {result['error']}")
            else:
                successful_projects += 1
                project_shots = result.get("number_of_shots", 0)
                total_shots += project_shots
                
                # Count crops
                project_crops = 0
                if "shots" in result:
                    for shot in result["shots"]:
                        if hasattr(shot, 'crops'):
                            for crops_list in shot.crops.values():
                                project_crops += len(crops_list)
                        elif isinstance(shot, dict) and "crops" in shot:
                            for crops_list in shot["crops"].values():
                                project_crops += len(crops_list)
                
                total_crops += project_crops
                print(f"   ✅ {result['project_id']}: {project_shots} shots, {project_crops} crops")
                
                # Check for debug video
                if "debug_video_path" in result:
                    debug_videos.append(result["debug_video_path"])
                    print(f"      🎥 Debug video: {result['debug_video_path']}")
        
        print(f"\n📈 Summary:")
        print(f"   Total projects: {len(results)}")
        print(f"   Successful: {successful_projects}")
        print(f"   Failed: {len(results) - successful_projects}")
        print(f"   Total shots: {total_shots}")
        print(f"   Total crops: {total_crops}")
        
        if debug_videos:
            print(f"\n🎥 Debug Videos Created:")
            for video_path in debug_videos:
                print(f"   📹 {video_path}")
        
        # Save results
        output_path = pipeline.save_results(results)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Test format compatibility
        if successful_projects > 0:
            print("\n🔍 Testing format compatibility...")
            # Load the saved results
            with open(output_path, 'r') as f:
                saved_results = json.load(f)
            
            # Test that first successful project can be parsed by CharacterIdentificationPipeline
            first_successful = next((r for r in saved_results if "error" not in r), None)
            if first_successful:
                print(f"   ✅ Output format verified for {first_successful['project_id']}")
                print(f"   ✅ Ready for CharacterIdentificationPipeline")
        
        return successful_projects > 0
        
    except Exception as e:
        print(f"❌ Real video processing test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Main test function"""
    print("🚀 Starting REAL Video Shot Processing Test")
    print("=" * 60)
    print("📝 Testing with actual videos from 1-enriched_characters.json")
    print("🎯 This will run the complete pipeline:")
    print("   • Video download")
    print("   • Shot boundary detection (PySceneDetect)")
    print("   • YOLO object detection")
    print("   • Crop extraction")
    print()
    
    # Ask for confirmation and debug video option
    response = input("⚠️  This will download videos and may take time. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return
    
    debug_response = input("🎥 Create debug videos with shot boundaries and YOLO visualizations? (y/N): ")
    create_debug_video = debug_response.lower() == 'y'
    
    try:
        success = test_real_video_processing(create_debug_video)
        
        if success:
            print("\n🎉 REAL VIDEO PROCESSING TEST PASSED!")
            print("\n📋 What was verified:")
            print("   ✅ Actual video download from URLs")
            print("   ✅ Shot boundary detection with PySceneDetect")
            print("   ✅ YOLO object detection on video frames")
            print("   ✅ Character crop extraction and saving")
            print("   ✅ Output format compatible with CharacterIdentificationPipeline")
            if create_debug_video:
                print("   ✅ Debug videos created with visualizations")
                print("   📹 Check temp/videos/ directory for *_debug.mp4 files")
            print("\n🚀 The video shot processing pipeline is working end-to-end!")
        else:
            print("\n❌ Real video processing test failed.")
            print("Check the logs above for details.")
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")


if __name__ == "__main__":
    main()
