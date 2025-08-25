#!/usr/bin/env python3
"""Test script for the critique agent"""

import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import Config
from critique_agent import CritiqueAgent
from models import Character, VideoOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_critique_agent_initialization() -> bool:
    """Test that the critique agent can be initialized"""
    try:
        agent = CritiqueAgent()
        logger.info("✓ Critique agent initialized successfully")
        logger.info(f"  Model: {agent.config['model']}")
        logger.info(f"  Max retries: {agent.config['max_retries']}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to initialize critique agent: {e}")
        return False


def test_critique_agent_config() -> bool:
    """Test that the critique agent configuration is loaded correctly"""
    try:
        config = Config.get_critique_agent_config()
        logger.info("✓ Critique agent config loaded successfully")
        logger.info(f"  API key present: {'Yes' if config['api_key'] else 'No'}")
        logger.info(f"  Virtual key: {config['virtual_key']}")
        logger.info(f"  Model: {config['model']}")
        logger.info(f"  Max retries: {config['max_retries']}")
        logger.info(f"  Retry delay: {config['retry_delay']}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load critique agent config: {e}")
        return False


def test_critique_agent_with_sample_data() -> bool:
    """Test the critique agent with sample data"""
    try:
        _ = CritiqueAgent()

        # Create sample character
        sample_character = Character(
            char_id="test_char_001",
            name="Test Character",
            image="https://example.com/test_image.jpg",
            description="A test character with brown hair and blue eyes",
        )

        # Create sample video output
        sample_video = VideoOutput(
            project_id="test_project_001",
            clip_url="https://example.com/test_video.mp4",
            prompt="Test prompt",
            enhanced_prompt="Enhanced test prompt",
            reference_image="https://example.com/ref_image.jpg",
            thumbnail_image="https://example.com/thumb.jpg",
            characters=[sample_character],
            debug={},
        )

        logger.info("✓ Sample data created successfully")
        logger.info(f"  Character: {sample_character.name}")
        logger.info(f"  Video: {sample_video.project_id}")

        # Note: We won't actually call the API in this test to avoid costs
        # but we can verify the agent is ready
        logger.info("✓ Critique agent is ready for use")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to test critique agent with sample data: {e}")
        return False


def main() -> int:
    """Run all tests"""
    logger.info("Starting critique agent tests...")

    tests = [
        test_critique_agent_config,
        test_critique_agent_initialization,
        test_critique_agent_with_sample_data,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                logger.warning(f"Test {test.__name__} failed")
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")

    logger.info(f"\nTest Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
