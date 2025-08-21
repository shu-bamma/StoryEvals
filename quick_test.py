#!/usr/bin/env python3
"""
Quick test script for the critique agent with your specific test data
Uses the official OpenAI Python client
"""

import os

from dotenv import load_dotenv

from critique_agent import CritiqueAgent
from models import Character


def main() -> None:
    print("Quick Test - Critique Agent (OpenAI Client)")
    print("=" * 50)

    # Load .env file explicitly
    print("🔍 Loading environment variables...")
    load_dotenv()

    # Debug environment variables
    print("\n📋 Environment Variables:")
    llm_key = os.getenv("LLM_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if llm_key:
        print(
            f"  LLM_API_KEY: {llm_key[:10]}...{llm_key[-4:] if len(llm_key) > 14 else ''}"
        )
    else:
        print("  LLM_API_KEY: ❌ Not set")

    if openai_key:
        print(
            f"  OPENAI_API_KEY: {openai_key[:10]}...{openai_key[-4:] if len(openai_key) > 14 else ''}"
        )
    else:
        print("  OPENAI_API_KEY: ❌ Not set")

    # Check .env file
    env_file_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_file_path):
        print(f"  .env file: ✅ Found at {env_file_path}")
        try:
            with open(env_file_path) as f:
                content = f.read().strip()
                if content:
                    print(f"  .env content: {len(content)} characters")
                    # Show first line (usually the API key)
                    first_line = content.split("\n")[0] if "\n" in content else content
                    if "=" in first_line:
                        key = first_line.split("=")[0]
                        value = first_line.split("=")[1]
                        print(f"  First variable: {key} = {value[:10]}...")
                else:
                    print("  .env content: ❌ Empty file")
        except Exception as e:
            print(f"  .env read error: {e}")
    else:
        print(f"  .env file: ❌ Not found at {env_file_path}")

    # Your test data
    image_url = "https://content.dashtoon.ai/stability-images/73afe87e-6e6a-4cbf-9825-9b5e6e37ec77.webp"
    video_url = "https://content.dashtoon.ai/saved-videos/ebf71682-2f6f-44fb-bcfc-36d75933c43e.mp4"

    print("\n🧪 Testing with:")
    print(f"Image: {image_url}")
    print(f"Video: {video_url}")
    print()

    # Check API key
    api_key = llm_key or openai_key
    if not api_key:
        print("❌ No OpenAI API key found!")
        print("\n💡 Troubleshooting:")
        print("1. Check if .env file exists and contains:")
        print("   LLM_API_KEY=your_actual_api_key_here")
        print("2. Or set environment variable manually:")
        print("   export LLM_API_KEY='your_openai_api_key_here'")
        print("3. Make sure .env file is in the same directory as this script")
        return

    print(f"✅ API key found: {api_key[:10]}...")

    # Create test character
    character = Character(
        name="Character from Image",
        image=image_url,
        description="The main character visible in the reference image",
    )

    try:
        # Initialize agent (now uses OpenAI client)
        print("✅ Initializing OpenAI client...")
        agent = CritiqueAgent()
        print("✅ Critique agent initialized with OpenAI client")

        # Test verification
        print("\n🔍 Running character verification...")
        print("   This will use OpenAI's official Python client")
        print("   Model: gpt-5 (from your config)")

        result = agent.verify_character_in_video(character, video_url)

        # Display results
        print("\n📊 Results:")
        print(f"Character Present: {'✅ YES' if result.is_present else '❌ NO'}")
        print(f"Confidence: {result.confidence_score:.1%}")
        print(f"Reasoning: {result.reasoning}")

        if result.timestamp_analysis:
            print(f"Timestamp Analysis: {result.timestamp_analysis}")

        print("\n🎉 Test completed successfully using OpenAI client!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 This might be due to:")
        print("1. Invalid API key")
        print("2. Model 'gpt-5' not available in your account")
        print("3. Insufficient credits")
        print("4. Network connectivity issues")


if __name__ == "__main__":
    main()
