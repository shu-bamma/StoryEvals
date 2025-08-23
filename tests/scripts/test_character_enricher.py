#!/usr/bin/env python3
"""
Test script for the character enricher
"""

import logging

from character_enricher import CharacterEnricher
from models import Character

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def create_sample_characters() -> list[Character]:
    """Create sample characters for testing"""
    characters = [
        Character(
            char_id="test_char_1",
            name="Anna",
            image="https://example.com/anna.jpg",
            description="A young woman with long blonde hair, blue eyes, and a bright smile. She's wearing a red dress and has a confident demeanor. She appears to be in her mid-twenties with fair skin.",
        ),
        Character(
            char_id="test_char_2",
            name="Marcus",
            image="https://example.com/marcus.jpg",
            description="An older man with graying hair, brown eyes, and a beard. He has a muscular build and is wearing a dark leather jacket. He appears to be in his forties with weathered skin and a serious expression.",
        ),
    ]
    return characters


def main() -> None:
    """Main test function"""
    print("🚀 Starting Character Enricher Test")
    print("=" * 50)

    try:
        # Create sample characters
        print("📝 Creating sample characters...")
        characters = create_sample_characters()
        print(f"✅ Created {len(characters)} sample characters")

        # Initialize enricher
        print("\n🔧 Initializing character enricher...")
        enricher = CharacterEnricher()
        print("✅ Character enricher initialized successfully")

        # Test enrichment
        print("\n🔄 Processing character enrichment...")
        print("Note: This will make actual LLM calls to extract traits")

        result = enricher.enrich_characters(characters)

        print("✅ Enrichment completed successfully!")
        if result.enrichment_metadata:
            print(
                f"📊 Results: {result.enrichment_metadata['successful_enrichments']}/{result.enrichment_metadata['total_characters']} characters enriched"
            )
        else:
            print("📊 Results: No metadata available")

        # Display results
        print("\n📋 Enrichment Results:")
        for enriched_char in result.characters_vault:
            print(f"\nCharacter: {enriched_char.name} ({enriched_char.char_id})")
            print(f"  Core traits: {enriched_char.traits.core}")
            print(f"  Supportive traits: {enriched_char.traits.supportive}")
            print(f"  Volatile traits: {enriched_char.traits.volatile}")
            print(f"  Age band: {enriched_char.traits.age_band}")
            print(f"  Skin tone: {enriched_char.traits.skin_tone}")
            print(f"  Type: {enriched_char.traits.type}")
            if enriched_char.traits.notes:
                print(f"  Notes: {enriched_char.traits.notes}")

        # Test metadata
        print("\n📈 Metadata:")
        metadata = result.enrichment_metadata
        if metadata:
            print(f"  Model used: {metadata.get('model_used', 'unknown')}")
            print(f"  Total characters: {metadata.get('total_characters', 0)}")
            print(f"  Successful: {metadata.get('successful_enrichments', 0)}")
            print(f"  Failed: {metadata.get('failed_enrichments', 0)}")

        print("\n🎉 Character enricher test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check that your OpenAI API key is set in environment variables")
        print("2. Verify that all dependencies are installed")
        print("3. Check the logs for detailed error information")
        raise


if __name__ == "__main__":
    main()
