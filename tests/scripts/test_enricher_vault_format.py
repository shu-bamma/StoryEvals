#!/usr/bin/env python3
"""
Test character enricher with vault data format
"""

import json
import logging

from character_enricher import CharacterEnricher

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def create_sample_vault_data() -> list[dict]:
    """Create sample vault data in the same format as the actual data"""
    return [
        {
            "character_vault": [
                {
                    "char_id": "sample_girl",
                    "name": "Emma",
                    "image": "https://example.com/emma.jpg",
                    "description": "A teenage girl with curly red hair, green eyes, and freckles. She has a cheerful personality and is often seen wearing colorful clothes.",
                },
                {
                    "char_id": "sample_boy",
                    "name": "Jake",
                    "image": "https://example.com/jake.jpg",
                    "description": "A young man with short black hair, dark brown eyes, and an athletic build. He has a confident stance and is wearing casual sports attire.",
                },
            ],
            "project_id": "TEST_PROJECT_001",
            "job_id": "test_job_001",
        }
    ]


def main() -> None:
    """Test character enricher with vault format"""
    print("🚀 Testing Character Enricher with Vault Data Format")
    print("=" * 60)

    try:
        # Create sample vault data
        print("📝 Creating sample vault data...")
        vault_data = create_sample_vault_data()
        print(f"✅ Created vault data with {len(vault_data)} projects")

        # Initialize enricher
        print("\n🔧 Initializing character enricher...")
        enricher = CharacterEnricher()
        print("✅ Character enricher initialized")

        # Process vault data
        print("\n🔄 Processing vault data enrichment...")
        enriched_vault_data = enricher.enrich_from_vault_data(vault_data)

        print("✅ Vault enrichment completed!")
        print(f"📊 Processed {len(enriched_vault_data)} projects")

        # Display results
        for project in enriched_vault_data:
            print(f"\n📋 Project: {project.get('project_id', 'unknown')}")

            if "character_vault" in project:
                print(f"   Characters: {len(project['character_vault'])}")

                for char in project["character_vault"]:
                    print(f"\n   Character: {char['name']} ({char['char_id']})")
                    traits = char["traits"]
                    print(f"     Core: {traits['core']}")
                    print(
                        f"     Age: {traits['age_band']}, Skin: {traits['skin_tone']}, Type: {traits['type']}"
                    )

            if "enrichment_metadata" in project:
                meta = project["enrichment_metadata"]
                print(
                    f"   Enrichment: {meta['successful_enrichments']}/{meta['total_characters']} successful"
                )

        # Save results for inspection
        output_file = "data/logs/test_enriched_vault.json"
        with open(output_file, "w") as f:
            json.dump(enriched_vault_data, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")

        print("\n🎉 Vault format test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
