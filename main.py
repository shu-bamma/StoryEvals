import json
import os

from character_enricher import CharacterEnricher
from config import Config


def main() -> None:
    print("Starting StoryEvals evaluation pipeline...")

    # Step 1: Data Processing (Supporting function for evaluations)
    print("\n=== Step 1: Data Processing ===")
    # processor = DataProcessor()
    # all_results = processor.process_all_projects()
    # save_results_to_json(all_results)

    # Step 1.5: Character Enrichment
    print("\n=== Step 1.5: Character Enrichment ===")

    # Check if input file exists
    input_path = Config.CHARACTER_VAULT_INPUT_PATH
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    print(f"📖 Loading character vault from: {input_path}")
    with open(input_path) as f:
        vault_data = json.load(f)

    print(f"📊 Found {len(vault_data)} projects with character data")

    # Initialize character enricher
    enricher = CharacterEnricher()

    # Enrich characters
    print("🔍 Starting character enrichment...")
    enriched_projects = enricher.enrich_from_vault_data(vault_data)

    # Save enriched results
    output_path = Config.CHARACTER_ENRICHMENT_OUTPUT_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the enriched projects directly (they already have the right structure)
    with open(output_path, "w") as f:
        json.dump(enriched_projects, f, indent=2, ensure_ascii=False)

    print(f"💾 Enriched projects saved to: {output_path}")

    # Display enrichment summary
    total_characters = 0
    successful_enrichments = 0
    failed_enrichments = 0

    for project in enriched_projects:
        if "enrichment_metadata" in project:
            metadata = project["enrichment_metadata"]
            total_characters += metadata.get("total_characters", 0)
            successful_enrichments += metadata.get("successful_enrichments", 0)
            failed_enrichments += metadata.get("failed_enrichments", 0)

    print("\n📈 Enrichment Summary:")
    print(f"   Total projects: {len(enriched_projects)}")
    print(f"   Total characters: {total_characters}")
    print(f"   Successful: {successful_enrichments}")
    print(f"   Failed: {failed_enrichments}")

    # Display sample enriched character from first project
    if enriched_projects and "character_vault" in enriched_projects[0]:
        first_project = enriched_projects[0]
        if first_project["character_vault"]:
            sample = first_project["character_vault"][0]
            print(
                f"\n🔍 Sample enriched character from {first_project.get('project_id', 'unknown')}:"
            )
            print(f"   Name: {sample['name']}")
            print(f"   Core traits: {', '.join(sample['traits']['core'][:3])}")
            print(f"   Age band: {sample['traits']['age_band']}")
            print(f"   Type: {sample['traits']['type']}")

    # Step 2: Run Evaluations
    print("\n=== Step 2: Running Evaluations ===")

    # Step 3: Save Results
    print("\n=== Step 3: Saving Results ===")

    # Step 4: Display Evaluation Summary
    print("\n=== Evaluation Summary ===")

    print("\nEvaluation pipeline complete!")


if __name__ == "__main__":
    main()
