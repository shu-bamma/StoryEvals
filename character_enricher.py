import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI

from config import Config
from models import (
    Character,
    CharacterEnrichmentResult,
    CharacterTraits,
    CharacterTraitsStructured,
    EnrichedCharacter,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CharacterEnricher:
    """Enriches character descriptions with structured traits using VLM"""

    def __init__(self) -> None:
        self.config = Config.get_character_enrichment_config()
        self.client = OpenAI(api_key=self.config["api_key"])

        # Structured output schema for character traits
        self.traits_schema = {
            "type": "object",
            "properties": {
                "core": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concise discriminative phrases, e.g., 'green eyes', 'long blond wavy hair'",
                },
                "supportive": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Less discriminative traits",
                },
                "volatile": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Clothes, accessories, mood, background style - situational appearances",
                },
                "age_band": {
                    "type": "string",
                    "enum": [
                        "infant",
                        "toddler",
                        "child",
                        "teen",
                        "young_adult",
                        "adult",
                        "middle_aged",
                        "elderly",
                    ],
                    "description": "Age category of the character",
                },
                "skin_tone": {
                    "type": "string",
                    "enum": [
                        "fair",
                        "light",
                        "medium",
                        "tan",
                        "dark",
                        "olive",
                        "brown",
                        "black",
                        "gray",
                        "blue",
                        "green",
                        "other",
                    ],
                    "description": "Skin tone or fur color",
                },
                "type": {
                    "type": "string",
                    "enum": ["human", "animal", "mix", "fantasy", "other"],
                    "description": "Type of character",
                },
                "notes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any ambiguities or conflicts in the description",
                },
            },
            "required": [
                "core",
                "supportive",
                "volatile",
                "age_band",
                "skin_tone",
                "type",
                "notes",
            ],
        }

    def _extract_character_traits(self, character: Character) -> CharacterTraits | None:
        """Extract traits from a single character description using VLM with structured outputs"""

        prompt = f"""Extract FACE-CENTRIC traits only from this character description. Focus on hair, eyes, eyebrows, skin tone, facial marks, age band, and other distinguishing features.

Character: {character.name}
Description: {character.description}

Guidelines:
- core: Most important distinguishing features (hair color/style, eye color, unique facial features)
- supportive: Secondary features that help identify the character
- volatile: Clothing, accessories, mood, background elements that can change
- age_band: Choose the most appropriate age category from: infant, toddler, child, teen, young_adult, adult, middle_aged, elderly
- skin_tone: Primary skin or fur color from: fair, light, medium, tan, dark, olive, brown, black, gray, blue, green, other
- type: Whether human, animal, mix, fantasy, or other
- notes: Any unclear or conflicting information in the description

Your response will be automatically structured to include these traits organized by category."""

        for attempt in range(self.config["max_retries"]):
            try:
                # Use OpenAI's structured outputs API
                response = self.client.responses.parse(
                    model=self.config["model"],
                    input=[
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": prompt}],
                        }
                    ],
                    text_format=CharacterTraitsStructured,
                )

                # Extract the parsed output
                if response.output_parsed:
                    structured_traits = response.output_parsed

                    # Convert to CharacterTraits format
                    traits = CharacterTraits(
                        core=structured_traits.core,
                        supportive=structured_traits.supportive,
                        volatile=structured_traits.volatile,
                        age_band=structured_traits.age_band,
                        skin_tone=structured_traits.skin_tone,
                        type=structured_traits.type,
                        notes=structured_traits.notes,
                    )

                    logger.info(f"Successfully extracted traits for {character.name}")
                    return traits
                else:
                    raise ValueError("No parsed output received from OpenAI")

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed for {character.name}: {str(e)}"
                )
                if attempt < self.config["max_retries"] - 1:
                    delay = self.config["retry_delay"] * (
                        2**attempt
                    )  # Exponential backoff
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Failed to extract traits for {character.name} after {self.config['max_retries']} attempts"
                    )
                    return None

        # This should never be reached, but mypy needs it
        return None

    def _enrich_single_character(
        self, character: Character
    ) -> EnrichedCharacter | None:
        """Enrich a single character with traits"""
        try:
            traits = self._extract_character_traits(character)
            if traits is None:
                return None

            enriched = EnrichedCharacter(
                char_id=character.char_id,
                name=character.name,
                ref_image=character.image,
                description=character.description or "",
                embedding=[],  # Empty array for now
                traits=traits,
            )

            return enriched

        except Exception as e:
            logger.error(f"Error enriching character {character.name}: {str(e)}")
            return None

    def enrich_characters(
        self, characters: list[Character]
    ) -> CharacterEnrichmentResult:
        """Enrich multiple characters in parallel"""
        logger.info(
            f"Starting enrichment of {len(characters)} characters with {self.config['max_workers']} workers"
        )

        enriched_characters = []
        failed_characters = []

        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            # Submit all characters for processing
            future_to_character = {
                executor.submit(self._enrich_single_character, char): char
                for char in characters
            }

            # Process completed futures
            for future in as_completed(future_to_character):
                character = future_to_character[future]
                try:
                    enriched = future.result()
                    if enriched:
                        enriched_characters.append(enriched)
                        logger.info(f"✓ Enriched {character.name}")
                    else:
                        failed_characters.append(character.name)
                        logger.warning(f"✗ Failed to enrich {character.name}")
                except Exception as e:
                    failed_characters.append(character.name)
                    logger.error(f"✗ Exception enriching {character.name}: {str(e)}")

        # Log summary
        logger.info(
            f"Enrichment complete: {len(enriched_characters)} successful, {len(failed_characters)} failed"
        )
        if failed_characters:
            logger.warning(f"Failed characters: {', '.join(failed_characters)}")

        # Create result with metadata
        metadata = {
            "total_characters": len(characters),
            "successful_enrichments": len(enriched_characters),
            "failed_enrichments": len(failed_characters),
            "failed_character_names": failed_characters,
            "model_used": self.config["model"],
            "timestamp": time.time(),
        }

        return CharacterEnrichmentResult(
            characters_vault=enriched_characters, enrichment_metadata=metadata
        )

    def enrich_from_vault_data(
        self, vault_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Enrich characters from the vault data structure while preserving project structure"""
        logger.info(f"Processing vault data with {len(vault_data)} projects")

        enriched_projects = []

        for project in vault_data:
            enriched_project = project.copy()

            if "character_vault" in project:
                logger.info(
                    f"Enriching characters for project {project.get('project_id', 'unknown')}"
                )

                # Convert character data to Character objects for enrichment
                characters = []
                for char_data in project["character_vault"]:
                    character = Character(
                        char_id=char_data.get("char_id", ""),
                        name=char_data.get("name", ""),
                        image=char_data.get("image", ""),
                        description=char_data.get("description", ""),
                    )
                    characters.append(character)

                # Enrich the characters
                enrichment_result = self.enrich_characters(characters)

                # Replace the character_vault with enriched characters
                enriched_project["character_vault"] = [
                    {
                        "char_id": char.char_id,
                        "name": char.name,
                        "ref_image": char.ref_image,
                        "description": char.description,
                        "embedding": char.embedding,
                        "traits": {
                            "core": char.traits.core,
                            "supportive": char.traits.supportive,
                            "volatile": char.traits.volatile,
                            "age_band": char.traits.age_band,
                            "skin_tone": char.traits.skin_tone,
                            "type": char.traits.type,
                            "notes": char.traits.notes,
                        },
                    }
                    for char in enrichment_result.characters_vault
                ]

                # Add enrichment metadata to the project
                metadata = enrichment_result.enrichment_metadata
                if metadata:
                    enriched_project["enrichment_metadata"] = {
                        "total_characters": metadata.get("total_characters", 0),
                        "successful_enrichments": metadata.get(
                            "successful_enrichments", 0
                        ),
                        "failed_enrichments": metadata.get("failed_enrichments", 0),
                        "model_used": metadata.get("model_used", "unknown"),
                        "timestamp": metadata.get("timestamp", 0),
                    }

            enriched_projects.append(enriched_project)

        logger.info(f"Completed enrichment for {len(enriched_projects)} projects")
        return enriched_projects
