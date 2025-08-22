import json
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
        """Extract traits from a single character description using VLM"""

        prompt = f"""Extract FACE-CENTRIC traits only from this character description. Focus on hair, eyes, eyebrows, skin tone, facial marks, age band, and other distinguishing features.

Character: {character.name}
Description: {character.description}

Put clothing/accessories in volatile_traits (they can change). Return structured JSON with the traits organized by category.

Guidelines:
- core: Most important distinguishing features (hair color/style, eye color, unique facial features)
- supportive: Secondary features that help identify the character
- volatile: Clothing, accessories, mood, background elements that can change
- age_band: Choose the most appropriate age category from: infant, toddler, child, teen, young_adult, adult, middle_aged, elderly
- skin_tone: Primary skin or fur color from: fair, light, medium, tan, dark, olive, brown, black, gray, blue, green, other
- type: Whether human, animal, mix, fantasy, or other
- notes: Any unclear or conflicting information in the description

IMPORTANT: Return ONLY valid JSON with these exact field names: core, supportive, volatile, age_band, skin_tone, type, notes"""

        for attempt in range(self.config["max_retries"]):
            try:
                response = self.client.chat.completions.create(
                    model=self.config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from VLM")

                traits_data = json.loads(content)

                # Validate and create CharacterTraits object
                # Ensure lists are properly formatted
                def ensure_list(value: Any) -> list[str]:
                    if isinstance(value, list):
                        return value
                    elif isinstance(value, dict):
                        # Convert dict to list of key-value strings
                        return [f"{k}: {v}" for k, v in value.items()]
                    elif value is None:
                        return []
                    else:
                        return [str(value)]

                traits = CharacterTraits(
                    core=ensure_list(traits_data.get("core", [])),
                    supportive=ensure_list(traits_data.get("supportive", [])),
                    volatile=ensure_list(traits_data.get("volatile", [])),
                    age_band=traits_data.get("age_band", "unknown"),
                    skin_tone=traits_data.get("skin_tone", "unknown"),
                    type=traits_data.get("type", "unknown"),
                    notes=ensure_list(traits_data.get("notes", [])),
                )

                logger.info(f"Successfully extracted traits for {character.name}")
                return traits

            except (json.JSONDecodeError, ValueError, KeyError) as e:
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

            except Exception as e:
                logger.error(f"Unexpected error processing {character.name}: {str(e)}")
                if attempt < self.config["max_retries"] - 1:
                    delay = self.config["retry_delay"] * (2**attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
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
