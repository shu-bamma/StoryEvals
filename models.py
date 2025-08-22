from dataclasses import dataclass
from typing import Any


@dataclass
class Character:
    char_id: str  # name in this case
    name: str
    image: str
    description: str | None = None


@dataclass
class CharacterTraits:
    """Structured traits extracted from character description"""

    core: list[
        str
    ]  # concise discriminative phrases, e.g., "green eyes", "long blond wavy hair"
    supportive: list[str]  # less discriminative traits
    volatile: list[
        str
    ]  # clothes, accessories, mood, background style - situational appearances
    age_band: str  # teen|young_adult|adult|etc
    skin_tone: str  # fair|light|medium|tan|dark|etc
    type: str  # animal|human|mix
    notes: list[str]  # ambiguities or conflicts in text


@dataclass
class EnrichedCharacter:
    """Character with enriched traits and embeddings"""

    char_id: str
    name: str
    ref_image: str
    description: str
    embedding: list[float]  # Empty array for now
    traits: CharacterTraits


@dataclass
class CharacterEnrichmentResult:
    """Result of character enrichment process"""

    characters_vault: list[EnrichedCharacter]
    enrichment_metadata: dict[str, Any] | None = None


@dataclass
class VideoOutput:
    project_id: str
    clip_url: str
    prompt: str
    enhanced_prompt: str
    reference_image: str
    thumbnail_image: str
    characters: list[Character]
    debug: dict[str, Any] | None = None


@dataclass
class CharacterVerification:
    """Result of verifying if a character appears in a video clip"""

    character_name: str
    character_image: str
    video_clip_url: str
    is_present: bool
    confidence_score: float
    reasoning: str
    timestamp_analysis: dict[str, Any] | None = None


@dataclass
class CritiqueAgentResult:
    """Complete result from the critique agent evaluation"""

    project_id: str
    video_output: VideoOutput
    character_verifications: list[CharacterVerification]
    overall_accuracy: float
    evaluation_notes: str
    llm_metadata: dict[str, Any] | None = None
