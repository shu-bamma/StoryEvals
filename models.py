from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class Character:
    char_id: str  # name in this case
    name: str
    image: str
    description: str | None = None


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


# New models for character identification
@dataclass
class CharacterTrait:
    """Character traits for identification"""

    core: list[str]
    supportive: list[str]
    volatile: list[str]
    age_band: str
    skin_tone: str
    type: str
    notes: list[str]


@dataclass
class CharacterVaultEntry:
    """Character entry in the vault"""

    char_id: str
    name: str
    ref_image: str
    traits: CharacterTrait


@dataclass
class CropQuality:
    """Quality metrics for a face crop"""

    blur: float | None = None
    pose: dict[str, float] | None = None


@dataclass
class Crop:
    """Face crop with identification results"""

    crop_id: str
    bbox_norm: list[float]
    crop_url: str
    detector: str
    face_conf: float
    quality: CropQuality | None = None
    # New fields for character identification
    pred_char_id: str | None = None
    confidence: float | None = None
    reason: str | None = None


@dataclass
class Shot:
    """Video shot with keyframes and crops"""

    keyframes_ms: list[int]
    crops: dict[int, list[Crop]]
    shot_id: str


@dataclass
class CharacterIdentificationResult:
    """Complete result from character identification"""

    character_vault: list[CharacterVaultEntry]
    shots: list[Shot]
    # Additional metadata
    project_id: str | None = None
    job_id: str | None = None
    video_url: str | None = None
    identification_metadata: dict[str, Any] | None = None
    # Critique agent verification results
    critique_agent_result: CritiqueAgentResult | None = None


@dataclass
class CharacterIdentificationBatch:
    """Batch of crops for character identification"""

    crops: list[Crop]
    character_vault: list[CharacterVaultEntry]
    batch_id: str


@dataclass
class CharacterIdentificationResponse:
    """LLM response for character identification"""

    crop_id: str
    pred_char_id: str
    confidence: float
    reason: str


# Models for character enrichment
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


# Pydantic models for structured LLM outputs (character identification)
class CropIdentification(BaseModel):
    """Structured output for a single crop identification"""

    crop_id: str = Field(description="The unique identifier for the crop")
    pred_char_id: str = Field(
        description="The predicted character ID from the vault, or 'Unknown' if no match found"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 for the prediction",
    )
    reason: str = Field(
        description="Short justification citing 1-3 visible CORE traits that support the identification"
    )


class CharacterIdentificationBatchResponse(BaseModel):
    """Structured output for a batch of crop identifications"""

    crops: list[CropIdentification] = Field(
        description="List of character identifications for each crop in the batch"
    )


# Pydantic models for structured LLM outputs (character enrichment)
class CharacterTraitsStructured(BaseModel):
    """Structured output for character traits from OpenAI"""

    core: list[str] = Field(
        description="Concise discriminative phrases, e.g., 'green eyes', 'long blond wavy hair'"
    )
    supportive: list[str] = Field(description="Less discriminative traits")
    volatile: list[str] = Field(
        description="Clothes, accessories, mood, background style - situational appearances"
    )
    age_band: str = Field(
        description="Age category of the character",
        json_schema_extra={
            "enum": [
                "infant",
                "toddler",
                "child",
                "teen",
                "young_adult",
                "adult",
                "middle_aged",
                "elderly",
            ]
        },
    )
    skin_tone: str = Field(
        description="Skin tone or fur color",
        json_schema_extra={
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
            ]
        },
    )
    type: str = Field(
        description="Type of character",
        json_schema_extra={"enum": ["human", "animal", "mix", "fantasy", "other"]},
    )
    notes: list[str] = Field(
        description="Any ambiguities or conflicts in the description"
    )
