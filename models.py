from dataclasses import dataclass
from typing import Any


@dataclass
class Character:
    name: str
    image: str
    description: str


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
