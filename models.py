from dataclasses import dataclass
from typing import List, Optional, Dict, Any


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
    characters: List[Character]
    debug: Optional[Dict[str, Any]] = None


@dataclass
class CharacterVerification:
    """Result of verifying if a character appears in a video clip"""
    character_name: str
    character_image: str
    video_clip_url: str
    is_present: bool
    confidence_score: float
    reasoning: str
    timestamp_analysis: Optional[Dict[str, Any]] = None


@dataclass
class CritiqueAgentResult:
    """Complete result from the critique agent evaluation"""
    project_id: str
    video_output: VideoOutput
    character_verifications: List[CharacterVerification]
    overall_accuracy: float
    evaluation_notes: str
    llm_metadata: Optional[Dict[str, Any]] = None 