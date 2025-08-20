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