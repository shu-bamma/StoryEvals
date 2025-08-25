import base64

import requests
from portkey_ai import Portkey
from pydantic import BaseModel, Field

from config import Config
from models import Character, CharacterVerification, CritiqueAgentResult, VideoOutput


class CharacterVerificationResponse(BaseModel):
    """Structured output for character verification from OpenAI"""

    is_present: bool = Field(
        description="Whether the character appears in the video clip"
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 for the verification",
    )
    reasoning: str = Field(
        description="Detailed explanation of the verification analysis"
    )
    timestamp_analysis: dict[str, str | list[str]] | None = Field(
        default=None, description="Analysis of when the character appears in the video"
    )


class CritiqueAgent:
    """
    Agent that uses OpenAI's GPT models to verify if characters in images appear in video clips
    """

    def __init__(self) -> None:
        """Initialize the critique agent with configuration from Config"""
        self.config = Config.get_critique_agent_config()

        if not self.config["api_key"]:
            raise ValueError(
                "OpenAI API key is required. Set LLM_API_KEY environment variable."
            )

        # Initialize Portkey client
        self.client = Portkey(
            api_key=self.config["api_key"],
            virtual_key=self.config["virtual_key"],
        )

    def encode_image_to_base64(self, image_url: str) -> str:
        """
        Download and encode image to base64 for OpenAI API input

        Args:
            image_url: URL of the image to encode

        Returns:
            Base64 encoded image string
        """
        try:
            response = requests.get(image_url, timeout=Config.IMAGE_DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            image_data = response.content
            return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image {image_url}: {e}")
            return ""

    def create_verification_prompt(
        self, character: Character, video_clip_url: str
    ) -> str:
        """
        Create a prompt for OpenAI to verify character presence in video

        Args:
            character: Character object with name, image, and description
            video_clip_url: URL of the video clip to analyze

        Returns:
            Formatted prompt for OpenAI
        """
        prompt = f"""
You are a character verification expert. Your task is to determine if a specific character appears in a video clip.

CHARACTER INFORMATION:
- Name: {character.name}
- Description: {character.description}
- Reference Image: [Image will be provided separately]

VIDEO CLIP:
- URL: {video_clip_url}

INSTRUCTIONS:
1. Analyze the character's reference image carefully
2. Watch the entire video clip from the URL provided
3. Determine if the character from the reference image appears in the video
4. Consider visual similarities, clothing, facial features, and context
5. If the character appears but looks different (e.g., different clothing, lighting), still mark as present
6. Provide specific reasoning for your decision
7. Analyze timestamps and key scenes where the character appears

Your response will be automatically structured to include:
- Whether the character is present (true/false)
- A confidence score between 0.0 and 1.0
- Detailed reasoning for your analysis
- Timestamp analysis including first appearance, total screen time, and key scenes
"""
        return prompt.strip()

    def call_openai_for_verification(
        self, prompt: str, image_base64: str
    ) -> CharacterVerificationResponse:
        """
        Call OpenAI's API to perform character verification using structured outputs

        Args:
            prompt: Text prompt for OpenAI
            image_base64: Base64 encoded reference image

        Returns:
            Structured verification response from OpenAI
        """
        for attempt in range(self.config["max_retries"]):
            try:
                # Check if using GPT-5
                is_gpt5 = "gpt-5" in self.config["model"]

                # Prepare API call parameters
                api_params = {
                    "model": self.config["model"],
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                                },
                            ],
                        }
                    ],
                    "text_format": CharacterVerificationResponse,
                }

                # GPT-5 uses different parameters
                if is_gpt5:
                    api_params["max_completion_tokens"] = self.config.get(
                        "max_completion_tokens"
                    )
                    # GPT-5 only supports temperature=1 (default)
                else:
                    api_params["max_tokens"] = self.config.get("max_tokens")
                    api_params["temperature"] = self.config.get("temperature")

                # Use OpenAI's structured outputs API
                response = self.client.responses.parse(
                    model=api_params["model"],
                    input=api_params["input"],
                    text_format=api_params["text_format"],
                    max_completion_tokens=api_params.get("max_completion_tokens"),
                    max_tokens=api_params.get("max_tokens"),
                    temperature=api_params.get("temperature"),
                )

                # Extract the parsed output
                if response.output_parsed:
                    # The output_parsed is already of type CharacterVerificationResponse
                    # but we need to ensure type safety
                    parsed_output = response.output_parsed
                    if isinstance(parsed_output, CharacterVerificationResponse):
                        # Explicit type assertion to help the type checker
                        result: CharacterVerificationResponse = parsed_output
                        return result
                    else:
                        # If somehow the type is wrong, reconstruct it
                        # Explicit type annotation to ensure the type checker understands this returns CharacterVerificationResponse
                        validated_response: CharacterVerificationResponse = (
                            CharacterVerificationResponse.model_validate(
                                parsed_output.model_dump()
                            )
                        )
                        return validated_response
                else:
                    raise ValueError("No parsed output received from OpenAI")

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config["max_retries"] - 1:
                    delay = self.config["retry_delay"] * (
                        2**attempt
                    )  # Exponential backoff
                    print(f"Retrying in {delay} seconds...")
                    import time

                    time.sleep(delay)
                else:
                    print(f"Failed after {self.config['max_retries']} attempts")
                    # Return default response on error
                    default_response: CharacterVerificationResponse = (
                        CharacterVerificationResponse(
                            is_present=False,
                            confidence_score=0.0,
                            reasoning=f"Error during verification: {str(e)}",
                            timestamp_analysis={},
                        )
                    )
                    return default_response

        # This should never be reached, but mypy needs it
        return CharacterVerificationResponse(
            is_present=False,
            confidence_score=0.0,
            reasoning="Unexpected error in verification",
            timestamp_analysis={},
        )

    def _parse_fallback_response(self, content: str) -> CharacterVerificationResponse:
        """
        Fallback parsing for when structured output fails

        Args:
            content: Raw text response from OpenAI

        Returns:
            Parsed verification response
        """
        # Default values
        result = {
            "is_present": False,
            "confidence_score": 0.5,
            "reasoning": content,
            "timestamp_analysis": {},
        }

        # Try to extract presence information
        content_lower = content.lower()
        if any(
            word in content_lower
            for word in ["yes", "present", "appears", "visible", "seen", "true"]
        ):
            result["is_present"] = True
        elif any(
            word in content_lower
            for word in [
                "no",
                "not present",
                "absent",
                "not visible",
                "not seen",
                "false",
            ]
        ):
            result["is_present"] = False

        # Try to extract confidence score
        import re

        confidence_match = re.search(r"confidence[:\s]*(\d+\.?\d*)", content_lower)
        if confidence_match:
            try:
                score = float(confidence_match.group(1))
                if 0 <= score <= 1:
                    result["confidence_score"] = score
            except ValueError:
                pass

        # Try to extract timestamp information
        timestamp_patterns = {
            "first_appearance": r"first.*?appear.*?(\d{1,2}:\d{2})",
            "total_screen_time": r"total.*?time.*?(\d{1,2}:\d{2})",
            "key_scenes": r"scene.*?(\d{1,2}:\d{2})",
        }

        for key, pattern in timestamp_patterns.items():
            matches = re.findall(pattern, content_lower)
            if matches:
                timestamp_value = matches[0] if key != "key_scenes" else matches
                if isinstance(result["timestamp_analysis"], dict):
                    result["timestamp_analysis"][key] = timestamp_value

        return CharacterVerificationResponse(
            is_present=result["is_present"],
            confidence_score=result["confidence_score"],
            reasoning=result["reasoning"],
            timestamp_analysis=result["timestamp_analysis"],
        )

    def verify_character_in_video(
        self, character: Character, video_clip_url: str
    ) -> CharacterVerification:
        """
        Verify if a character appears in a video clip using OpenAI

        Args:
            character: Character to verify
            video_clip_url: URL of the video clip

        Returns:
            CharacterVerification result
        """
        print(f"Verifying character '{character.name}' in video: {video_clip_url}")

        # Encode the character image
        image_base64 = self.encode_image_to_base64(character.image)
        if not image_base64:
            return CharacterVerification(
                character_name=character.name,
                character_image=character.image,
                video_clip_url=video_clip_url,
                is_present=False,
                confidence_score=0.0,
                reasoning="Failed to encode reference image",
                timestamp_analysis={},
            )

        # Create verification prompt
        prompt = self.create_verification_prompt(character, video_clip_url)

        # Call OpenAI API for verification
        openai_response = self.call_openai_for_verification(prompt, image_base64)

        # Convert to CharacterVerification format
        return CharacterVerification(
            character_name=character.name,
            character_image=character.image,
            video_clip_url=video_clip_url,
            is_present=openai_response.is_present,
            confidence_score=openai_response.confidence_score,
            reasoning=openai_response.reasoning,
            timestamp_analysis=openai_response.timestamp_analysis,
        )

    def evaluate_video_output(self, video_output: VideoOutput) -> CritiqueAgentResult:
        """
        Evaluate a video output by verifying all its characters using OpenAI

        Args:
            video_output: VideoOutput to evaluate

        Returns:
            CritiqueAgentResult with verification details
        """
        print(f"Evaluating video output for project: {video_output.project_id}")

        character_verifications = []

        # Verify each character in the video
        for character in video_output.characters:
            verification = self.verify_character_in_video(
                character, video_output.clip_url
            )
            character_verifications.append(verification)

        # Calculate overall accuracy
        if character_verifications:
            # For now, we'll use a simple metric - you can customize this
            # based on your evaluation criteria
            present_characters = sum(1 for v in character_verifications if v.is_present)
            total_characters = len(character_verifications)
            overall_accuracy = (
                present_characters / total_characters if total_characters > 0 else 0.0
            )
        else:
            overall_accuracy = 0.0

        # Generate evaluation notes
        evaluation_notes = self._generate_evaluation_notes(
            character_verifications, overall_accuracy
        )

        return CritiqueAgentResult(
            project_id=video_output.project_id,
            video_output=video_output,
            character_verifications=character_verifications,
            overall_accuracy=overall_accuracy,
            evaluation_notes=evaluation_notes,
            llm_metadata={
                "verification_count": len(character_verifications),
                "llm_provider": "OpenAI",
                "model": self.config["model"],
            },
        )

    def _generate_evaluation_notes(
        self, verifications: list[CharacterVerification], accuracy: float
    ) -> str:
        """
        Generate human-readable evaluation notes

        Args:
            verifications: List of character verifications
            accuracy: Overall accuracy score

        Returns:
            Formatted evaluation notes
        """
        notes = f"Overall Character Presence Accuracy: {accuracy:.2%}\n\n"

        for verification in verifications:
            status = "✓ PRESENT" if verification.is_present else "✗ NOT PRESENT"
            confidence = f"{verification.confidence_score:.1%}"
            notes += (
                f"{verification.character_name}: {status} (Confidence: {confidence})\n"
            )
            notes += f"  Reasoning: {verification.reasoning}\n\n"

        return notes.strip()
