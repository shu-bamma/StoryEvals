import base64
import logging
import time

import requests
from portkey_ai import Portkey

from config import Config
from models import (
    CharacterIdentificationBatch,
    CharacterIdentificationBatchResponse,
    CharacterIdentificationResponse,
)

logger = logging.getLogger(__name__)


class CharacterIdentifier:
    """LLM-based character identification agent"""

    def __init__(self) -> None:
        self.config = Config.get_character_identification_config()
        self.client = Portkey(
            api_key=self.config["api_key"], virtual_key=self.config["virtual_key"]
        )
        self.max_retries = self.config["max_retries"]
        self.retry_timeout = self.config["retry_timeout"]
        self.exponential_backoff_base = self.config["exponential_backoff_base"]

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        try:
            if image_path.startswith("http"):
                # Download remote image
                response = requests.get(
                    image_path, timeout=Config.IMAGE_DOWNLOAD_TIMEOUT
                )
                response.raise_for_status()
                image_data = response.content
            elif image_path.startswith("file://"):
                # Local file path
                local_path = image_path.replace("file://", "")
                with open(local_path, "rb") as f:
                    image_data = f.read()
            else:
                # Assume local file path
                with open(image_path, "rb") as f:
                    image_data = f.read()

            return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    def _create_prompt_with_images(
        self, batch: CharacterIdentificationBatch
    ) -> tuple[str, list[dict]]:
        """Create the prompt for character identification with images"""
        character_info = []
        images = []

        # Add character reference images
        for i, char in enumerate(batch.character_vault, 1):
            char_desc = f"Character {i} - ID: {char.char_id}, Name: {char.name}\n"
            char_desc += f"Core Traits: {', '.join(char.traits.core)}\n"
            char_desc += f"Supportive Traits: {', '.join(char.traits.supportive)}\n"
            char_desc += f"Age: {char.traits.age_band}, Skin: {char.traits.skin_tone}, Type: {char.traits.type}"
            character_info.append(char_desc)

            # Encode character reference image
            try:
                image_base64 = self._encode_image(char.ref_image)
                images.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}",
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Failed to encode character image for {char.char_id}: {e}"
                )

        crop_info = []
        # Add crop images
        for i, crop in enumerate(batch.crops):
            crop_desc = f"Crop {i+1} - ID: {crop.crop_id}"
            crop_info.append(crop_desc)

            # Encode crop image
            try:
                image_base64 = self._encode_image(crop.crop_url)
                images.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}",
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to encode crop image for {crop.crop_id}: {e}")

        prompt = f"""You are a precise character identifier specializing in facial and body feature analysis. Analyze the provided character crops and identify the best matching characters from the character vault.

Character References (images 1-{len(batch.character_vault)}):
{chr(10).join(character_info)}

Character Crops to Identify (images {len(batch.character_vault)+1}-{len(batch.character_vault)+len(batch.crops)}):
{chr(10).join(crop_info)}

Instructions:
- Compare each crop image with the character reference images
- Choose the best match among the provided characters, or "Unknown" if no match is found
- Focus EXCLUSIVELY on permanent facial and body characteristics:
  * Facial features: eye color, eye shape, nose shape, lip shape, facial structure
  * Hair characteristics: color, texture, length, style (but not temporary styling)
  * Skin tone and complexion
  * Facial hair: beard, mustache, stubble
  * Age-related features: wrinkles, facial structure
  * Unique facial marks: scars, birthmarks, moles
  * Body build and facial proportions

- IGNORE these temporary/variable traits:
  * Clothing, accessories, jewelry
  * Makeup, temporary styling
  * Mood, expression, emotion
  * Lighting, shadows, image quality
  * Background, context, props

- Provide a confidence score between 0.0 and 1.0 based on how well the permanent characteristics match
- In your reasoning, cite 1-3 specific permanent traits that led to your identification

Analyze each crop carefully and provide accurate identifications based solely on permanent physical characteristics."""

        return prompt, images

    def _create_prompt(self, batch: CharacterIdentificationBatch) -> str:
        """Create the prompt for character identification (for backwards compatibility)"""
        prompt, _ = self._create_prompt_with_images(batch)
        return prompt

    def _parse_llm_response(
        self, structured_response: CharacterIdentificationBatchResponse
    ) -> list[CharacterIdentificationResponse]:
        """Parse structured LLM response into CharacterIdentificationResponse objects"""
        try:
            results = []

            for crop_identification in structured_response.crops:
                result = CharacterIdentificationResponse(
                    crop_id=crop_identification.crop_id,
                    pred_char_id=crop_identification.pred_char_id,
                    confidence=crop_identification.confidence,
                    reason=crop_identification.reason,
                )
                results.append(result)

            logger.info(
                f"Successfully parsed {len(results)} character identifications from structured response"
            )
            return results

        except Exception as e:
            logger.error(f"Failed to parse structured LLM response: {e}")
            raise ValueError(f"Invalid structured response format: {e}") from e

    def _make_llm_call(
        self, prompt: str, images: list[dict]
    ) -> CharacterIdentificationBatchResponse:
        """Make a single LLM call with structured output and images
        Returns:
            CharacterIdentificationBatchResponse: The parsed response from the LLM
        Raises:
            openai.APIError: If there's an OpenAI API error
            ValueError: If no parsed output is received
            RuntimeError: If an unexpected error occurs
        """
        try:
            # Prepare content with text and images
            content = [{"type": "input_text", "text": prompt}]
            content.extend(images)

            # Use OpenAI's structured outputs API (same as working old version)
            response = self.client.responses.parse(
                model=self.config["model"],
                input=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                text_format=CharacterIdentificationBatchResponse,
            )

            # Extract the parsed output (same as working old version)
            if response.output_parsed:
                # The output_parsed is already of type CharacterIdentificationBatchResponse
                # but we need to ensure type safety
                parsed_output = response.output_parsed
                if isinstance(parsed_output, CharacterIdentificationBatchResponse):
                    # Explicit type assertion to help the type checker
                    result: CharacterIdentificationBatchResponse = parsed_output
                    return result
                else:
                    # If somehow the type is wrong, reconstruct it using Pydantic's parse_obj
                    # Explicit type annotation to ensure the type checker understands this returns CharacterIdentificationBatchResponse
                    validated_response: CharacterIdentificationBatchResponse = (
                        CharacterIdentificationBatchResponse.model_validate(
                            parsed_output
                        )
                    )
                    return validated_response
            else:
                raise ValueError("No parsed output received from OpenAI")

        except Exception as e:
            logger.error(f"Unexpected error in LLM call: {e}")
            raise

        # This should never be reached, but just in case
        raise RuntimeError("Unexpected error in LLM call")

    def identify_characters(
        self, batch: CharacterIdentificationBatch
    ) -> list[CharacterIdentificationResponse]:
        """Identify characters in a batch of crops with retry logic"""

        prompt, images = self._create_prompt_with_images(batch)
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(
                    f"Attempting character identification (attempt {attempt + 1}/{self.max_retries + 1})"
                )

                structured_response = self._make_llm_call(prompt, images)
                results = self._parse_llm_response(structured_response)

                # Validate that we got results for all crops
                expected_crop_ids = {crop.crop_id for crop in batch.crops}
                received_crop_ids = {result.crop_id for result in results}

                if expected_crop_ids != received_crop_ids:
                    missing = expected_crop_ids - received_crop_ids
                    extra = received_crop_ids - expected_crop_ids
                    logger.warning(
                        f"Crop ID mismatch. Missing: {missing}, Extra: {extra}"
                    )

                logger.info(
                    f"Successfully identified characters for {len(results)} crops"
                )
                return results

            except Exception as e:
                if (
                    "rate limit" in str(e).lower()
                    or "too many requests" in str(e).lower()
                ):
                    # Exponential backoff for rate limiting
                    wait_time = min(
                        self.retry_timeout * (2**attempt), 60
                    )  # Max 60 seconds
                    logger.warning(
                        f"Rate limited. Waiting {wait_time}s before retry..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Re-raise if it's not a rate limit error
                    raise
            except ValueError as e:
                elapsed_time = time.time() - start_time

                if attempt == self.max_retries or elapsed_time >= self.retry_timeout:
                    logger.error(
                        f"Failed to identify characters after {attempt + 1} attempts: {e}"
                    )
                    raise

                # Calculate delay with exponential backoff
                delay = self.exponential_backoff_base**attempt
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        # This should never be reached, but just in case
        raise RuntimeError("Unexpected error in character identification")

    def process_batch(
        self, batch: CharacterIdentificationBatch
    ) -> list[CharacterIdentificationResponse]:
        """Process a batch of crops for character identification"""
        logger.info(f"Processing batch {batch.batch_id} with {len(batch.crops)} crops")

        try:
            results = self.identify_characters(batch)

            # Update the crops with identification results
            for result in results:
                for crop in batch.crops:
                    if crop.crop_id == result.crop_id:
                        crop.pred_char_id = result.pred_char_id
                        crop.confidence = result.confidence
                        crop.reason = result.reason
                        break

            logger.info(f"Successfully processed batch {batch.batch_id}")
            return results

        except Exception as e:
            logger.error(f"Failed to process batch {batch.batch_id}: {e}")
            # Return unknown results for all crops in case of failure
            fallback_results = []
            for crop in batch.crops:
                fallback_result = CharacterIdentificationResponse(
                    crop_id=crop.crop_id,
                    pred_char_id="Unknown",
                    confidence=0.0,
                    reason="Identification failed due to system error",
                )
                fallback_results.append(fallback_result)

                # Update crop with fallback values
                crop.pred_char_id = "Unknown"
                crop.confidence = 0.0
                crop.reason = "Identification failed due to system error"

            return fallback_results
