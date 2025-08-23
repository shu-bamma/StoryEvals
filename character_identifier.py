import logging
import time

import openai
from openai import OpenAI

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
        self.client = OpenAI(api_key=self.config["api_key"])
        self.max_retries = self.config["max_retries"]
        self.retry_timeout = self.config["retry_timeout"]
        self.exponential_backoff_base = self.config["exponential_backoff_base"]

    def _create_prompt(self, batch: CharacterIdentificationBatch) -> str:
        """Create the prompt for character identification"""
        character_info = []
        for char in batch.character_vault:
            char_desc = f"Character ID: {char.char_id}\n"
            char_desc += f"Name: {char.name}\n"
            char_desc += f"Core Traits: {', '.join(char.traits.core)}\n"
            char_desc += f"Supportive Traits: {', '.join(char.traits.supportive)}\n"
            char_desc += f"Age: {char.traits.age_band}, Skin: {char.traits.skin_tone}, Type: {char.traits.type}\n"
            character_info.append(char_desc)

        crop_info = []
        for crop in batch.crops:
            crop_desc = f"Crop ID: {crop.crop_id}\n"
            crop_desc += f"URL: {crop.crop_url}\n"
            crop_info.append(crop_desc)

        prompt = f"""You are a precise character identifier. Analyze the provided face crops and identify the best matching characters from the character vault.

Character Vault:
{chr(10).join(character_info)}

Face Crops to Identify:
{chr(10).join(crop_info)}

Instructions:
- Choose the best match among the provided characters_vault, or "Unknown" if no match is found
- Cite 1–3 visible CORE traits in your reasoning (ignore VOLATILE traits like clothing, accessories, mood)
- Provide a confidence score between 0.0 and 1.0 based on how well the visible traits match
- Focus on distinctive physical characteristics that are clearly visible in the crop

Analyze each crop carefully and provide accurate identifications with clear reasoning."""

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

    def _make_llm_call(self, prompt: str) -> CharacterIdentificationBatchResponse:
        """Make a single LLM call with structured output
        Returns:
            CharacterIdentificationBatchResponse: The parsed response from the LLM
        Raises:
            openai.APIError: If there's an OpenAI API error
            ValueError: If no parsed output is received
            RuntimeError: If an unexpected error occurs
        """
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
                text_format=CharacterIdentificationBatchResponse,
            )

            # Extract the parsed output
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

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            # Re-raise to maintain the function's contract - this function never returns Any
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLM call: {e}")
            # Re-raise to maintain the function's contract - this function never returns Any
            raise

        # This should never be reached, but just in case
        # This ensures the function always either returns the expected type or raises an exception
        # This function never returns Any - it either returns CharacterIdentificationBatchResponse or raises an exception
        raise RuntimeError("Unexpected error in LLM call")

    def identify_characters(
        self, batch: CharacterIdentificationBatch
    ) -> list[CharacterIdentificationResponse]:
        """Identify characters in a batch of crops with retry logic"""

        prompt = self._create_prompt(batch)
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(
                    f"Attempting character identification (attempt {attempt + 1}/{self.max_retries + 1})"
                )

                structured_response = self._make_llm_call(prompt)
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

            except (ValueError, openai.APIError) as e:
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
