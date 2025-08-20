import os
import json
import base64
import requests
from typing import List, Dict, Any, Optional
from openai import OpenAI
from models import Character, VideoOutput, CharacterVerification, CritiqueAgentResult
from config import Config


class CritiqueAgent:
    """
    Agent that uses OpenAI's GPT models to verify if characters in images appear in video clips
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the critique agent with OpenAI API
        
        Args:
            openai_api_key: OpenAI API key (defaults to environment variable)
        """
        self.openai_api_key = openai_api_key or Config.LLM_API_KEY
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set LLM_API_KEY environment variable or pass it to constructor.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.openai_api_key)
    
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
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_url}: {e}")
            return ""
    
    def create_verification_prompt(self, character: Character, video_clip_url: str) -> str:
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
4. Provide your analysis in the following JSON format:

{{
    "is_present": true/false,
    "confidence_score": 0.0-1.0,
    "reasoning": "Detailed explanation of your analysis",
    "timestamp_analysis": {{
        "first_appearance": "MM:SS or description",
        "total_screen_time": "MM:SS or description",
        "key_scenes": ["Scene 1 description", "Scene 2 description"]
    }}
}}

IMPORTANT:
- Be thorough in your analysis
- Consider visual similarities, clothing, facial features, and context
- If the character appears but looks different (e.g., different clothing, lighting), still mark as present
- Confidence score should reflect your certainty (1.0 = completely certain, 0.0 = completely uncertain)
- Provide specific reasoning for your decision
- Respond ONLY with valid JSON in the exact format specified above
"""
        return prompt.strip()
    
    def call_openai_for_verification(self, prompt: str, image_base64: str) -> Dict[str, Any]:
        """
        Call OpenAI's API to perform character verification using the official client
        
        Args:
            prompt: Text prompt for OpenAI
            image_base64: Base64 encoded reference image
            
        Returns:
            OpenAI API response parsed as a dictionary
        """
        try:
            # Use OpenAI's official client for chat completions
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=Config.LLM_MAX_TOKENS,
                temperature=Config.LLM_TEMPERATURE
            )
            
            # Extract the response content
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                
                # Try to parse JSON from the response
                try:
                    # Look for JSON content in the response
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_content = content[json_start:json_end]
                        parsed_response = json.loads(json_content)
                        
                        # Validate required fields
                        if self._validate_verification_response(parsed_response):
                            return parsed_response
                        else:
                            print("Warning: Invalid response format from OpenAI, using fallback parsing")
                            return self._parse_unstructured_response(content)
                    else:
                        # Fallback: try to extract structured information
                        return self._parse_unstructured_response(content)
                        
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}, using fallback parsing")
                    return self._parse_unstructured_response(content)
            
            return {"error": "Failed to get valid response from OpenAI API"}
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {"error": str(e)}
    
    def _validate_verification_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate that the OpenAI response contains required fields
        
        Args:
            response: Parsed response from OpenAI
            
        Returns:
            True if response is valid, False otherwise
        """
        required_fields = ["is_present", "confidence_score", "reasoning"]
        
        for field in required_fields:
            if field not in response:
                print(f"Missing required field: {field}")
                return False
        
        # Validate data types
        if not isinstance(response["is_present"], bool):
            print("is_present must be a boolean")
            return False
        
        if not isinstance(response["confidence_score"], (int, float)):
            print("confidence_score must be a number")
            return False
        
        if not isinstance(response["reasoning"], str):
            print("reasoning must be a string")
            return False
        
        # Validate confidence score range
        if not (0.0 <= response["confidence_score"] <= 1.0):
            print("confidence_score must be between 0.0 and 1.0")
            return False
        
        return True
    
    def _parse_unstructured_response(self, content: str) -> Dict[str, Any]:
        """
        Parse unstructured OpenAI response to extract verification information
        
        Args:
            content: Raw text response from OpenAI
            
        Returns:
            Structured verification data
        """
        # Default values
        result = {
            "is_present": False,
            "confidence_score": 0.5,
            "reasoning": content,
            "timestamp_analysis": {}
        }
        
        # Try to extract presence information
        content_lower = content.lower()
        if any(word in content_lower for word in ['yes', 'present', 'appears', 'visible', 'seen', 'true']):
            result["is_present"] = True
        elif any(word in content_lower for word in ['no', 'not present', 'absent', 'not visible', 'not seen', 'false']):
            result["is_present"] = False
        
        # Try to extract confidence score
        import re
        confidence_match = re.search(r'confidence[:\s]*(\d+\.?\d*)', content_lower)
        if confidence_match:
            try:
                score = float(confidence_match.group(1))
                if 0 <= score <= 1:
                    result["confidence_score"] = score
            except ValueError:
                pass
        
        # Try to extract timestamp information
        timestamp_patterns = {
            "first_appearance": r'first.*?appear.*?(\d{1,2}:\d{2})',
            "total_screen_time": r'total.*?time.*?(\d{1,2}:\d{2})',
            "key_scenes": r'scene.*?(\d{1,2}:\d{2})'
        }
        
        for key, pattern in timestamp_patterns.items():
            matches = re.findall(pattern, content_lower)
            if matches:
                result["timestamp_analysis"][key] = matches[0] if key != "key_scenes" else matches
        
        return result
    
    def verify_character_in_video(self, character: Character, video_clip_url: str) -> CharacterVerification:
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
                timestamp_analysis={}
            )
        
        # Create verification prompt
        prompt = self.create_verification_prompt(character, video_clip_url)
        
        # Call OpenAI API for verification
        openai_response = self.call_openai_for_verification(prompt, image_base64)
        
        # Handle errors
        if "error" in openai_response:
            return CharacterVerification(
                character_name=character.name,
                character_image=character.image,
                video_clip_url=video_clip_url,
                is_present=False,
                confidence_score=0.0,
                reasoning=f"OpenAI API error: {openai_response['error']}",
                timestamp_analysis={}
            )
        
        # Extract timestamp analysis
        timestamp_analysis = openai_response.get("timestamp_analysis", {})
        
        return CharacterVerification(
            character_name=character.name,
            character_image=character.image,
            video_clip_url=video_clip_url,
            is_present=openai_response.get("is_present", False),
            confidence_score=openai_response.get("confidence_score", 0.5),
            reasoning=openai_response.get("reasoning", "No reasoning provided"),
            timestamp_analysis=timestamp_analysis
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
            verification = self.verify_character_in_video(character, video_output.clip_url)
            character_verifications.append(verification)
        
        # Calculate overall accuracy
        if character_verifications:
            # For now, we'll use a simple metric - you can customize this
            # based on your evaluation criteria
            present_characters = sum(1 for v in character_verifications if v.is_present)
            total_characters = len(character_verifications)
            overall_accuracy = present_characters / total_characters if total_characters > 0 else 0.0
        else:
            overall_accuracy = 0.0
        
        # Generate evaluation notes
        evaluation_notes = self._generate_evaluation_notes(character_verifications, overall_accuracy)
        
        return CritiqueAgentResult(
            project_id=video_output.project_id,
            video_output=video_output,
            character_verifications=character_verifications,
            overall_accuracy=overall_accuracy,
            evaluation_notes=evaluation_notes,
            llm_metadata={
                "verification_count": len(character_verifications),
                "llm_provider": "OpenAI",
                "model": Config.LLM_MODEL
            }
        )
    
    def _generate_evaluation_notes(self, verifications: List[CharacterVerification], accuracy: float) -> str:
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
            notes += f"{verification.character_name}: {status} (Confidence: {confidence})\n"
            notes += f"  Reasoning: {verification.reasoning}\n\n"
        
        return notes.strip()