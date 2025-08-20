import re
import json
from typing import Tuple, Dict, Any, List


def parse_character_prompt(prompt: str) -> Tuple[str, str]:
    """
    Parse character prompt in format: "Close up character portrait of <char name>: <char description>"
    Returns (character_name, character_description)
    """
    pattern = r"Close up character portrait of ([^:]+):\s*(.+)"
    match = re.match(pattern, prompt.strip())
    
    if match:
        name = match.group(1).strip()
        description = match.group(2).strip()
        return name, description
    
    return "", ""


def parse_metadata_json(metadata_str: str) -> List[Dict[str, Any]]:
    """
    Parse metadata column as JSON array with name/value pairs
    """
    try:
        return json.loads(metadata_str)
    except json.JSONDecodeError:
        return []


def parse_enhanced_prompt(enhanced_prompt_str: str) -> Dict[str, Any]:
    """
    Parse enhanced_prompt as JSON string (second level parsing)
    """
    try:
        return json.loads(enhanced_prompt_str)
    except json.JSONDecodeError:
        return {}


def find_best_character_match(subject_name: str, available_characters: List[str]) -> str:
    """
    Find the best character match using fuzzy matching
    Handles cases like "Baby Ganesha" matching "Ganesha"
    """
    if not subject_name or not available_characters:
        return ""
    
    subject_name_lower = subject_name.lower().strip()
    
    # First try exact match
    if subject_name in available_characters:
        return subject_name
    
    # Try case-insensitive exact match
    for char_name in available_characters:
        if char_name.lower().strip() == subject_name_lower:
            return char_name
    
    # Try partial matching (e.g., "Baby Ganesha" contains "Ganesha")
    for char_name in available_characters:
        char_name_lower = char_name.lower().strip()
        
        # Check if character name is contained in subject name
        if char_name_lower in subject_name_lower:
            return char_name
        
        # Check if subject name is contained in character name
        if subject_name_lower in char_name_lower:
            return char_name
    
    # Try word-based matching (e.g., "Baby Ganesha" -> "Ganesha")
    subject_words = subject_name_lower.split()
    for char_name in available_characters:
        char_name_lower = char_name.lower().strip()
        char_words = char_name_lower.split()
        
        # Check if any word from character name appears in subject name
        for char_word in char_words:
            if char_word in subject_words:
                return char_name
        
        # Check if any word from subject name appears in character name
        for subject_word in subject_words:
            if subject_word in char_words:
                return char_name
    
    # Try removing common prefixes/suffixes
    common_prefixes = ['baby', 'little', 'young', 'old', 'big', 'small']
    common_suffixes = ['jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv']
    
    # Remove common prefixes
    for prefix in common_prefixes:
        if subject_name_lower.startswith(prefix + ' '):
            cleaned_subject = subject_name_lower[len(prefix + ' '):].strip()
            for char_name in available_characters:
                if char_name.lower().strip() == cleaned_subject:
                    return char_name
    
    # Remove common suffixes
    for suffix in common_suffixes:
        if subject_name_lower.endswith(' ' + suffix):
            cleaned_subject = subject_name_lower[:-len(' ' + suffix)].strip()
            for char_name in available_characters:
                if char_name.lower().strip() == cleaned_subject:
                    return char_name
    
    return ""


def extract_subjects_from_metadata(metadata: List[Dict[str, Any]]) -> List[str]:
    """
    Extract subjects from metadata.enhanced_prompt.subjects or metadata.enhanced_prompt.subjects_found
    metadata is a list of dicts with 'name' and 'value' keys
    """
    try:
        # Find the ENHANCED_PROMPT item
        enhanced_prompt_item = None
        for item in metadata:
            if isinstance(item, dict) and item.get('name') == 'ENHANCED_PROMPT':
                enhanced_prompt_item = item
                break
        
        if not enhanced_prompt_item:
            return []
        
        # Parse the enhanced prompt value
        enhanced_prompt_data = parse_enhanced_prompt(enhanced_prompt_item.get('value', '{}'))
        
        # Look for both 'subjects' and 'subjects_found' fields
        subjects = enhanced_prompt_data.get('subjects', [])
        if not subjects:
            subjects = enhanced_prompt_data.get('subjects_found', [])
        
        # Extract character names from subjects
        character_names = []
        for subject in subjects:
            if isinstance(subject, dict) and 'name' in subject:
                character_names.append(subject['name'])
            elif isinstance(subject, str):
                character_names.append(subject)
        
        return character_names
    except Exception as e:
        print(f"Error extracting subjects: {e}")
        return [] 
    


def save_results_to_json(all_results, output_filename="story_evals_results.json"):
    """
    Save all results to a JSON file
    """
    # Convert all results to a structured format
    json_output = {
        "summary": {
            "total_projects": len(all_results),
            "total_videos": sum(len(videos) for videos in all_results.values())
        },
        "projects": {}
    }
    
    for project_id, video_outputs in all_results.items():
        # Only include projects that have valid video outputs
        if video_outputs and any(video.clip_url for video in video_outputs):
            valid_videos = []
            
            for video_output in video_outputs:
                # Skip videos with empty clip URLs
                if not video_output.clip_url:
                    continue
                
                # Skip videos with empty character arrays
                if not video_output.characters:
                    continue
                    
                # Convert to dictionary for JSON output
                output_dict = {
                    "ProjectID": video_output.project_id,
                    "ClipUrl": video_output.clip_url,
                    "Prompt": video_output.prompt,
                    "EnhancedPrompt": video_output.enhanced_prompt,
                    "ReferenceImage": video_output.reference_image,
                    "ThumbnailImage": video_output.thumbnail_image,
                    "Characters": [
                        {
                            "Name": char.name,
                            "Image": char.image,
                            "Description": char.description
                        }
                        for char in video_output.characters
                    ]
                }
                
                valid_videos.append(output_dict)
            
            # Only add the project if it has valid videos after filtering
            if valid_videos:
                json_output["projects"][project_id] = valid_videos
    
    # Save to JSON file
    with open(output_filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_output, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_filename}")
