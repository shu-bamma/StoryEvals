import csv
import json
from typing import Dict, List, Optional, Any
from models import Character, VideoOutput
from utils import parse_character_prompt, parse_metadata_json, extract_subjects_from_metadata, find_best_character_match, parse_enhanced_prompt


class DataProcessor:
    def __init__(self):
        self.characters_lookup: Dict[str, Character] = {}
    
    def load_characters_from_images(self, project_id: str) -> Dict[str, Character]:
        """
        Load characters from saved_images.csv for a specific project ID
        """
        characters = {}
        
        try:
            with open('data/saved_images.csv', 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    if row.get('show_id') == project_id:
                        prompt = row.get('prompt', '')
                        image_url = row.get('image_url', '')
                        
                        print(f"  Checking row for project {project_id}: prompt='{prompt[:100] if prompt else 'NO_PROMPT'}...' image_url='{image_url[:50] if image_url else 'NO_URL'}...'")
                        
                        if prompt and 'Close up character portrait of' in prompt:
                            print(f"  ✓ Found character prompt: {prompt[:100]}...")
                            name, description = parse_character_prompt(prompt)
                            print(f"  Parsed name: '{name}', description length: {len(description) if description else 0}")
                            
                            if name and description:
                                character = Character(
                                    name=name,
                                    image=image_url,
                                    description=description
                                )
                                characters[name] = character
                                print(f"  ✓ Added character: {name}")
                            else:
                                print(f"  ✗ Failed to parse character: name='{name}', description='{description}'")
                        else:
                            print(f"  ✗ Not a character prompt: {prompt[:100] if prompt else 'NO_PROMPT'}...")
                                
        except Exception as e:
            print(f"Error loading characters for project {project_id}: {e}")
        
        print(f"  Total characters loaded for project {project_id}: {len(characters)}")
        return characters
    
    def get_enhanced_prompt_from_metadata(self, metadata: List[Dict[str, Any]]) -> str:
        """
        Extract the enhanced prompt value from metadata
        """
        try:
            for item in metadata:
                if isinstance(item, dict) and item.get('name') == 'ENHANCED_PROMPT':
                    return item.get('value', '{}')
            return '{}'
        except Exception:
            return '{}'
    
    def get_reference_image_from_metadata(self, metadata: List[Dict[str, Any]]) -> str:
        """
        Extract the reference image from metadata[0].value
        """
        try:
            if metadata and len(metadata) > 0 and isinstance(metadata[0], dict):
                return metadata[0].get('value', '')
            return ''
        except Exception:
            return ''
    
    def get_thumbnail_image_from_metadata(self, metadata: List[Dict[str, Any]]) -> str:
        """
        Extract the thumbnail image from metadata[1].value
        """
        try:
            if metadata and len(metadata) > 1 and isinstance(metadata[1], dict):
                return metadata[1].get('value', '')
            return ''
        except Exception:
            return ''
    
    def get_all_project_images(self, project_id: str) -> List[Dict[str, str]]:
        """
        Get all images for a project from saved_images.csv for debug purposes
        """
        images = []
        try:
            with open('data/saved_images.csv', 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get('show_id') == project_id:
                        images.append({
                            'prompt': row.get('prompt', ''),
                            'image_url': row.get('image_url', ''),
                            'show_id': row.get('show_id', '')
                        })
        except Exception as e:
            print(f"Error loading images for debug: {e}")
        return images
    
    def process_videos_for_project(self, project_id: str, characters: Dict[str, Character]) -> List[VideoOutput]:
        """
        Process videos from saved_videos.csv for a specific project ID
        """
        video_outputs = []
        
        try:
            with open('data/saved_videos.csv', 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    if row.get('show_id') == project_id:
                        prompt = row.get('prompt', '')
                        video_url = row.get('video_url', '')
                        metadata_str = row.get('metadata', '')
                        
                        # Skip rows that don't have a video_url
                        if not video_url or video_url.strip() == '':
                            continue
                        
                        # Parse metadata JSON
                        metadata = parse_metadata_json(metadata_str)
                        
                        # Extract enhanced prompt and subjects
                        enhanced_prompt = self.get_enhanced_prompt_from_metadata(metadata)
                        subjects = extract_subjects_from_metadata(metadata)
                        
                        # Extract reference and thumbnail images from metadata
                        reference_image = self.get_reference_image_from_metadata(metadata)
                        thumbnail_image = self.get_thumbnail_image_from_metadata(metadata)
                        
                        # Extract names from enhanced prompt metadata for better matching
                        enhanced_prompt_names = []
                        try:
                            # Clean the enhanced prompt string - remove markdown code blocks if present
                            cleaned_prompt = enhanced_prompt.strip()
                            if cleaned_prompt.startswith('```json'):
                                cleaned_prompt = cleaned_prompt[7:]  # Remove ```json
                            if cleaned_prompt.endswith('```'):
                                cleaned_prompt = cleaned_prompt[:-3]  # Remove ```
                            
                            enhanced_prompt_data = parse_metadata_json(cleaned_prompt)
                            if isinstance(enhanced_prompt_data, dict):
                                # Look for both 'subjects' and 'subjects_found' fields
                                enhanced_subjects = enhanced_prompt_data.get('subjects', [])
                                if not enhanced_subjects:
                                    enhanced_subjects = enhanced_prompt_data.get('subjects_found', [])
                                
                                print(f"  Enhanced prompt data keys: {list(enhanced_prompt_data.keys())}")
                                print(f"  Found 'subjects': {enhanced_prompt_data.get('subjects', [])}")
                                print(f"  Found 'subjects_found': {enhanced_prompt_data.get('subjects_found', [])}")
                                print(f"  Using enhanced_subjects: {enhanced_subjects}")
                                
                                for subject in enhanced_subjects:
                                    if isinstance(subject, dict) and 'name' in subject:
                                        enhanced_prompt_names.append(subject['name'])
                                    elif isinstance(subject, str):
                                        enhanced_prompt_names.append(subject)
                        except Exception as e:
                            pass
                        
                        # Use enhanced prompt names if available, otherwise fall back to original subjects
                        subjects_to_match = enhanced_prompt_names if enhanced_prompt_names else subjects
                        
                        # Debug: Print character names and subjects for matching
                        if subjects_to_match:
                            print(f"  Found subjects: {subjects_to_match}")
                            print(f"  Available characters: {list(characters.keys())}")
                        else:
                            print(f"  No subjects found to match")
                            print(f"  Available characters: {list(characters.keys())}")
                        
                        # Match subjects with characters using fuzzy matching
                        matched_characters = []
                        for subject_name in subjects_to_match:
                            print(f"  Attempting to match subject: '{subject_name}'")
                            best_match = find_best_character_match(subject_name, list(characters.keys()))
                            if best_match:
                                matched_characters.append(characters[best_match])
                                print(f"  ✓ Matched '{subject_name}' with character '{best_match}'")
                            else:
                                print(f"  ✗ No match for '{subject_name}'")
                        
                        print(f"  Total matched characters: {len(matched_characters)}")
                        
                        # Prepare debug information if characters array is empty
                        debug_info = None
                        if not matched_characters:
                            # Extract names from enhanced prompt metadata
                            enhanced_prompt_names = []
                            try:
                                # Clean the enhanced prompt string - remove markdown code blocks if present
                                cleaned_prompt = enhanced_prompt.strip()
                                if cleaned_prompt.startswith('```json'):
                                    cleaned_prompt = cleaned_prompt[7:]  # Remove ```json
                                if cleaned_prompt.endswith('```'):
                                    cleaned_prompt = cleaned_prompt[:-3]  # Remove ```
                                
                                enhanced_prompt_data = parse_metadata_json(cleaned_prompt)
                                if isinstance(enhanced_prompt_data, dict):
                                    # Look for both 'subjects' and 'subjects_found' fields
                                    subjects = enhanced_prompt_data.get('subjects', [])
                                    if not subjects:
                                        subjects = enhanced_prompt_data.get('subjects_found', [])
                                    
                                    for subject in subjects:
                                        if isinstance(subject, dict) and 'name' in subject:
                                            enhanced_prompt_names.append(subject['name'])
                                        elif isinstance(subject, str):
                                            enhanced_prompt_names.append(subject)
                                
                                # Also try to extract from the original subjects if enhanced_prompt parsing fails
                                if not enhanced_prompt_names and subjects:
                                    for subject in subjects:
                                        if isinstance(subject, dict) and 'name' in subject:
                                            enhanced_prompt_names.append(subject['name'])
                                        elif isinstance(subject, str):
                                            enhanced_prompt_names.append(subject)
                                
                                # If still no names found, try to parse the raw enhanced prompt directly
                                if not enhanced_prompt_names:
                                    try:
                                        # Try to find subjects in the raw string using regex as fallback
                                        import re
                                        subject_pattern = r'"name":\s*"([^"]+)"'
                                        matches = re.findall(subject_pattern, enhanced_prompt)
                                        if matches:
                                            enhanced_prompt_names.extend(matches)
                                    except Exception:
                                        pass
                                        
                            except Exception as e:
                                enhanced_prompt_names = [f"Error parsing: {e}"]
                            
                            # Get character names from the first step (directory)
                            directory_character_names = list(characters.keys())
                            
                            debug_info = {
                                'all_project_images': self.get_all_project_images(project_id),
                                'subjects_found': subjects,
                                'available_characters': directory_character_names,
                                'metadata': metadata,
                                'enhanced_prompt_names': enhanced_prompt_names,
                                'directory_character_names': directory_character_names,
                                'raw_enhanced_prompt': enhanced_prompt,
                                'cleaned_enhanced_prompt': cleaned_prompt if 'cleaned_prompt' in locals() else None,
                                'parsed_enhanced_prompt': enhanced_prompt_data if 'enhanced_prompt_data' in locals() else None
                            }
                        
                        # Create video output
                        video_output = VideoOutput(
                            project_id=project_id,
                            clip_url=video_url,
                            prompt=prompt or "raw prompt",
                            enhanced_prompt=enhanced_prompt or "JSON prompt sent to veo3",
                            reference_image=reference_image or "image fed to vid model",
                            thumbnail_image=thumbnail_image or "idk",
                            characters=matched_characters,
                            debug=debug_info
                        )
                        
                        video_outputs.append(video_output)
                        
        except Exception as e:
            print(f"Error processing videos for project {project_id}: {e}")
        
        return video_outputs
    
    def process_project(self, project_id: str) -> List[VideoOutput]:
        """
        Process a single project ID to generate video outputs
        """
        print(f"Processing project: {project_id}")
        
        # Step 1: Load characters from images
        characters = self.load_characters_from_images(project_id)
        print(f"Found {len(characters)} characters for project {project_id}")
        
        # Step 2: Process videos and match with characters
        video_outputs = self.process_videos_for_project(project_id, characters)
        print(f"Generated {len(video_outputs)} video outputs for project {project_id}")
        
        return video_outputs
    
    def process_all_projects(self) -> Dict[str, List[VideoOutput]]:
        """
        Process all project IDs from unique_proj_ids.csv
        """
        all_results = {}
        
        try:
            with open('data/unique_proj_ids.csv', 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    project_id = row.get('show_id')
                    if project_id:
                        video_outputs = self.process_project(project_id)
                        all_results[project_id] = video_outputs
                        
        except Exception as e:
            print(f"Error reading project IDs: {e}")
        
        return all_results