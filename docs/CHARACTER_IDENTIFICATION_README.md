# Character Identification System

A comprehensive system for identifying characters in video face crops using LLM-based analysis and image processing.

## Features

- **LLM-based Character Identification**: Uses OpenAI models to identify characters in face crops
- **Structured Outputs**: Pydantic models ensure schema adherence and reliable parsing
- **Batch Processing**: Configurable batch sizes (4-8 crops per call) for efficient processing
- **Parallel Processing**: Process multiple shots concurrently for better performance
- **Image Processing**: Download and overlay crop IDs on images with red text and black outline
- **Robust Error Handling**: Exponential backoff retry logic with configurable timeouts
- **Progress Tracking**: Track completion status and timing
- **Result Aggregation**: Combine results from all batches
- **Flexible Configuration**: Easy configuration via environment variables or config file

## Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Input Data       │    │   Pipeline          │    │   Output            │
│                     │    │                     │    │                     │
│ • Character Vault  │───▶│ • Validation        │───▶│ • Identified Crops  │
│ • Video Shots      │    │ • Batching          │    │ • Processed Images  │
│ • Face Crops       │    │ • LLM Processing    │    │ • JSON Results      │
└─────────────────────┘    │ • Image Processing  │    └─────────────────────┘
                           └─────────────────────┘
```

## Components

### 1. Character Identifier (`character_identifier.py`)
- LLM-based character identification
- **Structured outputs using Pydantic models for reliable parsing**
- Retry logic with exponential backoff
- Response validation and parsing

### 2. Image Processor (`image_processor.py`)
- Download crop images from URLs
- Add crop ID overlays with custom styling
- Batch processing with progress tracking
- Automatic cleanup of old images

### 3. Batch Processor (`batch_processor.py`)
- Configurable batch sizes (4-8 crops)
- Parallel processing of multiple shots
- Progress tracking and ETA estimation
- Flexible processing strategies

### 4. Main Pipeline (`character_identification_pipeline.py`)
- Orchestrates all components
- Input validation and parsing
- Complete workflow management
- Output generation and saving

## Structured Outputs

The system uses OpenAI's structured outputs feature with Pydantic models to ensure reliable and consistent responses:

### Pydantic Models
```python
class CropIdentification(BaseModel):
    crop_id: str = Field(description="The unique identifier for the crop")
    pred_char_id: str = Field(description="The predicted character ID or 'Unknown'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(description="Justification citing visible core traits")

class CharacterIdentificationBatchResponse(BaseModel):
    crops: List[CropIdentification] = Field(description="List of identifications")
```

### Benefits
- **Schema Validation**: Automatic validation of LLM responses
- **Type Safety**: Guaranteed data types and structure
- **Reliability**: No more JSON parsing errors or malformed responses
- **Consistency**: Uniform output format across all API calls
- **Error Handling**: Clear error messages for validation failures

## Installation

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Or using uv (recommended)
uv sync
```

### 2. Set Environment Variables
```bash
# Required: OpenAI API Key
export OPENAI_API_KEY="your-api-key-here"

# Optional: Override defaults
export CHAR_ID_LLM_MODEL="gpt-4o-mini"
export BATCH_SIZE="6"
export MAX_PARALLEL_SHOTS="4"
export OUTPUT_IMAGE_DIR="data/processed_crops"
```

### 3. Verify Configuration
```python
from config import Config
print(Config.get_character_identification_config())
```

## Configuration

### Character Identification
- `CHAR_ID_LLM_MODEL`: LLM model for identification (default: "gpt-4o-mini")
- `CHAR_ID_MAX_TOKENS`: Maximum tokens per request (default: 1500)
- `CHAR_ID_TEMPERATURE`: LLM temperature (default: 0.1)
- `CHAR_ID_MAX_RETRIES`: Maximum retry attempts (default: 3)
- `CHAR_ID_RETRY_TIMEOUT`: Total timeout for retries (default: 30s)
- `CHAR_ID_EXPONENTIAL_BACKOFF_BASE`: Backoff multiplier (default: 2.0)

### Batch Processing
- `BATCH_SIZE`: Crops per batch (default: 6, range: 4-8)
- `MAX_PARALLEL_SHOTS`: Maximum concurrent shots (default: 4)

### Image Processing
- `OUTPUT_IMAGE_DIR`: Directory for processed images (default: "data/processed_crops")
- `CROP_ID_FONT_SIZE`: Font size for crop ID overlay (default: 24)
- `CROP_ID_COLOR`: Text color (default: "red")
- `CROP_ID_OUTLINE_COLOR`: Outline color (default: "black")
- `CROP_ID_OUTLINE_WIDTH`: Outline thickness (default: 3)

## Usage

### Basic Usage

```python
from character_identification_pipeline import CharacterIdentificationPipeline

# Initialize pipeline
pipeline = CharacterIdentificationPipeline()

# Process data
result = pipeline.process(input_data, process_images=True)

# Save results
output_path = pipeline.save_results(result)
```

### Input Data Format

```json
{
  "character_vault": [
    {
      "char_id": "GIRL_A",
      "name": "GIRL_A",
      "ref_image": "https://example.com/rina1.jpg",
      "traits": {
        "core": ["green eyes", "long blond wavy hair", "mole under left eye"],
        "supportive": ["heart-shaped face"],
        "volatile": ["hoodie", "earrings"],
        "age_band": "young_adult",
        "skin_tone": "light",
        "type": "human",
        "notes": []
      }
    }
  ],
  "shots": [
    {
      "keyframes_ms": [0, 420, 840],
      "crops": {
        "420": [
          {
            "crop_id": "c_shot0001_t0420_0",
            "bbox_norm": [0.31, 0.18, 0.14, 0.22],
            "crop_url": "https://example.com/shot0001_t0420_face0.jpg",
            "detector": "stylized_face_v2",
            "face_conf": 0.98,
            "quality": {
              "blur": 142.3,
              "pose": {"yaw": 6.1, "pitch": -2.3}
            }
          }
        ]
      },
      "shot_id": "shot_0001"
    }
  ]
}
```

### Output Format

The system adds three new fields to each crop:
- `pred_char_id`: Identified character ID or "Unknown"
- `confidence`: Confidence score (0.0-1.0)
- `reason`: Justification citing visible core traits

```json
{
  "crop_id": "c_shot0001_t0420_0",
  "bbox_norm": [0.31, 0.18, 0.14, 0.22],
  "crop_url": "https://example.com/shot0001_t0420_face0.jpg",
  "detector": "stylized_face_v2",
  "face_conf": 0.98,
  "pred_char_id": "GIRL_A",
  "confidence": 0.91,
  "reason": "Long blond wavy hair and mole under left eye match GIRL_A.",
  "quality": {
    "blur": 142.3,
    "pose": {"yaw": 6.1, "pitch": -2.3}
  }
}
```

## Testing

### Run Test Script
```bash
python test_character_identification.py
```

### Test with Real Data
1. Update sample URLs in test script to real image URLs
2. Set `process_images=True` to enable image processing
3. Ensure OpenAI API key is configured
4. Run the test script

## Performance Considerations

### Batch Size Optimization
- **Small batches (4-5)**: Better for complex characters, higher accuracy
- **Medium batches (6)**: Good balance of efficiency and accuracy (default)
- **Large batches (7-8)**: Better for simple characters, higher throughput

### Parallel Processing
- Adjust `MAX_PARALLEL_SHOTS` based on your system resources
- Monitor memory usage with large datasets
- Consider API rate limits for your OpenAI plan

### Image Processing
- Images are processed sequentially to avoid overwhelming the system
- Large numbers of images may take significant time
- Consider using `process_images=False` for testing without image processing

## Error Handling

### LLM Failures
- Automatic retry with exponential backoff
- Fallback to "Unknown" classification if all retries fail
- Detailed logging for debugging

### Image Processing Failures
- Individual crop failures don't stop the pipeline
- Failed downloads are logged but processing continues
- Graceful degradation for network issues

### Data Validation
- Comprehensive input validation
- Clear error messages for malformed data
- Automatic fallbacks for missing optional fields

## Monitoring and Logging

### Log Levels
- `INFO`: General progress and completion status
- `WARNING`: Non-critical issues (e.g., crop ID mismatches)
- `ERROR`: Critical failures that affect processing
- `DEBUG`: Detailed processing information

### Progress Tracking
- Real-time progress updates for large datasets
- ETA estimation based on current processing speed
- Batch-level and shot-level completion tracking

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Verify API key is set correctly
   - Check API rate limits and billing
   - Ensure model name is valid

2. **Image Download Failures**
   - Verify image URLs are accessible
   - Check network connectivity
   - Ensure URLs return valid image files

3. **Memory Issues**
   - Reduce `MAX_PARALLEL_SHOTS`
   - Process smaller batches
   - Monitor system resources

4. **Performance Issues**
   - Adjust batch size based on character complexity
   - Optimize parallel processing settings
   - Consider using faster LLM models

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Reference

### CharacterIdentificationPipeline

#### `process(input_data, process_images=True)`
Main processing method that orchestrates the entire pipeline.

**Parameters:**
- `input_data`: Dictionary containing character vault and shots data
- `process_images`: Boolean to enable/disable image processing

**Returns:**
- `CharacterIdentificationResult` object with identification results

#### `save_results(result, output_path=None)`
Save results to JSON file.

**Parameters:**
- `result`: CharacterIdentificationResult object
- `output_path`: Optional custom output path

**Returns:**
- Path to saved results file

### CharacterIdentifier

#### `identify_characters(batch)`
Identify characters in a batch of crops.

**Parameters:**
- `batch`: CharacterIdentificationBatch object

**Returns:**
- List of CharacterIdentificationResponse objects

### ImageProcessor

#### `process_crops_batch(crops)`
Process multiple crops with progress tracking.

**Parameters:**
- `crops`: List of Crop objects

**Returns:**
- List of tuples (Crop, output_path)

### BatchProcessor

#### `process_shots_parallel(shots, character_vault, identifier_func)`
Process multiple shots in parallel.

**Parameters:**
- `shots`: List of Shot objects
- `character_vault`: List of CharacterVaultEntry objects
- `identifier_func`: Function to process batches

**Returns:**
- List of processed Shot objects

## Contributing

1. Follow the existing code style and patterns
2. Add comprehensive error handling
3. Include logging for debugging
4. Update configuration as needed
5. Add tests for new functionality

## License

This project is part of the StoryEvals system. Please refer to the main project license.
