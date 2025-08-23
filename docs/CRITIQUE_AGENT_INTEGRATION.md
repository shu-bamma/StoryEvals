# Critique Agent Integration with Character Identification Pipeline

This document describes the integration of the Critique Agent as a double-check mechanism for character identification results in the StoryEvals system.

## Overview

The Critique Agent integration provides an additional layer of verification after the main character identification process. It uses OpenAI's GPT models to analyze video clips and verify whether identified characters actually appear in the video, serving as a quality assurance mechanism.

## How It Works

### 1. Character Identification Phase
- The pipeline processes video shots and identifies characters using the main identification system
- Face crops are analyzed and matched against the character vault
- Initial confidence scores and reasoning are generated

### 2. Critique Agent Verification Phase (Double-Check)
- After character identification, the critique agent analyzes the same video clips
- It uses the character reference images and descriptions to verify presence
- Provides independent verification with confidence scores and reasoning
- Compares results with the initial identification for consistency

### 3. Result Integration
- Both sets of results are combined in the final output
- Metadata includes information about whether critique agent verification was performed
- Results can be compared to identify discrepancies or validate high-confidence identifications

## Configuration

The integration can be enabled/disabled via configuration:

```python
# In config.py
ENABLE_CRITIQUE_AGENT: bool = True  # Enable critique agent as double-check
```

## Data Flow

```
Input Data → Character Identification → Critique Agent Verification → Combined Results
     ↓              ↓                        ↓                        ↓
Character Vault → LLM Analysis → Video Analysis → Final Output with Both Results
```

## Output Structure

The final output includes:

- **Character Identification Results**: Original identification with crops, confidence scores, and reasoning
- **Critique Agent Results**: Verification results including:
  - Character presence verification
  - Confidence scores
  - Detailed reasoning
  - Timestamp analysis
  - Overall accuracy metrics

## Benefits

1. **Quality Assurance**: Double-check mechanism catches potential misidentifications
2. **Independent Verification**: Uses different analysis approach (video vs. crop analysis)
3. **Confidence Validation**: Helps validate high-confidence identifications
4. **Discrepancy Detection**: Identifies cases where initial and verification results differ
5. **Comprehensive Analysis**: Provides both crop-level and video-level character analysis

## Usage Example

```python
from character_identification_pipeline import CharacterIdentificationPipeline

# Initialize pipeline (critique agent enabled by default)
pipeline = CharacterIdentificationPipeline()

# Process data with critique agent verification
result = pipeline.process(input_data, process_images=False)

# Access critique agent results
if result.critique_agent_result:
    critique = result.critique_agent_result
    print(f"Overall accuracy: {critique.overall_accuracy:.1%}")

    for verification in critique.character_verifications:
        status = "✓ PRESENT" if verification.is_present else "✗ NOT PRESENT"
        print(f"{verification.character_name}: {status}")
```

## Requirements

- OpenAI API key (`LLM_API_KEY` environment variable)
- Video URL in input data for critique agent verification
- Character reference images accessible via URLs

## Testing

Run the test script to verify the integration:

```bash
python test_critique_integration.py
```

This will demonstrate the complete pipeline with critique agent verification.

## Troubleshooting

### Critique Agent Not Running
- Check if `ENABLE_CRITIQUE_AGENT` is set to `True` in config
- Verify that `video_url` is provided in input data
- Ensure OpenAI API key is properly configured

### Performance Considerations
- Critique agent adds additional API calls to OpenAI
- Consider disabling for large-scale processing if cost is a concern
- Results are cached in the output structure for reuse

## Future Enhancements

- Batch processing for critique agent calls
- Confidence threshold filtering
- Automated discrepancy reporting
- Integration with evaluation metrics
