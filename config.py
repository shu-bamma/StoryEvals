import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for StoryEvals with Portkey AI integration"""

    # Portkey AI Configuration
    LLM_API_KEY: str | None = None  # Portkey API key
    LLM_VIRTUAL_KEY: str = "open-ai-virtual-ceb436"  # Portkey virtual key
    LLM_MODEL: str = "gpt-5"  # Model to use (GPT-5 for main operations)
    LLM_MAX_TOKENS: int = 1000
    LLM_TEMPERATURE: float = 0.1

    # GPT-5 specific configuration
    GPT5_MAX_COMPLETION_TOKENS: int = 1000  # GPT-5 uses max_completion_tokens
    GPT5_TEMPERATURE: float = 1.0  # GPT-5 only supports temperature=1 (default)

    # Character Enrichment Configuration
    CHARACTER_ENRICHMENT_LLM_MODEL: str = "gpt-5"  # Model for character enrichment
    CHARACTER_ENRICHMENT_MAX_TOKENS: int = 2000
    CHARACTER_ENRICHMENT_TEMPERATURE: float = 1.0  # GPT-5 compatible
    CHARACTER_ENRICHMENT_MAX_RETRIES: int = 3
    CHARACTER_ENRICHMENT_RETRY_DELAY: float = 1.0  # Base delay in seconds

    # Character Identification Configuration
    CHAR_ID_LLM_MODEL: str = "gpt-5"  # Model for character identification
    CHAR_ID_MAX_TOKENS: int = 1500
    CHAR_ID_TEMPERATURE: float = 1.0  # GPT-5 compatible
    CHAR_ID_MAX_RETRIES: int = 3
    CHAR_ID_RETRY_TIMEOUT: int = 30  # seconds
    CHAR_ID_EXPONENTIAL_BACKOFF_BASE: float = 2.0

    # Batch Processing Configuration
    BATCH_SIZE: int = 6  # Number of crops per batch (4-8 configurable)
    MAX_PARALLEL_SHOTS: int = 4  # Number of shots to process in parallel

    # Image Processing Configuration
    OUTPUT_IMAGE_DIR: str = "data/logs/processed_crops"
    CROP_ID_FONT_SIZE: int = 24
    CROP_ID_COLOR: str = "red"
    CROP_ID_OUTLINE_COLOR: str = "black"
    CROP_ID_OUTLINE_WIDTH: int = 3

    # File Paths
    CHARACTER_VAULT_INPUT_PATH: str = (
        "data/cleaned_data/3-character_vault_with_full_video_with_clips_by_proj_id.json"
    )
    CHARACTER_ENRICHMENT_OUTPUT_PATH: str = "data/logs/1-enriched_characters.json"
    CHARACTER_IDENTIFICATION_OUTPUT_PATH: str = (
        "data/logs/2-character_identification_results.json"
    )

    # Parallel Processing
    CHARACTER_ENRICHMENT_MAX_WORKERS: int = 4  # Number of parallel workers

    # Video Shot Processing Configuration
    VIDEO_SHOT_DETECTION_MODEL: str = "yolov8n.pt"  # YOLO model for detection
    VIDEO_SHOT_DETECTION_CLASSES: list[str] = ["person"]  # Classes to detect
    VIDEO_SHOT_DETECTION_CONFIDENCE: float = 0.5  # Detection confidence threshold
    VIDEO_SHOT_BOUNDARY_THRESHOLD: float = 30.0  # Shot boundary detection threshold
    VIDEO_SHOT_KEYFRAME_INTERVAL_MS: int = 420  # Keyframe interval in milliseconds
    VIDEO_SHOT_BBOX_EXPANSION: float = 0.15  # Bbox expansion ratio
    VIDEO_SHOT_TEMP_VIDEO_DIR: str = "temp/videos"  # Temp directory for videos
    VIDEO_SHOT_TEMP_CROPS_DIR: str = "temp/crops"  # Temp directory for crops
    VIDEO_SHOT_CROP_UPLOAD_ENABLED: bool = False  # Enable crop upload
    VIDEO_SHOT_CROP_BASE_URL: str = (
        "https://cdn.example.com/crops/"  # Base URL for crops
    )
    VIDEO_SHOT_OUTPUT_PATH: str = (
        "data/logs/2-video_shots_with_crops.json"  # Output path
    )
    VIDEO_SHOT_NMS_ENABLED: bool = True  # Enable Non-Maximum Suppression
    VIDEO_SHOT_NMS_IOU_THRESHOLD: float = 0.5  # NMS IoU threshold

    # Evaluation Configuration
    EVALUATION_TIMEOUT: int = 60  # seconds
    IMAGE_DOWNLOAD_TIMEOUT: int = 30  # seconds
    MAX_RETRIES: int = 3

    # Critique Agent Configuration
    ENABLE_CRITIQUE_AGENT: bool = True  # Enable critique agent as double-check
    CRITIQUE_AGENT_LLM_MODEL: str = "gpt-5"  # Model for critique agent
    CRITIQUE_AGENT_MAX_TOKENS: int = 2000
    CRITIQUE_AGENT_TEMPERATURE: float = 1.0  # GPT-5 compatible
    CRITIQUE_AGENT_MAX_RETRIES: int = 3
    CRITIQUE_AGENT_RETRY_DELAY: float = 1.0  # Base delay in seconds

    # Output Configuration
    RESULTS_FILENAME: str = "story_evals_results.json"
    CRITIQUE_RESULTS_FILENAME: str = "critique_evaluation_results.json"

    @classmethod
    def load_from_env(cls) -> None:
        """Load configuration from environment variables"""
        cls.LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("PORTKEY_API_KEY")
        cls.LLM_VIRTUAL_KEY = os.getenv("LLM_VIRTUAL_KEY", cls.LLM_VIRTUAL_KEY)
        cls.LLM_MODEL = os.getenv("LLM_MODEL", cls.LLM_MODEL) or cls.LLM_MODEL
        cls.LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", str(cls.LLM_MAX_TOKENS)))
        cls.LLM_TEMPERATURE = float(
            os.getenv("LLM_TEMPERATURE", str(cls.LLM_TEMPERATURE))
        )
        cls.EVALUATION_TIMEOUT = int(
            os.getenv("EVALUATION_TIMEOUT", str(cls.EVALUATION_TIMEOUT))
        )
        cls.IMAGE_DOWNLOAD_TIMEOUT = int(
            os.getenv("IMAGE_DOWNLOAD_TIMEOUT", str(cls.IMAGE_DOWNLOAD_TIMEOUT))
        )
        cls.MAX_RETRIES = int(os.getenv("MAX_RETRIES", str(cls.MAX_RETRIES)))

        # Character enrichment config
        cls.CHARACTER_ENRICHMENT_LLM_MODEL = (
            os.getenv(
                "CHARACTER_ENRICHMENT_LLM_MODEL", cls.CHARACTER_ENRICHMENT_LLM_MODEL
            )
            or cls.CHARACTER_ENRICHMENT_LLM_MODEL
        )
        cls.CHARACTER_ENRICHMENT_MAX_TOKENS = int(
            os.getenv(
                "CHARACTER_ENRICHMENT_MAX_TOKENS",
                str(cls.CHARACTER_ENRICHMENT_MAX_TOKENS),
            )
        )
        cls.CHARACTER_ENRICHMENT_TEMPERATURE = float(
            os.getenv(
                "CHARACTER_ENRICHMENT_TEMPERATURE",
                str(cls.CHARACTER_ENRICHMENT_TEMPERATURE),
            )
        )
        cls.CHARACTER_ENRICHMENT_MAX_RETRIES = int(
            os.getenv(
                "CHARACTER_ENRICHMENT_MAX_RETRIES",
                str(cls.CHARACTER_ENRICHMENT_MAX_RETRIES),
            )
        )
        cls.CHARACTER_ENRICHMENT_RETRY_DELAY = float(
            os.getenv(
                "CHARACTER_ENRICHMENT_RETRY_DELAY",
                str(cls.CHARACTER_ENRICHMENT_RETRY_DELAY),
            )
        )
        cls.CHARACTER_ENRICHMENT_MAX_WORKERS = int(
            os.getenv(
                "CHARACTER_ENRICHMENT_MAX_WORKERS",
                str(cls.CHARACTER_ENRICHMENT_MAX_WORKERS),
            )
        )

        # Character identification config
        cls.CHAR_ID_LLM_MODEL = (
            os.getenv("CHAR_ID_LLM_MODEL", cls.CHAR_ID_LLM_MODEL)
            or cls.CHAR_ID_LLM_MODEL
        )
        cls.CHAR_ID_MAX_TOKENS = int(
            os.getenv("CHAR_ID_MAX_TOKENS", str(cls.CHAR_ID_MAX_TOKENS))
        )
        cls.CHAR_ID_TEMPERATURE = float(
            os.getenv("CHAR_ID_TEMPERATURE", str(cls.CHAR_ID_TEMPERATURE))
        )
        cls.CHAR_ID_MAX_RETRIES = int(
            os.getenv("CHAR_ID_MAX_RETRIES", str(cls.CHAR_ID_MAX_RETRIES))
        )
        cls.CHAR_ID_RETRY_TIMEOUT = int(
            os.getenv("CHAR_ID_RETRY_TIMEOUT", str(cls.CHAR_ID_RETRY_TIMEOUT))
        )
        cls.CHAR_ID_EXPONENTIAL_BACKOFF_BASE = float(
            os.getenv(
                "CHAR_ID_EXPONENTIAL_BACKOFF_BASE",
                str(cls.CHAR_ID_EXPONENTIAL_BACKOFF_BASE),
            )
        )

        # Critique agent config
        cls.CRITIQUE_AGENT_LLM_MODEL = (
            os.getenv("CRITIQUE_AGENT_LLM_MODEL", cls.CRITIQUE_AGENT_LLM_MODEL)
            or cls.CRITIQUE_AGENT_LLM_MODEL
        )
        cls.CRITIQUE_AGENT_MAX_TOKENS = int(
            os.getenv(
                "CRITIQUE_AGENT_MAX_TOKENS",
                str(cls.CRITIQUE_AGENT_MAX_TOKENS),
            )
        )
        cls.CRITIQUE_AGENT_TEMPERATURE = float(
            os.getenv(
                "CRITIQUE_AGENT_TEMPERATURE",
                str(cls.CRITIQUE_AGENT_TEMPERATURE),
            )
        )
        cls.CRITIQUE_AGENT_MAX_RETRIES = int(
            os.getenv(
                "CRITIQUE_AGENT_MAX_RETRIES",
                str(cls.CRITIQUE_AGENT_MAX_RETRIES),
            )
        )
        cls.CRITIQUE_AGENT_RETRY_DELAY = float(
            os.getenv(
                "CRITIQUE_AGENT_RETRY_DELAY",
                str(cls.CRITIQUE_AGENT_RETRY_DELAY),
            )
        )

        # Batch processing config
        cls.BATCH_SIZE = int(os.getenv("BATCH_SIZE", str(cls.BATCH_SIZE)))
        cls.MAX_PARALLEL_SHOTS = int(
            os.getenv("MAX_PARALLEL_SHOTS", str(cls.MAX_PARALLEL_SHOTS))
        )

        # Image processing config
        cls.OUTPUT_IMAGE_DIR = os.getenv("OUTPUT_IMAGE_DIR", cls.OUTPUT_IMAGE_DIR)
        cls.CROP_ID_FONT_SIZE = int(
            os.getenv("CROP_ID_FONT_SIZE", str(cls.CROP_ID_FONT_SIZE))
        )
        cls.CROP_ID_COLOR = os.getenv("CROP_ID_COLOR", cls.CROP_ID_COLOR)
        cls.CROP_ID_OUTLINE_COLOR = os.getenv(
            "CROP_ID_OUTLINE_COLOR", cls.CROP_ID_OUTLINE_COLOR
        )
        cls.CROP_ID_OUTLINE_WIDTH = int(
            os.getenv("CROP_ID_OUTLINE_WIDTH", str(cls.CROP_ID_OUTLINE_WIDTH))
        )

        # File paths
        cls.CHARACTER_VAULT_INPUT_PATH = os.getenv(
            "CHARACTER_VAULT_INPUT_PATH", cls.CHARACTER_VAULT_INPUT_PATH
        )
        cls.CHARACTER_ENRICHMENT_OUTPUT_PATH = os.getenv(
            "CHARACTER_ENRICHMENT_OUTPUT_PATH", cls.CHARACTER_ENRICHMENT_OUTPUT_PATH
        )
        cls.CHARACTER_IDENTIFICATION_OUTPUT_PATH = os.getenv(
            "CHARACTER_IDENTIFICATION_OUTPUT_PATH",
            cls.CHARACTER_IDENTIFICATION_OUTPUT_PATH,
        )

        # Video shot processing config
        cls.VIDEO_SHOT_DETECTION_MODEL = os.getenv(
            "VIDEO_SHOT_DETECTION_MODEL", cls.VIDEO_SHOT_DETECTION_MODEL
        )
        # Handle list environment variable for detection classes
        detection_classes_env = os.getenv("VIDEO_SHOT_DETECTION_CLASSES")
        if detection_classes_env:
            cls.VIDEO_SHOT_DETECTION_CLASSES = [
                cls.strip() for cls in detection_classes_env.split(",")
            ]
        cls.VIDEO_SHOT_DETECTION_CONFIDENCE = float(
            os.getenv(
                "VIDEO_SHOT_DETECTION_CONFIDENCE",
                str(cls.VIDEO_SHOT_DETECTION_CONFIDENCE),
            )
        )
        cls.VIDEO_SHOT_BOUNDARY_THRESHOLD = float(
            os.getenv(
                "VIDEO_SHOT_BOUNDARY_THRESHOLD", str(cls.VIDEO_SHOT_BOUNDARY_THRESHOLD)
            )
        )
        cls.VIDEO_SHOT_KEYFRAME_INTERVAL_MS = int(
            os.getenv(
                "VIDEO_SHOT_KEYFRAME_INTERVAL_MS",
                str(cls.VIDEO_SHOT_KEYFRAME_INTERVAL_MS),
            )
        )
        cls.VIDEO_SHOT_BBOX_EXPANSION = float(
            os.getenv("VIDEO_SHOT_BBOX_EXPANSION", str(cls.VIDEO_SHOT_BBOX_EXPANSION))
        )
        cls.VIDEO_SHOT_TEMP_VIDEO_DIR = os.getenv(
            "VIDEO_SHOT_TEMP_VIDEO_DIR", cls.VIDEO_SHOT_TEMP_VIDEO_DIR
        )
        cls.VIDEO_SHOT_TEMP_CROPS_DIR = os.getenv(
            "VIDEO_SHOT_TEMP_CROPS_DIR", cls.VIDEO_SHOT_TEMP_CROPS_DIR
        )
        cls.VIDEO_SHOT_CROP_UPLOAD_ENABLED = (
            os.getenv(
                "VIDEO_SHOT_CROP_UPLOAD_ENABLED",
                str(cls.VIDEO_SHOT_CROP_UPLOAD_ENABLED),
            ).lower()
            == "true"
        )
        cls.VIDEO_SHOT_CROP_BASE_URL = os.getenv(
            "VIDEO_SHOT_CROP_BASE_URL", cls.VIDEO_SHOT_CROP_BASE_URL
        )
        cls.VIDEO_SHOT_OUTPUT_PATH = os.getenv(
            "VIDEO_SHOT_OUTPUT_PATH", cls.VIDEO_SHOT_OUTPUT_PATH
        )
        cls.VIDEO_SHOT_NMS_ENABLED = (
            os.getenv("VIDEO_SHOT_NMS_ENABLED", str(cls.VIDEO_SHOT_NMS_ENABLED)).lower()
            == "true"
        )
        cls.VIDEO_SHOT_NMS_IOU_THRESHOLD = float(
            os.getenv(
                "VIDEO_SHOT_NMS_IOU_THRESHOLD", str(cls.VIDEO_SHOT_NMS_IOU_THRESHOLD)
            )
        )

    @classmethod
    def validate_portkey_config(cls) -> bool:
        """Validate that Portkey AI configuration is complete"""
        return bool(cls.LLM_API_KEY) and bool(cls.LLM_VIRTUAL_KEY)

    @classmethod
    def get_portkey_config(cls) -> dict:
        """Get Portkey AI configuration as dictionary"""
        # Check if using GPT-5
        is_gpt5 = "gpt-5" in cls.LLM_MODEL

        config = {
            "api_key": cls.LLM_API_KEY,
            "virtual_key": cls.LLM_VIRTUAL_KEY,
            "model": cls.LLM_MODEL,
        }

        # GPT-5 uses different parameters
        if is_gpt5:
            config["max_completion_tokens"] = str(cls.GPT5_MAX_COMPLETION_TOKENS)
            # GPT-5 only supports temperature=1 (default)
        else:
            config["max_tokens"] = str(cls.LLM_MAX_TOKENS)
            config["temperature"] = str(cls.LLM_TEMPERATURE)

        return config

    @classmethod
    def get_character_enrichment_config(cls) -> dict:
        """Get character enrichment configuration as dictionary"""
        # Check if using GPT-5
        is_gpt5 = "gpt-5" in cls.CHARACTER_ENRICHMENT_LLM_MODEL

        config = {
            "api_key": cls.LLM_API_KEY,
            "virtual_key": cls.LLM_VIRTUAL_KEY,
            "model": cls.CHARACTER_ENRICHMENT_LLM_MODEL,
            "max_retries": cls.CHARACTER_ENRICHMENT_MAX_RETRIES,
            "retry_delay": cls.CHARACTER_ENRICHMENT_RETRY_DELAY,
            "max_workers": cls.CHARACTER_ENRICHMENT_MAX_WORKERS,
        }

        # GPT-5 uses different parameters
        if is_gpt5:
            config["max_completion_tokens"] = cls.GPT5_MAX_COMPLETION_TOKENS
            # GPT-5 only supports temperature=1 (default)
        else:
            config["max_tokens"] = cls.CHARACTER_ENRICHMENT_MAX_TOKENS
            config["temperature"] = cls.CHARACTER_ENRICHMENT_TEMPERATURE

        return config

    @classmethod
    def get_character_identification_config(cls) -> dict:
        """Get character identification configuration as dictionary"""
        # Check if using GPT-5
        is_gpt5 = "gpt-5" in cls.CHAR_ID_LLM_MODEL

        config = {
            "api_key": cls.LLM_API_KEY,
            "virtual_key": cls.LLM_VIRTUAL_KEY,
            "model": cls.CHAR_ID_LLM_MODEL,
            "max_retries": cls.CHAR_ID_MAX_RETRIES,
            "retry_timeout": cls.CHAR_ID_RETRY_TIMEOUT,
            "exponential_backoff_base": cls.CHAR_ID_EXPONENTIAL_BACKOFF_BASE,
        }

        # GPT-5 uses different parameters
        if is_gpt5:
            config["max_completion_tokens"] = cls.GPT5_MAX_COMPLETION_TOKENS
            # GPT-5 only supports temperature=1 (default)
        else:
            config["max_tokens"] = cls.CHAR_ID_MAX_TOKENS
            config["temperature"] = cls.CHAR_ID_TEMPERATURE

        return config

    @classmethod
    def get_batch_processing_config(cls) -> dict:
        """Get batch processing configuration as dictionary"""
        return {
            "batch_size": cls.BATCH_SIZE,
            "max_parallel_shots": cls.MAX_PARALLEL_SHOTS,
        }

    @classmethod
    def get_image_processing_config(cls) -> dict:
        """Get image processing configuration as dictionary"""
        return {
            "output_dir": cls.OUTPUT_IMAGE_DIR,
            "font_size": cls.CROP_ID_FONT_SIZE,
            "color": cls.CROP_ID_COLOR,
            "outline_color": cls.CROP_ID_OUTLINE_COLOR,
            "outline_width": cls.CROP_ID_OUTLINE_WIDTH,
        }

    @classmethod
    def get_video_shot_processing_config(cls) -> dict:
        """Get video shot processing configuration as dictionary"""
        return {
            "detection_model": cls.VIDEO_SHOT_DETECTION_MODEL,
            "detection_classes": cls.VIDEO_SHOT_DETECTION_CLASSES,
            "detection_confidence": cls.VIDEO_SHOT_DETECTION_CONFIDENCE,
            "shot_boundary_threshold": cls.VIDEO_SHOT_BOUNDARY_THRESHOLD,
            "keyframe_interval_ms": cls.VIDEO_SHOT_KEYFRAME_INTERVAL_MS,
            "bbox_expansion": cls.VIDEO_SHOT_BBOX_EXPANSION,
            "temp_video_dir": cls.VIDEO_SHOT_TEMP_VIDEO_DIR,
            "temp_crops_dir": cls.VIDEO_SHOT_TEMP_CROPS_DIR,
            "crop_upload_enabled": cls.VIDEO_SHOT_CROP_UPLOAD_ENABLED,
            "crop_base_url": cls.VIDEO_SHOT_CROP_BASE_URL,
            "output_path": cls.VIDEO_SHOT_OUTPUT_PATH,
            "nms_enabled": cls.VIDEO_SHOT_NMS_ENABLED,
            "nms_iou_threshold": cls.VIDEO_SHOT_NMS_IOU_THRESHOLD,
        }

    @classmethod
    def get_critique_agent_config(cls) -> dict:
        """Get critique agent configuration as dictionary"""
        # Check if using GPT-5
        is_gpt5 = "gpt-5" in cls.CRITIQUE_AGENT_LLM_MODEL

        config = {
            "api_key": cls.LLM_API_KEY,
            "virtual_key": cls.LLM_VIRTUAL_KEY,
            "model": cls.CRITIQUE_AGENT_LLM_MODEL,
            "max_retries": cls.CRITIQUE_AGENT_MAX_RETRIES,
            "retry_delay": cls.CRITIQUE_AGENT_RETRY_DELAY,
        }

        # GPT-5 uses different parameters
        if is_gpt5:
            config["max_completion_tokens"] = cls.GPT5_MAX_COMPLETION_TOKENS
            # GPT-5 only supports temperature=1 (default)
        else:
            config["max_tokens"] = cls.CRITIQUE_AGENT_MAX_TOKENS
            config["temperature"] = cls.CRITIQUE_AGENT_TEMPERATURE

        return config


# Load configuration when module is imported
Config.load_from_env()
