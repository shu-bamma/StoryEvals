import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for StoryEvals with OpenAI integration"""

    # OpenAI Configuration
    LLM_API_KEY: str | None = None  # OpenAI API key
    LLM_MODEL: str = "gpt-5"  # OpenAI model to use (GPT-5 for main operations)
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

    # Evaluation Configuration
    EVALUATION_TIMEOUT: int = 60  # seconds
    IMAGE_DOWNLOAD_TIMEOUT: int = 30  # seconds
    MAX_RETRIES: int = 3

    # Critique Agent Configuration
    ENABLE_CRITIQUE_AGENT: bool = True  # Enable critique agent as double-check

    # Output Configuration
    RESULTS_FILENAME: str = "story_evals_results.json"
    CRITIQUE_RESULTS_FILENAME: str = "critique_evaluation_results.json"

    @classmethod
    def load_from_env(cls) -> None:
        """Load configuration from environment variables"""
        cls.LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
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

    @classmethod
    def validate_openai_config(cls) -> bool:
        """Validate that OpenAI configuration is complete"""
        return bool(cls.LLM_API_KEY)

    @classmethod
    def get_openai_config(cls) -> dict:
        """Get OpenAI configuration as dictionary"""
        # Check if using GPT-5
        is_gpt5 = "gpt-5" in cls.LLM_MODEL

        config = {
            "api_key": cls.LLM_API_KEY,
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


# Load configuration when module is imported
Config.load_from_env()
