import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for StoryEvals with OpenAI integration"""

    # OpenAI Configuration
    LLM_API_KEY: str | None = None  # OpenAI API key
    LLM_MODEL: str = "gpt-5"  # OpenAI model to use
    LLM_MAX_TOKENS: int = 1000
    LLM_TEMPERATURE: float = 0.1

    # Character Enrichment Configuration
    CHARACTER_ENRICHMENT_LLM_MODEL: str = (
        "gpt-4o-mini"  # Model for character enrichment
    )
    CHARACTER_ENRICHMENT_MAX_TOKENS: int = 2000
    CHARACTER_ENRICHMENT_TEMPERATURE: float = 0.1
    CHARACTER_ENRICHMENT_MAX_RETRIES: int = 3
    CHARACTER_ENRICHMENT_RETRY_DELAY: float = 1.0  # Base delay in seconds

    # File Paths
    CHARACTER_VAULT_INPUT_PATH: str = (
        "data/cleaned_data/3-character_vault_with_full_video_with_clips_by_proj_id.json"
    )
    CHARACTER_ENRICHMENT_OUTPUT_PATH: str = "data/logs/1-enriched_characters.json"

    # Parallel Processing
    CHARACTER_ENRICHMENT_MAX_WORKERS: int = 4  # Number of parallel workers

    # Evaluation Configuration
    EVALUATION_TIMEOUT: int = 60  # seconds
    IMAGE_DOWNLOAD_TIMEOUT: int = 30  # seconds
    MAX_RETRIES: int = 3

    # Output Configuration
    RESULTS_FILENAME: str = "story_evals_results.json"
    CRITIQUE_RESULTS_FILENAME: str = "critique_evaluation_results.json"

    @classmethod
    def load_from_env(cls) -> None:
        """Load configuration from environment variables"""
        cls.LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        cls.LLM_MODEL = os.getenv("LLM_MODEL", cls.LLM_MODEL)
        cls.LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", cls.LLM_MAX_TOKENS))
        cls.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", cls.LLM_TEMPERATURE))
        cls.EVALUATION_TIMEOUT = int(
            os.getenv("EVALUATION_TIMEOUT", cls.EVALUATION_TIMEOUT)
        )
        cls.IMAGE_DOWNLOAD_TIMEOUT = int(
            os.getenv("IMAGE_DOWNLOAD_TIMEOUT", cls.IMAGE_DOWNLOAD_TIMEOUT)
        )
        cls.MAX_RETRIES = int(os.getenv("MAX_RETRIES", cls.MAX_RETRIES))

        # Character enrichment config
        cls.CHARACTER_ENRICHMENT_LLM_MODEL = os.getenv(
            "CHARACTER_ENRICHMENT_LLM_MODEL", cls.CHARACTER_ENRICHMENT_LLM_MODEL
        )
        cls.CHARACTER_ENRICHMENT_MAX_TOKENS = int(
            os.getenv(
                "CHARACTER_ENRICHMENT_MAX_TOKENS", cls.CHARACTER_ENRICHMENT_MAX_TOKENS
            )
        )
        cls.CHARACTER_ENRICHMENT_TEMPERATURE = float(
            os.getenv(
                "CHARACTER_ENRICHMENT_TEMPERATURE", cls.CHARACTER_ENRICHMENT_TEMPERATURE
            )
        )
        cls.CHARACTER_ENRICHMENT_MAX_RETRIES = int(
            os.getenv(
                "CHARACTER_ENRICHMENT_MAX_RETRIES", cls.CHARACTER_ENRICHMENT_MAX_RETRIES
            )
        )
        cls.CHARACTER_ENRICHMENT_RETRY_DELAY = float(
            os.getenv(
                "CHARACTER_ENRICHMENT_RETRY_DELAY", cls.CHARACTER_ENRICHMENT_RETRY_DELAY
            )
        )
        cls.CHARACTER_ENRICHMENT_MAX_WORKERS = int(
            os.getenv(
                "CHARACTER_ENRICHMENT_MAX_WORKERS", cls.CHARACTER_ENRICHMENT_MAX_WORKERS
            )
        )

        # File paths
        cls.CHARACTER_VAULT_INPUT_PATH = os.getenv(
            "CHARACTER_VAULT_INPUT_PATH", cls.CHARACTER_VAULT_INPUT_PATH
        )
        cls.CHARACTER_ENRICHMENT_OUTPUT_PATH = os.getenv(
            "CHARACTER_ENRICHMENT_OUTPUT_PATH", cls.CHARACTER_ENRICHMENT_OUTPUT_PATH
        )

    @classmethod
    def validate_openai_config(cls) -> bool:
        """Validate that OpenAI configuration is complete"""
        return bool(cls.LLM_API_KEY)

    @classmethod
    def get_openai_config(cls) -> dict:
        """Get OpenAI configuration as dictionary"""
        return {
            "api_key": cls.LLM_API_KEY,
            "model": cls.LLM_MODEL,
            "max_tokens": cls.LLM_MAX_TOKENS,
            "temperature": cls.LLM_TEMPERATURE,
        }

    @classmethod
    def get_character_enrichment_config(cls) -> dict:
        """Get character enrichment configuration as dictionary"""
        return {
            "api_key": cls.LLM_API_KEY,
            "model": cls.CHARACTER_ENRICHMENT_LLM_MODEL,
            "max_tokens": cls.CHARACTER_ENRICHMENT_MAX_TOKENS,
            "temperature": cls.CHARACTER_ENRICHMENT_TEMPERATURE,
            "max_retries": cls.CHARACTER_ENRICHMENT_MAX_RETRIES,
            "retry_delay": cls.CHARACTER_ENRICHMENT_RETRY_DELAY,
            "max_workers": cls.CHARACTER_ENRICHMENT_MAX_WORKERS,
        }


# Load configuration when module is imported
Config.load_from_env()
