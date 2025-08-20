import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for StoryEvals with OpenAI integration"""
    
    # OpenAI Configuration
    LLM_API_KEY: Optional[str] = None  # OpenAI API key
    LLM_MODEL: str = "gpt-5"  # OpenAI model to use
    LLM_MAX_TOKENS: int = 1000
    LLM_TEMPERATURE: float = 0.1
    
    # Evaluation Configuration
    EVALUATION_TIMEOUT: int = 60  # seconds
    IMAGE_DOWNLOAD_TIMEOUT: int = 30  # seconds
    MAX_RETRIES: int = 3
    
    # Output Configuration
    RESULTS_FILENAME: str = "story_evals_results.json"
    CRITIQUE_RESULTS_FILENAME: str = "critique_evaluation_results.json"
    
    @classmethod
    def load_from_env(cls):
        """Load configuration from environment variables"""
        cls.LLM_API_KEY = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
        cls.LLM_MODEL = os.getenv('LLM_MODEL', cls.LLM_MODEL)
        cls.LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', cls.LLM_MAX_TOKENS))
        cls.LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', cls.LLM_TEMPERATURE))
        cls.EVALUATION_TIMEOUT = int(os.getenv('EVALUATION_TIMEOUT', cls.EVALUATION_TIMEOUT))
        cls.IMAGE_DOWNLOAD_TIMEOUT = int(os.getenv('IMAGE_DOWNLOAD_TIMEOUT', cls.IMAGE_DOWNLOAD_TIMEOUT))
        cls.MAX_RETRIES = int(os.getenv('MAX_RETRIES', cls.MAX_RETRIES))
    
    @classmethod
    def validate_openai_config(cls) -> bool:
        """Validate that OpenAI configuration is complete"""
        return bool(cls.LLM_API_KEY)
    
    @classmethod
    def get_openai_config(cls) -> dict:
        """Get OpenAI configuration as dictionary"""
        return {
            'api_key': cls.LLM_API_KEY,
            'model': cls.LLM_MODEL,
            'max_tokens': cls.LLM_MAX_TOKENS,
            'temperature': cls.LLM_TEMPERATURE
        }


# Load configuration when module is imported
Config.load_from_env() 