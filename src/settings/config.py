import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # OpenAI API
    openai_api_key: str
    
    # Audio settings
    audio_format: str = "wav"
    sample_rate: int = 16000
    max_file_size_mb: int = 25  # Whisper API limit
    
    # Paths
    input_dir: str = "data/input"
    temp_dir: str = "data/temp"
    output_dir: str = "data/output"
    
    # Quality thresholds
    min_transcript_length: int = 100
    min_confidence_score: float = 0.7
    medical_terms_threshold: int = 5
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            input_dir=os.getenv("INPUT_DIR", "data/input"),
            temp_dir=os.getenv("TEMP_DIR", "data/temp"),
            output_dir=os.getenv("OUTPUT_DIR", "data/output"),
        )