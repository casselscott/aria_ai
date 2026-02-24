# config.py - Configuration settings
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List

class Settings(BaseSettings):
    # App settings
    app_name: str = "ARIA Real Estate AI"
    app_version: str = "2.0"
    environment: str = "development"
    debug: bool = True
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS settings
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"  # Remove in production
    ]
    allow_credentials: bool = True
    
    # AI settings
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.3
    max_search_results: int = 5
    
    @field_validator('allowed_origins', mode='before')
    @classmethod
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()