"""
Configuration settings for the LLM-Powered Intelligent Query-Retrieval System
"""
import os
from typing import Dict, Any
from pydantic import BaseSettings, Field


class Config(BaseSettings):
    """System configuration settings"""
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    max_tokens: int = Field(default=1000, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    
    # Embedding Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # FAISS Configuration
    faiss_index_type: str = Field(default="IndexFlatIP", env="FAISS_INDEX_TYPE")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    max_retrieval_docs: int = Field(default=5, env="MAX_RETRIEVAL_DOCS")
    
    # Document Processing
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    
    # System Paths
    data_dir: str = Field(default="data", env="DATA_DIR")
    index_dir: str = Field(default="indices", env="INDEX_DIR")
    temp_dir: str = Field(default="temp", env="TEMP_DIR")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global configuration instance
config = Config()

# Domain-specific configurations
DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "insurance": {
        "query_templates": [
            "coverage analysis",
            "policy conditions",
            "exclusions",
            "claim procedures",
            "premium calculations"
        ],
        "key_terms": [
            "coverage", "deductible", "premium", "claim", "exclusion",
            "policy", "benefit", "liability", "copay", "coinsurance"
        ]
    },
    "legal": {
        "query_templates": [
            "contract clause analysis",
            "legal obligations",
            "compliance requirements",
            "penalty provisions",
            "termination conditions"
        ],
        "key_terms": [
            "clause", "agreement", "obligation", "liability", "breach",
            "termination", "penalty", "compliance", "jurisdiction", "damages"
        ]
    },
    "hr": {
        "query_templates": [
            "employee benefits",
            "policy violations",
            "compensation structure",
            "leave policies",
            "performance requirements"
        ],
        "key_terms": [
            "employee", "benefit", "salary", "leave", "performance",
            "policy", "violation", "compensation", "evaluation", "training"
        ]
    },
    "compliance": {
        "query_templates": [
            "regulatory requirements",
            "audit procedures",
            "risk assessment",
            "documentation standards",
            "reporting obligations"
        ],
        "key_terms": [
            "regulation", "compliance", "audit", "risk", "requirement",
            "standard", "reporting", "documentation", "procedure", "violation"
        ]
    }
}