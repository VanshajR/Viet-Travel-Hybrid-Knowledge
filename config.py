# config.py - Load configuration from .env file or Streamlit secrets
import os
from dotenv import load_dotenv

# Load environment variables from .env file (local dev)
load_dotenv()

# Try to import streamlit for cloud deployment
try:
    import streamlit as st
    # Check if we're in Streamlit Cloud (secrets will be available)
    if hasattr(st, 'secrets') and len(st.secrets) > 0:
        USE_STREAMLIT_SECRETS = True
    else:
        USE_STREAMLIT_SECRETS = False
except ImportError:
    USE_STREAMLIT_SECRETS = False

def get_config(key, default=""):
    """Get config from Streamlit secrets (if available) or environment variables."""
    if USE_STREAMLIT_SECRETS:
        try:
            return st.secrets.get(key, default)
        except:
            pass
    return os.getenv(key, default)

# Neo4j Configuration
NEO4J_URI = get_config("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = get_config("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = get_config("NEO4J_PASSWORD", "password")

# LLM Provider Selection
LLM_PROVIDER = get_config("LLM_PROVIDER", "groq")  # "groq" or "openai"

# GROQ API (Free)
GROQ_API_KEY = get_config("GROQ_API_KEY", "")

# OpenAI API (Paid - optional if using Groq)
OPENAI_API_KEY = get_config("OPENAI_API_KEY", "")

# Hugging Face API (for embeddings with Groq)
HF_TOKEN = get_config("HF_TOKEN", "")

# Pinecone Configuration
PINECONE_API_KEY = get_config("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = get_config("PINECONE_INDEX_NAME", "vietnam-travel")

# Auto-configure vector dimensions based on LLM provider
# Groq uses HuggingFace embeddings (384 dim), OpenAI uses text-embedding-3-small (1536 dim)
if LLM_PROVIDER == "groq":
    DEFAULT_DIM = "384"
else:
    DEFAULT_DIM = "1536"

# Get vector dimensions - handle both string and int from secrets/env
pinecone_dim_raw = get_config("PINECONE_VECTOR_DIM", DEFAULT_DIM)
PINECONE_VECTOR_DIM = int(pinecone_dim_raw) if pinecone_dim_raw else int(DEFAULT_DIM)

# Legacy compatibility
PINECONE_ENV = "us-east1-gcp"  # Not used in Pinecone v5+, kept for compatibility
