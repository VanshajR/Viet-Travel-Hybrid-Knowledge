# config.py - Load configuration from .env file
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# LLM Provider Selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "openai"

# GROQ API (Free)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# OpenAI API (Paid - optional if using Groq)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Hugging Face API (for embeddings with Groq)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "vietnam-travel")

# Auto-configure vector dimensions based on LLM provider
# Groq uses HuggingFace embeddings (384 dim), OpenAI uses text-embedding-3-small (1536 dim)
if LLM_PROVIDER == "groq":
    DEFAULT_DIM = "384"
else:
    DEFAULT_DIM = "1536"
PINECONE_VECTOR_DIM = int(os.getenv("PINECONE_VECTOR_DIM", DEFAULT_DIM))

# Legacy compatibility
PINECONE_ENV = "us-east1-gcp"  # Not used in Pinecone v5+, kept for compatibility
