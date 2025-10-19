# Quick Start - VietTravel AI

Get up and running in 5 minutes with free APIs.

---

## Prerequisites

- Python 3.8+
- Get free API keys:
  - **Groq**: https://console.groq.com (takes 2 min)
  - **Pinecone**: https://pinecone.io (free tier available)
  - **Neo4j**: Local install or https://neo4j.com/cloud/aura/ (optional)

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create/edit `config.py`:

```python
import os

# LLM Provider
LLM_PROVIDER = "groq"  # Free Or use "openai" if you prefer

# Neo4j (optional but recommended)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

# Pinecone
PINECONE_API_KEY = "pcsk_..."  # Get from pinecone.io
PINECONE_INDEX_NAME = "vietnam-travel"

# Groq (FREE)
GROQ_API_KEY = "gsk_..."  # Get from console.groq.com

# OpenAI (optional, paid)
OPENAI_API_KEY = "sk-proj-..."  # Only needed if LLM_PROVIDER = "openai"
```

**Note:** Vector dimensions auto-configure based on provider (384 for Groq, 1536 for OpenAI).

### 3. Load Data

```bash
# Upload vectors to Pinecone 
python pinecone_upload.py

# Load graph to Neo4j 
python load_to_neo4j.py
```

### 4. Run the App

**Web UI (recommended):**
```bash
streamlit run streamlit_app.py
```
Opens at http://localhost:8501 with:
- Modern chat interface
- Smart suggestions
- Conversation memory
- Verbose mode toggle

**CLI alternative:**
```bash
python hybrid_chat.py
```

---

## Try These Queries

- "Create a romantic 5-day itinerary"
- "Best beaches in Vietnam?"
- "Cultural experiences in Hanoi"
- "Plan a food tour in Ho Chi Minh City"
- "Suggest vegan food options in the most populated city"
- "Find hotels and restaurants that are at least 4 stars and vegan, and their cost"

---

## Cost Breakdown

**Using Groq (default):**
- Groq LLM: $0 (free tier is generous)
- Pinecone: $0 (free tier: 1 index, 100K vectors)
- Neo4j Desktop: $0 (unlimited local use)

**Total: $0/month** 🎉

**Using OpenAI:**
- GPT-4o-mini: ~$50/month at moderate usage
- Pinecone: $0 (free tier)
- Neo4j: $0

---

## Switching Providers

Change one line in `config.py`:

```python
LLM_PROVIDER = "groq"   # Free Llama 3.3 70B
# or
LLM_PROVIDER = "openai" # Paid GPT-4o-mini
```

If switching providers, recreate the Pinecone index (different embedding dimensions):
```bash
# Delete old index
python -c "from pinecone import Pinecone; import config; pc = Pinecone(api_key=config.PINECONE_API_KEY); pc.delete_index(config.PINECONE_INDEX_NAME)"

# Upload with new provider
python pinecone_upload.py
```

---

## Need Help?

- Full docs: **README.md**
- Technical details: **improvements.md**
- Deployment guide: **DEPLOYMENT.md**

---

**You're ready!** Get your Groq key from https://console.groq.com and start chatting.
