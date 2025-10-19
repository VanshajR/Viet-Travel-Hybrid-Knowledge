# pinecone_upload.py
import json
import time
from tqdm import tqdm

# Support both OpenAI and Groq for embeddings
if hasattr(__import__('config'), 'LLM_PROVIDER') and __import__('config').LLM_PROVIDER == "groq":
    # Use Groq for free API (chat only) + HuggingFace for embeddings
    from groq import Groq
    from sentence_transformers import SentenceTransformer
    USE_GROQ = True
else:
    # Use OpenAI (original approach)
    from openai import OpenAI
    USE_GROQ = False

from pinecone import Pinecone, ServerlessSpec
import config

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32

INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
# GROQ (Free API) - For chat only, uses HuggingFace for embeddings
# OPENAI (Paid) - Direct OpenAI API for both chat and embeddings
if USE_GROQ:
    print("Using Groq API for chat + HuggingFace for embeddings...")
    client = Groq(api_key=config.GROQ_API_KEY)
    # Initialize HuggingFace embedding model (384 dimensions)
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    VECTOR_DIM = 384
else:
    print("Using OpenAI API for chat and embeddings...")
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    embedding_model = None
    VECTOR_DIM = 1536  # OpenAI text-embedding-3-small

pc = Pinecone(api_key=config.PINECONE_API_KEY)

# -----------------------------
# Create serverless index if it doesn't exist
# -----------------------------
try:
    # Pinecone v5+ returns list of index objects
    existing_indexes = [idx.name for idx in pc.list_indexes()]
except AttributeError:
    # Fallback for older versions
    existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    print(f"Creating serverless index: {INDEX_NAME}")
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Waiting for index to be ready...")
        time.sleep(5)
    except Exception as e:
        print(f"Error creating index: {e}")
        exit(1)
else:
    print(f"Index {INDEX_NAME} already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
def get_embeddings(texts, model="text-embedding-3-small"):
    """Generate embeddings using HuggingFace (Groq) or OpenAI."""
    try:
        if USE_GROQ:
            # Use HuggingFace sentence-transformers for embeddings
            embeddings = embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            # Use OpenAI embeddings API
            resp = client.embeddings.create(model=model, input=texts)
            return [data.embedding for data in resp.data]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    print(f"Loading data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    
    print(f"Loaded {len(nodes)} nodes from dataset.")

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        
        # Convert tags list to string for metadata
        tags = node.get("tags", [])
        tags_str = ",".join(tags) if isinstance(tags, list) else str(tags)
        
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": tags_str
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts, model="text-embedding-3-small")

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            print(f"Error upserting batch: {e}")
            continue
        
        time.sleep(0.2)

    # Verify upload
    try:
        stats = index.describe_index_stats()
        print(f"\n✓ Upload complete!")
        print(f"Index stats: {stats}")
    except Exception as e:
        print(f"Upload complete (stats unavailable: {e})")

# -----------------------------
if __name__ == "__main__":
    main()
