# hybrid_chat.py
import json
import asyncio
from typing import List, Dict, Tuple
from functools import lru_cache

# Support both OpenAI and Groq
if hasattr(__import__('config'), 'LLM_PROVIDER') and __import__('config').LLM_PROVIDER == "groq":
    # Use Groq for free API + HuggingFace for embeddings
    from groq import Groq
    from sentence_transformers import SentenceTransformer
    USE_GROQ = True
    print("🚀 Using Groq API (Free) + HuggingFace Embeddings")
else:
    # Use OpenAI (original approach - requires paid API key)
    from openai import OpenAI
    USE_GROQ = False
    print("🚀 Using OpenAI API")

from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import config

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"  # Works with both OpenAI and Groq

# Chat models:
# Groq (Free): "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"
# OpenAI (Paid): "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"
if USE_GROQ:
    CHAT_MODEL = "llama-3.3-70b-versatile"  # Groq's latest free model for chat
else:
    CHAT_MODEL = "gpt-4o-mini"  # OpenAI's efficient model

TOP_K = 5

INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
# Initialize LLM client (Groq or OpenAI)
if USE_GROQ:
    client = Groq(api_key=config.GROQ_API_KEY)
    # Initialize HuggingFace embedding model (384 dimensions)
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(f"✓ Groq client initialized (Model: {CHAT_MODEL})")
    print(f"✓ HuggingFace embedding model loaded (all-MiniLM-L6-v2)")
else:
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    embedding_model = None
    print(f"✓ OpenAI client initialized (Model: {CHAT_MODEL})")

pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Connect to Pinecone index
try:
    existing_indexes = [idx.name for idx in pc.list_indexes()]
except AttributeError:
    existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    print(f"Creating serverless index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# Connect to Neo4j with error handling
try:
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    print("✓ Connected to Neo4j")
except Exception as e:
    print(f"⚠ Neo4j connection issue: {e}")
    print("⚠ Graph features will be disabled")
    driver = None

# -----------------------------
# Helper functions
# -----------------------------
@lru_cache(maxsize=128)
def embed_text(text: str) -> Tuple[float, ...]:
    """Get embedding for a text string with caching."""
    try:
        if USE_GROQ:
            # Use HuggingFace sentence-transformers for embeddings
            embedding = embedding_model.encode(text, convert_to_numpy=True)
            # Return tuple for hashability (required by lru_cache)
            return tuple(embedding.tolist())
        else:
            # Use OpenAI embeddings API
            resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
            embedding = resp.data[0].embedding
            # Return tuple for hashability (required by lru_cache)
            return tuple(embedding)
    except Exception as e:
        print(f"⚠ Error generating embedding: {e}")
        return None

def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding."""
    try:
        vec = embed_text(query_text)
        if vec is None:
            print("⚠ Failed to generate embedding")
            return []
        
        # Convert tuple back to list for Pinecone
        vec_list = list(vec)
        res = index.query(
            vector=vec_list,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        matches = res.get("matches", [])
        print(f"✓ Pinecone returned {len(matches)} matches")
        return matches
    except Exception as e:
        print(f"⚠ Pinecone query error: {e}")
        return []

def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """Fetch neighboring nodes from Neo4j."""
    if not driver or not node_ids:
        return []
    
    facts = []
    try:
        with driver.session() as session:
            for nid in node_ids:
                q = (
                    "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                    "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                    "m.name AS name, m.type AS type, m.description AS description "
                    "LIMIT 10"
                )
                recs = session.run(q, nid=nid)
                for r in recs:
                    facts.append({
                        "source": nid,
                        "rel": r["rel"],
                        "target_id": r["id"],
                        "target_name": r["name"],
                        "target_desc": (r["description"] or "")[:400],
                        "labels": r["labels"]
                    })
        print(f"✓ Neo4j returned {len(facts)} graph facts")
        return facts
    except ServiceUnavailable:
        print("⚠ Neo4j service unavailable")
        return []
    except Exception as e:
        print(f"⚠ Neo4j query error: {e}")
        return []

# -----------------------------
# Async support for parallel fetching
# -----------------------------
async def fetch_vector_async(query: str) -> List[Dict]:
    """Async wrapper for Pinecone query."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, pinecone_query, query, TOP_K)

async def fetch_graph_async(node_ids: List[str]) -> List[Dict]:
    """Async wrapper for Neo4j graph query."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fetch_graph_context, node_ids, 1)

async def hybrid_search(query: str) -> Tuple[List[Dict], List[Dict]]:
    """Perform hybrid search: vector + graph retrieval in parallel."""
    # Fetch vector matches
    matches = await fetch_vector_async(query)
    
    # Extract node IDs for graph query
    node_ids = [m.get("id") or m.get("metadata", {}).get("id") for m in matches]
    node_ids = [nid for nid in node_ids if nid]
    
    # Fetch graph context in parallel if we have node IDs
    if node_ids:
        facts = await fetch_graph_async(node_ids)
    else:
        facts = []
    
    return matches, facts

# -----------------------------
# Search summarization helper
# -----------------------------
def search_summary(matches: List[Dict], facts: List[Dict]) -> Dict:
    """
    Intelligent summarization of retrieved search results.
    Reduces token usage and improves response quality.
    """
    # Deduplicate and categorize matches
    cities = []
    attractions = []
    activities = []
    other = []
    
    seen_ids = set()
    
    for match in matches:
        meta = match.get("metadata", {})
        node_id = meta.get("id", match.get("id"))
        
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        
        # Get name - CRITICAL: skip if no name available
        node_name = meta.get("name", "").strip()
        if not node_name:
            # Try alternate name field
            node_name = meta.get("title", "").strip()
        
        # Skip entries without proper names to avoid ID confusion
        if not node_name:
            continue
        
        node_type = meta.get("type", "").lower()
        node_info = {
            "id": node_id,
            "name": node_name,
            "type": meta.get("type", ""),
            "score": match.get("score", 0),
            "tags": meta.get("tags", "")
        }
        
        if "city" in node_type:
            cities.append(node_info)
        elif "attraction" in node_type or "landmark" in node_type:
            attractions.append(node_info)
        elif "activity" in node_type or "experience" in node_type:
            activities.append(node_info)
        else:
            other.append(node_info)
    
    # Summarize graph connections
    connections_by_rel = {}
    for fact in facts:
        rel_type = fact.get("rel", "RELATED_TO")
        if rel_type not in connections_by_rel:
            connections_by_rel[rel_type] = []
        connections_by_rel[rel_type].append({
            "source": fact["source"],
            "target": fact["target_id"],
            "name": fact["target_name"]
        })
    
    return {
        "cities": cities,
        "attractions": attractions,
        "activities": activities,
        "other": other,
        "connections": connections_by_rel,
        "total_matches": len(seen_ids),
        "total_facts": len(facts)
    }

def build_prompt(user_query, pinecone_matches, graph_facts, conversation_history=None):
    """Build an enhanced chat prompt with intelligent context organization and conversation memory."""
    system = (
        "You are VietTravel AI, an expert Vietnamese travel assistant with deep knowledge "
        "of Vietnam's destinations, culture, and experiences. Your goal is to create "
        "personalized, engaging travel recommendations.\n\n"
        "Guidelines:\n"
        "- Use the place names from the retrieved context naturally in your recommendations\n"
        "- If names seem generic or placeholder-like, you may briefly note this ONCE if relevant, then move on\n"
        "- Don't repeatedly mention 'database' or 'listed as' - focus on providing helpful travel advice\n"
        "- Use the tags, types, and context to give valuable recommendations even with limited names\n"
        "- Consider travel logistics, best times to visit, and local tips\n"
        "- Create coherent itineraries that account for geography and timing\n"
        "- Be enthusiastic and use descriptive language\n"
        "- Use proper formatting with headers, bullet points, and clear sections\n"
        "- Balance popular attractions with authentic experiences\n"
        "- Do NOT start responses with greetings like 'Xin chào' - just answer directly\n\n"
        "FORMATTING:\n"
        "- Write ONLY in plain ASCII text - NO unicode, NO italic math symbols (𝑡𝑜, 𝑓𝑜𝑟), NO fancy characters\n"
        "- For ranges: write '40 to 70' or '40-70' using regular letters and hyphen only\n"
        "- Use standard markdown for emphasis, but all letters must be regular ASCII\n"
        "- If tempted to use mathematical/italic letters, use plain English instead"
    )
    
    # Add conversation context if available
    if conversation_history:
        context_msg = "\n\nPrevious conversation:\n" + "\n".join(conversation_history)
        system += context_msg

    # Use search_summary for better organization
    summary = search_summary(pinecone_matches, graph_facts)
    
    # Build context sections - WITHOUT showing IDs to the LLM
    context_parts = []
    
    # Cities section
    if summary["cities"]:
        cities_text = "**Key Cities:**\n"
        for city in summary["cities"][:5]:
            tags_str = city['tags'] if isinstance(city['tags'], str) else ', '.join(city['tags']) if city['tags'] else 'general'
            cities_text += f"- {city['name']} - tags: {tags_str}\n"
        context_parts.append(cities_text)
    
    # Attractions section
    if summary["attractions"]:
        attr_text = "**Top Attractions:**\n"
        for attr in summary["attractions"][:8]:
            tags_str = attr['tags'] if isinstance(attr['tags'], str) else ', '.join(attr['tags']) if attr['tags'] else 'general'
            attr_text += f"- {attr['name']} - {tags_str}\n"
        context_parts.append(attr_text)
    
    # Activities section
    if summary["activities"]:
        act_text = "**Activities & Experiences:**\n"
        for act in summary["activities"][:5]:
            tags_str = act['tags'] if isinstance(act['tags'], str) else ', '.join(act['tags']) if act['tags'] else 'general'
            act_text += f"- {act['name']} - {tags_str}\n"
        context_parts.append(act_text)
    
    # Other matches
    if summary["other"]:
        other_text = "**Other Relevant Locations:**\n"
        for item in summary["other"][:5]:
            type_str = item['type'] if item['type'] else 'location'
            tags_str = item['tags'] if isinstance(item['tags'], str) else ', '.join(item['tags']) if item['tags'] else ''
            other_text += f"- {item['name']} ({type_str})"
            if tags_str:
                other_text += f" - {tags_str}"
            other_text += "\n"
        context_parts.append(other_text)
    
    # Connections section
    if summary["connections"]:
        conn_text = "**Nearby & Connected Places:**\n"
        for rel_type, conns in list(summary["connections"].items())[:3]:
            conn_names = [c['name'] for c in conns[:4] if c.get('name')]
            if conn_names:
                conn_text += f"- {', '.join(conn_names)}\n"
        context_parts.append(conn_text)
    
    # Combine context
    full_context = "\n\n".join(context_parts)

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"User query: {user_query}\n\n"
         f"Retrieved Context:\n{full_context}\n\n"
         f"Based on the above information ({summary['total_matches']} locations, "
         f"{summary['total_facts']} connections), provide a comprehensive, personalized response."}
    ]
    return prompt

def call_chat(prompt_messages):
    """Call chat completion API with error handling (works with both Groq and OpenAI)."""
    try:
        # Both Groq and OpenAI use the same API format
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=prompt_messages,
            max_tokens=800,
            temperature=0.7  # Slightly higher for more creative travel responses
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠ Error generating response: {e}"

# -----------------------------
# Interactive chat
# -----------------------------
def classify_user_intent(query: str) -> str:
    """Use LLM to intelligently classify user intent with smarter logic."""
    query_lower = query.lower().strip()
    
    # Rule-based quick classification for common patterns
    # Only classify as GREETING if it's clearly an initial greeting (not casual responses)
    greeting_patterns = ['hello', 'hi there', 'hey there', 'good morning', 'good afternoon', 
                        'good evening', 'xin chào', 'chào']
    if any(query_lower == pattern or query_lower.startswith(pattern + ' ') for pattern in greeting_patterns):
        return "GREETING"
    
    # Casual responses like "ok", "thanks", "yes", "no" are NOT greetings - treat as general
    casual_responses = ['ok', 'okay', 'thanks', 'thank you', 'yes', 'no', 'sure', 'alright', 'got it']
    if query_lower in casual_responses:
        return "GENERAL_INFO"
    
    # Use LLM for more complex queries
    classification_prompt = [
        {"role": "system", "content": """You are an intent classifier for a Vietnam travel assistant chatbot.

Classify the user's message into ONE of these categories:

1. GREETING - ONLY initial greetings like "hello", "hi", "good morning" (NOT casual responses like "ok", "thanks")
2. GENERAL_INFO - General questions about Vietnam travel, casual acknowledgments, follow-ups
3. SPECIFIC_SEARCH - Questions requiring specific locations, itineraries, recommendations

Respond with ONLY the category name, nothing else."""},
        {"role": "user", "content": f"Classify this message: {query}"}
    ]
    
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=classification_prompt,
            max_tokens=10,
            temperature=0.3
        )
        intent = resp.choices[0].message.content.strip().upper()
        return intent if intent in ["GREETING", "GENERAL_INFO", "SPECIFIC_SEARCH"] else "SPECIFIC_SEARCH"
    except Exception as e:
        print(f"⚠ Intent classification error: {e}")
        # Default to search if classification fails
        return "SPECIFIC_SEARCH"

def handle_casual_or_general(query: str, intent: str, conversation_history=None) -> str:
    """Generate response for casual greetings or general questions using LLM."""
    if intent == "GREETING":
        system_msg = """You are VietTravel AI, a friendly Vietnam travel assistant. 
The user greeted you. Respond warmly and briefly (2-3 sentences). You may use 'Xin chào!' as a greeting."""
    else:  # GENERAL_INFO
        system_msg = """You are VietTravel AI, a helpful Vietnam travel assistant. 
Answer general Vietnam travel questions concisely and naturally. 
Do NOT start with greetings like 'Xin chào' - just answer directly.
If they need specifics, suggest asking more detailed questions."""
        
        # Add conversation context if available
        if conversation_history:
            context_msg = "\n\nPrevious conversation:\n" + "\n".join(conversation_history)
            system_msg += context_msg
    
    prompt = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query}
    ]
    
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=prompt,
            max_tokens=400,
            temperature=0.7
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠ Error generating response: {e}"

def interactive_chat():
    """Enhanced interactive CLI for the travel assistant with conversation memory."""
    provider_name = "Groq (Free)" if USE_GROQ else "OpenAI (Paid)"
    print("=" * 70)
    print("🌏 VietTravel AI - Hybrid Travel Assistant")
    print("=" * 70)
    print(f"Powered by: Pinecone (vector) + Neo4j (graph) + {provider_name}")
    print("\nAsk me anything about Vietnam travel!")
    print("\nExample questions:")
    print("  • Create a romantic 4-day itinerary for Vietnam")
    print("  • What are the best beaches in Vietnam?")
    print("  • Suggest cultural experiences in Hanoi")
    print("  • Plan a food tour in Ho Chi Minh City")
    print("\nType 'exit' or 'quit' to end the conversation.\n")
    print("=" * 70)
    
    conversation_count = 0
    conversation_history = []  # Track last 10 messages (5 Q&A pairs)
    
    while True:
        # Get user input
        query = input("\n🗨️  Enter your travel question: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ("exit", "quit", "q", "bye"):
            print("\n👋 Thanks for using VietTravel AI! Have a wonderful trip!")
            break
        
        conversation_count += 1
        print(f"\n{'─' * 70}")
        print(f"Query #{conversation_count}: {query}")
        print(f"{'─' * 70}\n")
        
        try:
            # Use LLM to intelligently classify user intent
            print("🤔 Understanding your question...")
            intent = classify_user_intent(query)
            
            # Handle greetings and general questions without expensive searches
            if intent in ["GREETING", "GENERAL_INFO"]:
                print(f"💬 Intent: {intent.replace('_', ' ').title()}\n")
                print("=" * 70)
                print("✨ VietTravel AI Response:")
                print("=" * 70)
                answer = handle_casual_or_general(query, intent, conversation_history[-10:] if conversation_history else None)
                print(answer)
                print("\n" + "=" * 70)
                
                # Update conversation history
                conversation_history.append(f"User: {query[:150]}...")
                conversation_history.append(f"Assistant: {answer[:150]}...")
                continue
            
            # For specific searches, perform hybrid search
            print(f"💬 Intent: Specific Search\n")
            print("🔍 Searching vector database and knowledge graph...")
            matches, facts = asyncio.run(hybrid_search(query))
            
            # Check if we got results
            if not matches and not facts:
                print("⚠ No results found. Please try rephrasing your query.")
                if not matches:
                    print("  - No vector search results (check Pinecone connection)")
                if not facts and driver:
                    print("  - No graph results (data may not be loaded in Neo4j)")
                continue
            
            # Build prompt and generate response with conversation history
            print("💭 Generating personalized response...\n")
            prompt = build_prompt(query, matches, facts, conversation_history[-10:] if conversation_history else None)
            answer = call_chat(prompt)
            
            # Display response
            print("=" * 70)
            print("✨ VietTravel AI Response:")
            print("=" * 70)
            print(answer)
            print("\n" + "=" * 70)
            
            # Show stats
            summary = search_summary(matches, facts)
            print(f"\n📊 Retrieved: {summary['total_matches']} locations, "
                  f"{summary['total_facts']} connections")
            
            # Update conversation history
            conversation_history.append(f"User: {query[:150]}...")
            conversation_history.append(f"Assistant: {answer[:150]}...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠ Error processing query: {e}")
            print("Please try again or rephrase your question.")
    
    # Cleanup
    if driver:
        driver.close()
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    interactive_chat()
