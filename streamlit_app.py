# streamlit_app.py
"""
VietTravel AI - Streamlit Web Interface
A modern chat interface for Vietnam travel assistance using hybrid search.
"""

import streamlit as st
from typing import List, Dict, Tuple
from functools import lru_cache

# Support both OpenAI and Groq
if hasattr(__import__('config'), 'LLM_PROVIDER') and __import__('config').LLM_PROVIDER == "groq":
    from groq import Groq
    from sentence_transformers import SentenceTransformer
    USE_GROQ = True
else:
    from openai import OpenAI
    USE_GROQ = False

from pinecone import Pinecone
from neo4j import GraphDatabase
import config

# Page config
st.set_page_config(
    page_title="VietTravel AI",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Vietnam-themed UI
st.markdown("""
<style>
    .intent-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin: 5px 0;
    }
    .greeting { background-color: #ffd700; color: #333; }
    .general { background-color: #87ceeb; color: #333; }
    .search { background-color: #98fb98; color: #333; }
    
    .source-card {
        background-color: var(--secondary-background-color);
        border-left: 4px solid #667eea;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .thinking-step {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 5px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Initialize clients (cached)
# -----------------------------
# Groq models with fallback priority
GROQ_MODELS = [
    "llama-3.3-70b-versatile",  # Primary - Best quality
    "llama-3.1-8b-instant",     # Fallback - Fast and reliable
    "mixtral-8x7b-32768",       # Final fallback - Alternative
]

@st.cache_resource
def init_static_clients():
    """Initialize static clients (Pinecone, Neo4j) once and cache them."""
    # Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)
    
    # Neo4j - with error handling for connection issues
    driver = None
    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            max_connection_lifetime=3600,  # 1 hour
            connection_timeout=30,  # 30 seconds
            max_connection_pool_size=50
        )
        # Test connection
        driver.verify_connectivity()
    except Exception as e:
        st.warning(f"⚠️ Neo4j connection unavailable: {str(e)[:100]}. Will use Pinecone only.")
        driver = None
    
    return index, driver

@st.cache_resource
def get_default_llm_client():
    """Get cached default LLM client based on config."""
    if USE_GROQ:
        client = Groq(api_key=config.GROQ_API_KEY)
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        chat_models = GROQ_MODELS
        provider = "groq"
    else:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        embedding_model = None
        chat_models = ["gpt-4o-mini"]
        provider = "openai"
    
    return client, embedding_model, chat_models, provider

def init_llm_client(api_key_override=None, use_openai_override=False):
    """Initialize LLM client with optional API key override."""
    # If user provided OpenAI key, use OpenAI regardless of config
    if api_key_override and use_openai_override:
        try:
            client = OpenAI(api_key=api_key_override)
            # Test the key by making a simple call
            client.models.list()
            embedding_model = None
            chat_models = ["gpt-4o-mini"]
            return client, embedding_model, chat_models, "openai", True
        except Exception as e:
            st.error(f"❌ Invalid OpenAI API key: {str(e)[:100]}")
            return None, None, None, None, False
    
    # Otherwise use cached default config
    client, embedding_model, chat_models, provider = get_default_llm_client()
    return client, embedding_model, chat_models, provider, True

# Initialize static clients
index, driver = init_static_clients()

# -----------------------------
# Helper functions
# -----------------------------
def call_llm_with_fallback(client, chat_models, messages, max_tokens=800, temperature=0.7, stream=False):
    """
    Call LLM with automatic fallback to alternative models.
    Shows verbose thinking process for each attempt.
    """
    for i, model in enumerate(chat_models):
        try:
            if i > 0:
                # Show fallback message
                st.markdown(f'<div class="thinking-step">⚠️ Retrying with fallback model: <b>{model}</b></div>', 
                          unsafe_allow_html=True)
            
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )
            
            if i > 0:
                # Show success message if we fell back
                st.markdown(f'<div class="thinking-step">✅ Successfully connected using <b>{model}</b></div>', 
                          unsafe_allow_html=True)
            
            return resp, model
        except Exception as e:
            error_msg = str(e)
            if i < len(chat_models) - 1:
                # Not the last model, show retry message
                st.markdown(f'<div class="thinking-step">⚠️ <b>{model}</b> failed: {error_msg[:100]}... Trying next model...</div>', 
                          unsafe_allow_html=True)
            else:
                # Last model failed
                st.markdown(f'<div class="thinking-step">❌ All models failed. Last error: {error_msg[:150]}</div>', 
                          unsafe_allow_html=True)
                raise Exception(f"All models failed. Last error: {error_msg}")
    
    raise Exception("No models available")

def generate_suggested_questions(query: str, intent: str, matches: List = None) -> List[str]:
    """Generate contextual follow-up questions based on the conversation."""
    suggestions = []
    query_lower = query.lower()
    
    if intent == "GREETING":
        suggestions = [
            "What are the best places to visit in Vietnam?",
            "Plan a 5-day itinerary for me",
            "What are the must-try Vietnamese foods?",
            "Tell me about beaches in Vietnam"
        ]
    elif intent == "GENERAL_INFO":
        # Contextual based on what they asked
        if "food" in query_lower or "eat" in query_lower or "cuisine" in query_lower:
            suggestions = [
                "Where can I try authentic pho?",
                "What street food should I try?",
                "Tell me about Vietnamese coffee culture",
                "What are regional food specialties?"
            ]
        elif "beach" in query_lower or "island" in query_lower:
            suggestions = [
                "Which beach destination is best for families?",
                "Tell me about Phu Quoc Island",
                "What water activities are available?",
                "Compare Da Nang vs Nha Trang beaches"
            ]
        elif "itinerary" in query_lower or "plan" in query_lower:
            suggestions = [
                "Create a north to south Vietnam itinerary",
                "What's a realistic 7-day plan?",
                "Help me plan a budget trip",
                "What should I not miss in 10 days?"
            ]
        else:
            suggestions = [
                "What's the best time to visit Vietnam?",
                "Tell me about Ho Chi Minh City",
                "What are the must-see attractions?",
                "How much should I budget per day?"
            ]
    elif intent == "SPECIFIC_SEARCH" and matches:
        # Generate highly contextual suggestions based on actual search results
        suggestions = []
        if len(matches) > 0:
            cities = list(set([m.get('city', '') for m in matches[:3] if m.get('city')]))
            types = list(set([m.get('type', '') for m in matches[:3] if m.get('type')]))
            
            # Suggestion 1: Related attractions in same city
            if cities and len(cities) > 0:
                suggestions.append(f"Show me more things to do in {cities[0]}")
            
            # Suggestion 2: Different type of place
            if types and len(types) > 1:
                suggestions.append(f"Find {types[1]}s near these locations")
            elif types:
                other_type = "restaurants" if types[0] != "restaurant" else "attractions"
                suggestions.append(f"Find {other_type} nearby")
            
            # Suggestion 3: Itinerary building
            if cities:
                suggestions.append(f"Build a 3-day itinerary for {cities[0]}")
            
            # Suggestion 4: Practical info
            suggestions.append("What's the best way to get around?")
    
    # Filter out suggestions that have already been used
    used = st.session_state.get("used_suggestions", set())
    suggestions = [s for s in suggestions if s.lower() not in used]
    
    return suggestions[:4]  # Return max 4 suggestions

def embed_text_with_client(text: str, client, embedding_model) -> Tuple[float, ...]:
    """Get embedding for a text string."""
    try:
        if embedding_model is not None:  # Using HuggingFace (Groq)
            embedding = embedding_model.encode(text, convert_to_numpy=True)
            return tuple(embedding.tolist())
        else:  # Using OpenAI
            resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
            return tuple(resp.data[0].embedding)
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def pinecone_query(query_text: str, client, embedding_model, top_k=5):
    """Query Pinecone index using embedding."""
    try:
        vec = embed_text_with_client(query_text, client, embedding_model)
        if vec is None:
            return []
        
        results = index.query(vector=list(vec), top_k=top_k, include_metadata=True)
        matches = []
        for match in results.matches:
            name = match.metadata.get("name", "").strip()
            if not name:
                continue
            
            matches.append({
                "id": match.id,
                "score": match.score,
                "name": name,
                "type": match.metadata.get("type", "location"),
                "city": match.metadata.get("city", ""),
                "tags": match.metadata.get("tags", "")
            })
        return matches
    except Exception as e:
        st.error(f"Pinecone error: {e}")
        return []

def neo4j_query(query_text: str, top_k=5):
    """Query Neo4j for related entities."""
    if not driver:
        return []
    
    query = """
    MATCH (n)
    WHERE n.name IS NOT NULL
    RETURN n.id AS id, n.name AS name, n.type AS type, 
           n.city AS city, n.description AS description
    LIMIT $limit
    """
    
    try:
        # Verify connection first
        driver.verify_connectivity()
        
        with driver.session(database="neo4j") as session:
            results = session.run(query, limit=top_k * 2)
            facts = []
            for record in results:
                name = record.get("name")
                if not name or not str(name).strip():
                    continue
                
                facts.append({
                    "id": record.get("id"),
                    "name": str(name).strip(),
                    "type": record.get("type", "location"),
                    "city": record.get("city", ""),
                    "description": record.get("description", "")[:200]
                })
            return facts[:top_k]
    except Exception as e:
        # Don't show error in UI for connection issues, just return empty
        return []

def hybrid_search(query: str, client, embedding_model):
    """Perform search in Pinecone and Neo4j (synchronous for Streamlit compatibility)."""
    import concurrent.futures
    
    # Run searches in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_pinecone = executor.submit(pinecone_query, query, client, embedding_model)
        future_neo4j = executor.submit(neo4j_query, query)
        
        matches = future_pinecone.result()
        facts = future_neo4j.result()
    
    return matches, facts

def classify_intent(query: str, client, chat_models) -> str:
    """Classify user intent using LLM with smarter logic."""
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
    prompt = [
        {"role": "system", "content": """Classify the user's message into ONE category:
1. GREETING - ONLY initial greetings like "hello", "hi", "good morning" (NOT casual responses like "ok", "thanks")
2. GENERAL_INFO - General questions about Vietnam travel, casual acknowledgments, follow-ups
3. SPECIFIC_SEARCH - Questions requiring specific locations/itineraries/recommendations

Respond with ONLY the category name."""},
        {"role": "user", "content": f"Classify: {query}"}
    ]
    
    try:
        resp, model_used = call_llm_with_fallback(
            client, chat_models,
            messages=prompt,
            max_tokens=10,
            temperature=0.3,
            stream=False
        )
        intent = resp.choices[0].message.content.strip().upper()
        return intent if intent in ["GREETING", "GENERAL_INFO", "SPECIFIC_SEARCH"] else "SPECIFIC_SEARCH"
    except:
        return "SPECIFIC_SEARCH"

def generate_response(query: str, intent: str, matches=None, facts=None, conversation_history=None):
    """Generate response based on intent and search results with conversation history."""
    if intent == "GREETING":
        system_msg = "You are VietTravel AI, a friendly Vietnam travel assistant. The user greeted you. Respond warmly and briefly (2-3 sentences). You may use 'Xin chào!' as a greeting."
        prompt = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query}
        ]
    elif intent == "GENERAL_INFO":
        system_msg = "You are VietTravel AI, a helpful Vietnam travel assistant. Answer general Vietnam travel questions concisely and naturally. Do NOT start with greetings like 'Xin chào' - just answer directly. If they need specifics, suggest asking more detailed questions."
        
        # Add conversation context for general info questions
        if conversation_history:
            context_msg = "Previous conversation:\n" + "\n".join(conversation_history)
            system_msg += f"\n\n{context_msg}"
        
        prompt = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query}
        ]
    else:  # SPECIFIC_SEARCH
        # Build context from search results
        context_parts = []
        
        if matches:
            context_parts.append("=== Recommended Locations ===")
            for m in matches:
                name = m.get('name', '').strip()
                if not name:
                    continue
                location_type = m.get('type', 'location')
                city = m.get('city', '')
                city_str = f", {city}" if city else ""
                context_parts.append(f"- {name} ({location_type}{city_str})")
        
        if facts:
            context_parts.append("\n=== Connected Places ===")
            for f in facts:
                name = f.get('name', '').strip()
                if not name:
                    continue
                location_type = f.get('type', 'location')
                city = f.get('city', '')
                city_str = f", {city}" if city else ""
                context_parts.append(f"- {name} ({location_type}{city_str})")
        
        context = "\n".join(context_parts)
        
        system_msg = """You are VietTravel AI, an expert Vietnam travel assistant.

Guidelines:
- Use the place names provided in the context naturally in your recommendations
- If names seem generic or placeholder-like (e.g., "Attraction 14"), briefly note "while specific names aren't available" ONCE if needed, then continue with helpful advice
- Don't repeatedly mention "database" or "listed as" - just provide useful travel recommendations
- Focus on the tags, location types, and context to give good advice even with limited names
- Structure itineraries day-by-day with specific locations when possible
- Be enthusiastic and helpful!
- Do NOT start responses with greetings like 'Xin chào' - just answer directly

FORMATTING RULES:
- Write ONLY in plain ASCII text - absolutely NO unicode, NO italic math symbols, NO fancy characters
- For ranges: write "40 to 70" or "40-70" using regular hyphen/dash only
- NEVER use mathematical italic letters (like 𝑡𝑜, 𝑓𝑜𝑟) - use regular letters only
- Use standard markdown: **bold** for emphasis, - for bullet points
- All text must be readable in plain text format
- If you're tempted to use fancy formatting, DON'T - just use normal English words"""
        
        # Add conversation context for specific searches
        if conversation_history:
            context_msg = "\n\nPrevious conversation:\n" + "\n".join(conversation_history)
            system_msg += context_msg
        
        prompt = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}
        ]
    
    # Generate response with streaming support
    return prompt  # Return prompt for streaming in main function

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    # Sidebar
    with st.sidebar:
        st.title("🌏 VietTravel AI")
        st.markdown("---")
        
        # Settings section right at the top
        st.markdown("### ⚙️ Settings")
        
        # Optional OpenAI API Key override
        st.markdown("#### 🔑 API Key Override (Optional)")
        openai_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use OpenAI instead of the configured provider",
            placeholder="sk-proj-..."
        )
        use_input_key = st.checkbox(
            "Use entered API key",
            value=False,
            help="Check this to use the API key entered above"
        )
        
        st.markdown("---")
        
        show_verbose = st.checkbox(
            "Show detailed search process",
            value=False,
            help="Display thinking process, intent classification, and search results"
        )
        if "show_verbose" not in st.session_state:
            st.session_state.show_verbose = False
        st.session_state.show_verbose = show_verbose
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.used_suggestions = set()
            st.rerun()
        
        # Initialize LLM client based on settings
        client, embedding_model, chat_models, provider, is_valid = init_llm_client(
            api_key_override=openai_key_input if use_input_key else None,
            use_openai_override=use_input_key
        )
        
        if not is_valid:
            st.error("⚠️ Unable to initialize LLM client. Check your API keys in config.")
            st.stop()
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        # Show provider based on actual client being used
        if use_input_key and openai_key_input:
            provider_display = "OpenAI (Custom Key) 🔑"
            embedding_display = "OpenAI"
        elif provider == "groq":
            provider_display = "Groq (Free) 🆓"
            embedding_display = "HuggingFace"
        else:
            provider_display = "OpenAI (Config) 💰"
            embedding_display = "OpenAI"
        
        st.info(f"**LLM Provider:** {provider_display}")
        st.info(f"**Models:** {', '.join(chat_models)}")
        st.info(f"**Embeddings:** {embedding_display}")
        
        st.success("✅ Pinecone Connected")
        if driver:
            st.success("✅ Neo4j Connected")
        else:
            st.warning("⚠️ Neo4j Unavailable (Pinecone only)")
        st.success(f"✅ {provider} Ready")
        
        st.markdown("---")
        st.markdown("### 💡 Try asking:")
        st.markdown("""
        - Create a romantic 5-day itinerary
        - Best beaches in Vietnam?
        - Cultural experiences in Hanoi
        - Food tour in Ho Chi Minh City
        """)
    
    # Main chat interface
    st.title("🌏 VietTravel AI - Your Vietnam Travel Assistant")
    st.caption("Powered by Hybrid Search: Pinecone + Neo4j + AI")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize next_query for suggested questions
    if "next_query" not in st.session_state:
        st.session_state.next_query = None
    
    # Track used suggestions to avoid repeating them
    if "used_suggestions" not in st.session_state:
        st.session_state.used_suggestions = set()
    
    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata if available
            if "metadata" in message and message["metadata"]:
                meta = message["metadata"]
                
                # Intent badge
                if "intent" in meta:
                    intent = meta["intent"]
                    badge_class = "greeting" if "GREETING" in intent else "general" if "GENERAL" in intent else "search"
                    st.markdown(f'<span class="intent-badge {badge_class}">💭 {intent.replace("_", " ")}</span>', 
                              unsafe_allow_html=True)
                
                # Search results preview
                if "matches" in meta and meta["matches"]:
                    with st.expander(f"🔍 Vector Search Results ({len(meta['matches'])} matches)"):
                        for m in meta["matches"]:
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>{m['name']}</strong> ({m['type']})<br>
                                📍 {m['city']} | ⭐ Score: {m['score']:.2f}
                            </div>
                            """, unsafe_allow_html=True)
                
                if "facts" in meta and meta["facts"]:
                    with st.expander(f"🕸️ Knowledge Graph Results ({len(meta['facts'])} facts)"):
                        for f in meta["facts"]:
                            st.markdown(f"- **{f['name']}** ({f['type']}, {f.get('city', '')})")
            
            # Show suggestions for the LAST assistant message only (on reruns, not during initial processing)
            if (message["role"] == "assistant" and 
                idx == len(st.session_state.messages) - 1 and
                "suggestions" in message and message["suggestions"] and
                not st.session_state.get("processing", False)):  # Only show if not currently processing
                
                st.markdown("---")
                st.markdown("**💡 Suggested questions:**")
                cols = st.columns(2)
                for i, suggestion in enumerate(message["suggestions"]):
                    with cols[i % 2]:
                        # Use SAME key as inline buttons so they're consistent
                        unique_key = f"sug_{idx}_{i}"
                        if st.button(suggestion, key=unique_key, use_container_width=True):
                            st.session_state.next_query = suggestion
                            # Mark this suggestion as used
                            st.session_state.used_suggestions.add(suggestion.lower())
                            st.rerun()
    
    # Chat input - ALWAYS render
    user_input = st.chat_input("Ask me anything about Vietnam travel...")
    
    # Only process new input if message count hasn't changed (meaning we haven't just added a response)
    current_msg_count = len(st.session_state.messages)
    
    # Initialize last_msg_count if not exists
    if "last_msg_count" not in st.session_state:
        st.session_state.last_msg_count = 0
    
    # Check if we have NEW input to process
    if st.session_state.next_query:
        # Suggestion was clicked
        prompt = st.session_state.next_query
        st.session_state.next_query = None
    elif user_input and current_msg_count == st.session_state.last_msg_count:
        # User typed something AND we haven't processed it yet
        prompt = user_input
    else:
        # No new input to process
        prompt = None
    
    # Process chat input
    if prompt:
        # Set processing flag to hide suggestions in message loop
        st.session_state.processing = True
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            # Step 1: Classify intent with verbose thinking
            if st.session_state.show_verbose:
                st.markdown('<div class="thinking-step">🧠 <b>Step 1:</b> Analyzing your question...</div>', unsafe_allow_html=True)
            
            intent = classify_intent(prompt, client, chat_models)
            
            if st.session_state.show_verbose:
                st.markdown(f'<div class="thinking-step">✅ <b>Intent detected:</b> {intent.replace("_", " ")}</div>', unsafe_allow_html=True)
            
            # Step 2: Perform search if needed
            matches, facts = [], []
            if intent == "SPECIFIC_SEARCH":
                if st.session_state.show_verbose:
                    st.markdown('<div class="thinking-step">🔍 <b>Step 2:</b> Searching databases...</div>', unsafe_allow_html=True)
                    
                    # Show sub-steps
                    search_progress = st.empty()
                    search_progress.markdown('<div class="thinking-step">📊 Querying Pinecone vector database...</div>', unsafe_allow_html=True)
                
                matches, facts = hybrid_search(prompt, client, embedding_model)
                
                if st.session_state.show_verbose:
                    search_progress.markdown(f'<div class="thinking-step">✅ Found <b>{len(matches)}</b> locations + <b>{len(facts)}</b> connections</div>', unsafe_allow_html=True)
            
            # Step 3: Generate response with streaming
            if st.session_state.show_verbose:
                st.markdown('<div class="thinking-step">✍️ <b>Step 3:</b> Generating personalized response...</div>', unsafe_allow_html=True)
            
            # Build conversation history for context (last 5 Q&A pairs = 10 messages)
            conversation_history = []
            if len(st.session_state.messages) > 1:  # Exclude current user message
                history_messages = st.session_state.messages[:-1][-10:]  # Last 10 messages (5 Q&A pairs)
                for msg in history_messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    conversation_history.append(f"{role}: {msg['content'][:150]}...")  # Truncate to 150 chars
            
            # Build prompt with conversation history
            prompt_messages = generate_response(prompt, intent, matches, facts, conversation_history)
            
            # Stream response
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                stream_resp, model_used = call_llm_with_fallback(
                    client=client,
                    chat_models=chat_models,
                    messages=prompt_messages,
                    max_tokens=800,
                    temperature=0.7,
                    stream=True
                )
                
                for chunk in stream_resp:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
                
                # Final response without cursor
                response_placeholder.markdown(full_response)
                response = full_response
                
            except Exception as e:
                response = f"⚠️ Error generating response: {e}"
                response_placeholder.markdown(response)
            
            # Show metadata
            metadata = {"intent": intent}
            
            if matches or facts:
                metadata["matches"] = matches
                metadata["facts"] = facts
                
                # Intent badge
                badge_class = "search"
                st.markdown(f'<span class="intent-badge {badge_class}">💭 {intent.replace("_", " ")}</span>', 
                          unsafe_allow_html=True)
                
                # Search results
                if matches:
                    with st.expander(f"🔍 Vector Search Results ({len(matches)} matches)"):
                        for m in matches:
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>{m['name']}</strong> ({m['type']})<br>
                                📍 {m['city']} | ⭐ Score: {m['score']:.2f}
                            </div>
                            """, unsafe_allow_html=True)
                
                if facts:
                    with st.expander(f"🕸️ Knowledge Graph Results ({len(facts)} facts)"):
                        for f in facts:
                            st.markdown(f"- **{f['name']}** ({f['type']}, {f.get('city', '')})")
            else:
                # Show intent badge for non-search queries
                badge_class = "greeting" if "GREETING" in intent else "general"
                st.markdown(f'<span class="intent-badge {badge_class}">💭 {intent.replace("_", " ")}</span>', 
                          unsafe_allow_html=True)
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "metadata": metadata,
                "suggestions": generate_suggested_questions(prompt, intent, matches)  # Store suggestions with message
            })
            
            # Update last message count
            st.session_state.last_msg_count = len(st.session_state.messages)
            
            # Show suggestions immediately after this response (inside the assistant chat message container)
            last_message = st.session_state.messages[-1]
            if "suggestions" in last_message and last_message["suggestions"]:
                st.markdown("---")
                st.markdown("**💡 Suggested questions:**")
                # Create 2x2 grid layout - same as message loop
                cols = st.columns(2)
                msg_index = len(st.session_state.messages) - 1
                
                for i, suggestion in enumerate(last_message["suggestions"]):
                    col_index = i % 2  # 0,1,0,1 for 4 buttons
                    with cols[col_index]:
                        # Use SAME key format as message loop buttons: sug_{msg_index}_{i}
                        unique_key = f"sug_{msg_index}_{i}"
                        clicked = st.button(suggestion, key=unique_key, use_container_width=True)
                        if clicked:
                            st.session_state.next_query = suggestion
                            # Mark this suggestion as used
                            st.session_state.used_suggestions.add(suggestion.lower())
                            st.rerun()
        
        # Clear processing flag - response complete
        st.session_state.processing = False
        
        # Clear prompt after processing so suggestions can show
        prompt = None

if __name__ == "__main__":
    main()
