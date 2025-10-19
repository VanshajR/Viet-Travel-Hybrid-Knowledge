# Deployment Guide - VietTravel AI

Deploy the Streamlit web app to production.

---

## Streamlit Community Cloud (Recommended - FREE)

Easiest option with permanent free hosting.

### Setup Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/viettravel-ai.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit https://share.streamlit.io
   - Click "New app"
   - Select your repo and branch
   - Set main file: `streamlit_app.py`
   - Click "Deploy"

3. **Configure Secrets**
   
   In app settings → Secrets, add:
   ```toml
   # LLM Provider
   LLM_PROVIDER = "groq"
   
   # API Keys
   GROQ_API_KEY = "gsk_..."
   PINECONE_API_KEY = "pcsk_..."
   PINECONE_INDEX_NAME = "vietnam-travel"
   
   # Neo4j (optional)
   NEO4J_URI = "bolt://your-neo4j-uri:7687"
   NEO4J_USER = "neo4j"
   NEO4J_PASSWORD = "your_password"
   ```

4. **Done!** Your app is live at `https://your-app.streamlit.app`

**Features:**
- ✅ Free forever
- ✅ Auto-deploys on git push
- ✅ Built-in secrets management
- ✅ HTTPS included
- ✅ No server maintenance

---

## Alternative: Render

Free tier with 750 hours/month.

### Setup Steps

1. **Create account** at https://render.com

2. **New Web Service** → Connect your GitHub repo

3. **Configure**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
   - Environment: Python 3

4. **Add Environment Variables**:
   ```
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_...
   PINECONE_API_KEY=pcsk_...
   PINECONE_INDEX_NAME=vietnam-travel
   NEO4J_URI=bolt://...
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=...
   ```

5. **Deploy** - Render handles the rest

**Note:** First load may be slow (cold start). Upgrade to $7/month for instant loading.

---

## Alternative: Railway

$5 free credit, then pay-as-you-go.

### Setup Steps

1. **Create account** at https://railway.app

2. **New Project** → Deploy from GitHub

3. **Add Environment Variables** (same as Render)

4. **Set Start Command**:
   ```bash
   streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

5. **Deploy** - Railway auto-detects Python and installs deps

**Cost:** ~$5-10/month after free credit runs out.

---

## Docker Deployment

For self-hosting or custom infrastructure.

### Dockerfile

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t viettravel-ai .

# Run container
docker run -p 8501:8501 \
  -e LLM_PROVIDER=groq \
  -e GROQ_API_KEY=gsk_... \
  -e PINECONE_API_KEY=pcsk_... \
  -e PINECONE_INDEX_NAME=vietnam-travel \
  -e NEO4J_URI=bolt://... \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=... \
  viettravel-ai
```

Access at http://localhost:8501

---

## Pre-Deployment Checklist

Before deploying:

- [ ] Data loaded to Pinecone (`python pinecone_upload.py`)
- [ ] Graph loaded to Neo4j (`python load_to_neo4j.py`)
- [ ] All API keys obtained and tested
- [ ] `.env` file in `.gitignore`
- [ ] Tested locally (`streamlit run streamlit_app.py`)
- [ ] Committed to git repository

---

## Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `LLM_PROVIDER` | Yes | groq or openai | `groq` |
| `GROQ_API_KEY` | If using Groq | Groq API key | `gsk_...` |
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key | `sk-proj-...` |
| `PINECONE_API_KEY` | Yes | Pinecone API key | `pcsk_...` |
| `PINECONE_INDEX_NAME` | Yes | Index name | `vietnam-travel` |
| `NEO4J_URI` | No | Neo4j connection | `bolt://localhost:7687` |
| `NEO4J_USER` | No | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | No | Neo4j password | `your_password` |

**Note:** Vector dimensions auto-configure based on `LLM_PROVIDER` (384 for Groq, 1536 for OpenAI).

---

## Security Best Practices

1. **Never commit secrets** - Add `.env` to `.gitignore`
2. **Use platform secrets** - Each platform has secure secret storage
3. **Rotate keys** - Change API keys periodically
4. **HTTPS only** - All platforms provide HTTPS automatically
5. **Monitor usage** - Check API usage dashboards regularly

---

## Cost Optimization

**Free Tier Setup (Recommended):**
- Hosting: Streamlit Cloud (free)
- LLM: Groq (free)
- Embeddings: HuggingFace via Groq (free)
- Vector DB: Pinecone free tier (100K vectors)
- Graph DB: Neo4j Aura free tier

**Total: $0/month** for moderate usage

**Paid Setup (Higher limits):**
- Hosting: Render ($7/month) or Railway (~$5-10/month)
- LLM: OpenAI GPT-4o-mini (~$50/month)
- Pinecone: Pod-based (~$70/month for dedicated)
- Neo4j: Professional (~$65/month)

**Total: ~$200/month** for production scale

---

## Performance Expectations

**Free Tier:**
- Response time: 2-5 seconds
- Concurrent users: 5-10
- Monthly requests: ~10,000
- Cold start: 10-30 seconds (Render/Railway)
- Zero cold start: Streamlit Cloud

**Paid Tier:**
- Response time: 1-3 seconds
- Concurrent users: 50+
- Monthly requests: unlimited (within API limits)
- Always-on instances

---

## Troubleshooting

**"Module not found" errors:**
- Ensure `requirements.txt` lists all dependencies
- Platform may need `streamlit` explicitly listed

**Slow first load:**
- HuggingFace model downloads on first run (~80MB for Groq)
- Consider using OpenAI embeddings API to avoid model download

**Connection timeouts:**
- Check Neo4j firewall rules
- Verify API keys are correct
- Ensure services are running (Neo4j, Pinecone index exists)

**Out of memory:**
- Streamlit Cloud: 1GB limit - use OpenAI embeddings instead of local models
- Upgrade to paid tier for more RAM
- Reduce `top_k` results to use less memory

---

## Monitoring

Track these metrics:

- **API usage**: Check Groq/OpenAI dashboards
- **Error rates**: Monitor Streamlit Cloud logs
- **Response times**: User feedback
- **Database limits**: Pinecone vector count, Neo4j nodes
- **Costs**: If using paid services

---

## Scaling Tips

When you need to scale:

1. **Cache responses** - Add Redis for repeated queries
2. **Upgrade tiers** - Move to paid hosting for better performance  
3. **Load balancing** - Deploy multiple instances behind load balancer
4. **Database optimization** - Index Neo4j properties, use Pinecone pods
5. **CDN** - Cloudflare for static assets

---

## What's Next

After deployment:

- Monitor user feedback
- Track popular queries
- A/B test different prompts
- Add analytics (Google Analytics, Mixpanel)
- Implement user accounts
- Add conversation history persistence
- Enable social sharing

---

**Recommended:** Start with Streamlit Cloud (free) + Groq (free) for zero-cost deployment, then upgrade based on usage.
