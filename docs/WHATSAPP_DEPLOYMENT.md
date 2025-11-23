# WhatsApp Deployment Guide

Complete guide to deploy Strontium with WhatsApp integration.

---

## Architecture Overview

```
WhatsApp User
    ↓
Meta WhatsApp Cloud API
    ↓
whatsapp_bot.py (Webhook receiver)
    ↓
backend_strontium.py (Strontium + Orchestrator)
    ↓
Reply sent back to user
```

---

## Prerequisites

### 1. Python Dependencies

```bash
pip install fastapi uvicorn pydantic requests python-dotenv openai
```

### 2. WhatsApp Setup

You already have:
- ✅ WhatsApp Business account
- ✅ Access token: `WHATSAPP_TOKEN`
- ✅ Phone number ID: `WHATSAPP_PHONE_NUMBER_ID`
- ✅ Test recipient number

### 3. Environment Variables

Your `.env` file:
```env
# WhatsApp Cloud API credentials
WHATSAPP_TOKEN=EAATgrBxTv7MBPzCaXHK9dZCzMMT0baoVCJZCWjrfdvQphTJ6ECZCMpGPtlx0biSfcaZAK1cnWfdEw8z8nXhKpQgOBRdnkHgW5W2jDzmLjyUxPFRCFJipmWCGAHZAk7ZA76NEI3kGWW4xBigbOy2UtjPp9xO9QRQtSVreLFAgJ8ZAloTlG6X0FpPe2RBpDrywy05cilq9xxGO4wEGzBjjlEcWcxxQ4edvFiNsiBgb6ZBhAAZDZD
WHATSAPP_PHONE_NUMBER_ID=786795584527185
TEST_RECIPIENT_NUMBER=917893127444

# Webhook verification
VERIFY_TOKEN=my_verify_token_123

# Backend URL (change for production deployment)
BACKEND_URL=http://localhost:5001/process
```

---

## Local Testing

### Step 1: Start Backend

```bash
cd "/home/varshith_kada/Files/kurse_local/Search - Orchestrator/The Whole Story/test & debug"
python3 backend_strontium.py
```

**Expected output:**
```
🚀 Initializing Strontium Backend...
✅ Strontium Backend initialized successfully!

================================================================================
🚀 Starting Strontium WhatsApp Backend
================================================================================
📍 Endpoint: http://localhost:5001/process
🔧 NVIDIA API: Enabled
💾 User Data: /home/varshith_kada/.../mock_user_data
================================================================================

INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:5001
```

### Step 2: Test Backend (Without WhatsApp)

In a new terminal:

```bash
python3 test_backend.py
```

This tests:
- ✅ Chat queries ("Hello!", "What can you do?")
- ✅ Search queries ("Red cotton shirt under $30")
- ✅ Detail queries ("What material is p-001 made of?")
- ✅ User context enrichment

**Troubleshooting:**
- If connection fails: Check backend is running on port 5001
- If tests fail: Check error messages for missing dependencies

### Step 3: Start WhatsApp Webhook (Separate Terminal)

```bash
python3 whatsapp_bot.py
```

**Expected output:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Test End-to-End with ngrok

WhatsApp needs a public URL to send webhooks. Use ngrok:

```bash
# In a new terminal
ngrok http 8000
```

Copy the ngrok URL (e.g., `https://abc123.ngrok.io`)

**Configure Meta Webhook:**
1. Go to Meta Developer Portal
2. WhatsApp > Configuration > Webhooks
3. Edit webhook URL: `https://abc123.ngrok.io/webhook`
4. Verify token: `my_verify_token_123`
5. Subscribe to `messages` events

### Step 5: Send Test Messages

Send WhatsApp messages to your test number:

**Chat test:**
```
Hello!
```
Expected: Strontium greeting

**Search test:**
```
I need a red cotton shirt under $30
```
Expected: List of products with prices

**Detail test:**
```
What material is p-001 made of?
```
Expected: Product details

---

## Production Deployment

### Option 1: Deploy to Render/Railway/Fly.io

1. **Create requirements.txt:**
```txt
fastapi
uvicorn
pydantic
requests
python-dotenv
openai
sentence-transformers
chromadb
neo4j
```

2. **Create Procfile (for Railway):**
```
backend: python backend_strontium.py
webhook: python whatsapp_bot.py
```

3. **Set environment variables in platform:**
- All variables from `.env`
- Update `BACKEND_URL` to production backend URL

4. **Deploy both services**

### Option 2: Deploy to AWS/GCP/Azure

1. **Containerize with Docker:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports
EXPOSE 5001 8000

CMD ["uvicorn", "backend_strontium:app", "--host", "0.0.0.0", "--port", "5001"]
```

2. **Deploy to cloud platform**

3. **Update Meta webhook** to production URL

---

## Configuration for Production

### 1. Update config.py

```python
# Use environment variables for production
USE_NVIDIA_LLM: bool = os.getenv("USE_NVIDIA_LLM", "True").lower() == "true"
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY")

# Production database paths
SQL_DB_PATH: str = os.getenv("SQL_DB_PATH", "/app/databases/inventory.db")
USER_CONTEXT_DATA_DIR: str = os.getenv("USER_CONTEXT_DATA_DIR", "/app/user_data")
```

### 2. User Data Management

**For new users:**
- Backend automatically creates user profile using phone number
- Uses default preferences until user history is built

**For existing users:**
- Phone number maps to user_id
- Loads category-based preferences and purchase history
- Enriches queries with user context

### 3. Database Setup

**Production databases:**
- SQL: Product inventory
- Vector DB: Product embeddings
- Memgraph: Knowledge graph
- User data: JSON files or database

---

## Monitoring & Debugging

### Logs

Backend logs show:
```
📩 Incoming: [917893127444] Hello!
🔍 Parsing query with Strontium...
📊 Query type: chat
💬 Handling chat...
✅ Reply: Hello! I'm Strontium...
```

### Common Issues

**1. Backend connection refused**
- Check backend is running: `curl http://localhost:5001/`
- Check port 5001 is not in use: `lsof -i :5001`

**2. WhatsApp not receiving responses**
- Check ngrok is running
- Verify webhook URL in Meta portal
- Check whatsapp_bot.py logs for errors

**3. NVIDIA API errors**
- Verify API key is valid
- Check rate limits
- Fallback to mock mode: Set `USE_NVIDIA_LLM=False`

**4. User context not working**
- Check user_profiles.json exists
- Verify phone number mapping
- Check mock_user_data path in config

---

## Response Format Examples

### Chat Response
```
Hello! I'm Strontium, your curator at BeeKurse. How can I help you find what you need today?
```

### Search Response
```
🔍 Found 5 products:

1. *Classic Red Cotton Shirt*
   $29.99 | ⭐4.5
   Store: BeeKurse Main
   ID: p-001

2. *Premium Polo Shirt*
   $34.99 | ⭐5.0
   Store: Fashion Hub
   ID: p-002

... and 3 more results

💡 Tip: Ask about a specific product using its ID
Example: "What material is p-001 made of?"
```

### Detail Response
```
📦 Product Information: p-001

Material: 100% Cotton
Care Instructions: Machine washable
Available Colors: Red, Blue, Black
Size Range: S-XXL
```

---

## Testing Checklist

Before going live:

- [ ] Backend starts without errors
- [ ] Test script passes all tests
- [ ] WhatsApp webhook receives messages
- [ ] Chat queries respond correctly
- [ ] Search queries return products
- [ ] Detail queries work
- [ ] User context enrichment works (test with user-001)
- [ ] Error handling works (test invalid queries)
- [ ] NVIDIA API works (if enabled)
- [ ] Production databases are populated

---

## Quick Start Commands

```bash
# Terminal 1: Start backend
python3 backend_strontium.py

# Terminal 2: Test backend
python3 test_backend.py

# Terminal 3: Start webhook
python3 whatsapp_bot.py

# Terminal 4: Tunnel for WhatsApp
ngrok http 8000
```

---

## Support

**Backend issues:** Check `backend_strontium.py` logs
**WhatsApp issues:** Check `whatsapp_bot.py` logs
**Meta issues:** Check Meta Developer Portal > WhatsApp > Webhooks

**Need help?**
- Review error logs
- Test each component separately
- Check all environment variables are set correctly
