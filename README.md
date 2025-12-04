# BeeKurse

**Conversational E-commerce Platform**

A natural language product search and recommendation system built on top of [KURSE](./KURSE) - our neuro-symbolic inference engine.

## Demo Videos

| User Demo | Vendor Demo |
|:---------:|:-----------:|
| [Watch User Demo](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/vishnutejas_iisc_ac_in/IgCenVPbqPGdRr3i8FvoIkjDAVbtFSs3InCEhxSGvSguu0U?e=wJKD61) | [Watch Vendor Demo](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/vishnutejas_iisc_ac_in/IgCenVPbqPGdRr3i8FvoIkjDAVbtFSs3InCEhxSGvSguu0U?e=wJKD61) |

---

## Overview

BeeKurse is a conversational e-commerce platform that excels in product recommendations through natural language understanding. Key capabilities include:

- **Natural Language Search**: Find products by describing what you want in plain English
- **WhatsApp Integration**: Shop directly through WhatsApp messaging
- **Small Vendor Support**: Vendors can update inventory by simply photographing handwritten notes - OCR handles the rest
- **Smart Recommendations**: VDB/KG structure enables intelligent product suggestions and comparisons

## Architecture

BeeKurse uses a **tri-store architecture** for intelligent data storage and retrieval:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   SQLite    │     │   Qdrant    │     │  Memgraph   │
│  (SQL DB)   │     │   (VDB)     │     │    (KG)     │
└─────────────┘     └─────────────┘     └─────────────┘
       │                  │                   │
       └──────────────────┼───────────────────┘
                         │
              ┌──────────┴──────────┐
              │   Search Orchestrator│
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              │   Strontium Parser   │
              └──────────┬──────────┘
                         │
                    User Query
```

The core inference engine is powered by [KURSE](./KURSE).

## Key Features

### Query Classification
The Strontium LLM Parser classifies queries into 5 types:
- **SEARCH** - Product discovery ("red cotton shirt under $30")
- **DETAIL** - Product info Q&A ("compare materials of p-123 and p-456")
- **CHAT** - Conversational ("Hello!", "How does this work?")
- **CART_ACTION** - Add/remove items ("add p-123 to my cart")
- **CART_VIEW** - View cart/wishlist ("what's in my cart?")

### Search Orchestration
Four parallel scoring paths for optimal results:
- **HQ (Hurry Query)** - Fast path for repeat purchases
- **Property Search (RQ)** - VDB similarity + KG relation matching
- **Connected Search (SQ)** - Find related products via KG traversal
- **Subcategory Scoring** - Type-matching bonus

### Intelligent Scoring
- Property weights (0.5 = nice-to-have, 2.0 = must-have)
- Connected bonus (+0.5) for KG-related products
- Subcategory bonus (up to +0.4) for type matches
- Superlative handling (70% relevance, 30% literal value for "cheapest", "best rated")

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, FastAPI |
| Frontend | React (vendor portal), WhatsApp UI |
| SQL Database | SQLite |
| Vector Database | Qdrant |
| Knowledge Graph | Memgraph |
| LLM | NVIDIA NIM APIs |
| OCR | olmOCR (Qwen 7B) |

## Project Structure

```
BeeKurse/
├── backend/           # FastAPI backend server
├── search_agent/      # Search orchestration & Strontium parser
├── vendor_frontend/   # React vendor portal
├── whatsapp-frontend/ # WhatsApp chat interface
├── config/            # Configuration files
├── data/              # User data & product data
├── KURSE/             # Core inference engine (submodule)
└── Documentation/     # Reports and presentations
```

## Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker (for Qdrant and Memgraph)
- NVIDIA NIM API access

### Environment Setup

1. Clone the repository:
```bash
git clone <repo-url>
cd BeeKurse
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start database services:
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start Memgraph
docker run -p 7687:7687 memgraph/memgraph
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your NVIDIA NIM API keys
```

5. Run the backend:
```bash
cd backend
uvicorn main:app --reload
```

6. Run the frontend (vendor portal):
```bash
cd vendor_frontend
npm install
npm start
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | Main search endpoint |
| `/chat` | POST | Conversational chat |
| `/cart` | GET/POST | Cart operations |
| `/products` | GET | Product listing |
| `/vendor/upload` | POST | Vendor inventory upload |

## Team

- Abhinav Goyal
- Bhuvan
- Himesh
- Kunjan Manoj
- Siripuru Abhiram
- Varshith Kada
- Vishnu Teja
