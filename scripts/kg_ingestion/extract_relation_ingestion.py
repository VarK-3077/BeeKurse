"""
Extract Relation Ingestion - Optimized Pipeline with Batch Verification

This module provides an optimized pipeline for extracting relations from product
descriptions with semantic standardization and batch verification.

Pipeline (per product):
1. Agent1 (parallel) - Extract triplets from description chunks
2. Semantic Standardizer - Compare relations to VDB, standardize or add new
3. Batch Verification (1 LLM call) - Verify ALL triplets in one call (keep/drop)
4. Output - Add verified triplets to queue and Property VDB

Key Features:
- Semantic similarity for relation standardization (threshold: 0.85)
- Per-product batch verification (90-95% fewer LLM calls vs per-triplet)
- Property VDB population for search (format: {Type}:{Value})
- Type extraction from relation names (HAS_COLOR -> Color)

Output:
- Triplets queued to UnifiedIngestionQueue
- Properties added to Property VDB for search semantic matching
- Debug file with intermediate processing steps
"""

import asyncio
import re
import json
import os
import argparse
import itertools
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
# Import queue helpers
from unified_ingestion_queue import UnifiedIngestionQueue, get_global_queue, shutdown_global_queue

# Load environment variables from .env file in parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# --- Configuration ---
CUSTOM_RELATIONS_VDB_PATH = os.path.join(os.path.dirname(__file__), "custom_relations_vdb")
COLLECTION_NAME = "ecommerce_relations"
NUM_VERIFICATION_WORKERS = 3  # Number of parallel verification workers
SIMILARITY_THRESHOLD = 0.85   # Threshold for standardizing relations

# --- LLM Setup (API Key Rotation) ---
# Load all available NVIDIA API keys
API_KEYS = []
i = 1
while True:
    key = os.getenv(f"NVIDIA_API_KEY_{i}")
    if key:
        API_KEYS.append(key)
        i += 1
    else:
        break  # Stop when no more numbered keys are found

# Fallback to the original single key if no numbered keys are found
if not API_KEYS:
    key = os.getenv("NVIDIA_API_KEY")
    if key:
        API_KEYS.append(key)
    else:
        raise ValueError("No NVIDIA_API_KEY or NVIDIA_API_KEY_n environment variables found!")

print(f"[LLM Setup] Loaded {len(API_KEYS)} NVIDIA API Keys for rotation.")

# Create client pools for each LLM configuration
extractor_clients = [
    ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        api_key=key,
        temperature=0.2,
        top_p=0.85,
        max_completeion_tokens=65536,
        streaming=True
    ) for key in API_KEYS
]

# NOTE: Standardizer clients removed as Step 2 is now embedding-based

verifier_clients = [
    ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        api_key=key,
        temperature=0.0,
        top_p=0.95,
        max_completeion_tokens=65536,
        streaming=True
    ) for key in API_KEYS
]

# Create iterators that cycle through the clients
llm_extractor_cycler = itertools.cycle(extractor_clients)
llm_verifier_cycler = itertools.cycle(verifier_clients)

# --- Helper functions to get the next client from the pool ---
def get_next_extractor(_):
    client = next(llm_extractor_cycler)
    return client

def get_next_verifier(_):
    client = next(llm_verifier_cycler)
    return client


# --- Load Custom Relations VDB ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
custom_relations_vdb = Chroma(
    persist_directory=CUSTOM_RELATIONS_VDB_PATH,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)
print(f"[INGESTION] Custom Relations VDB loaded from {CUSTOM_RELATIONS_VDB_PATH}")

# --- Load Property VDB for search ---
# Property VDB stores properties in {Type}:{Value} format for semantic search
PROPERTY_VDB_PATH = os.path.join(os.path.dirname(__file__), "..", "property_vdb")
PROPERTY_COLLECTION_NAME = "product_properties"

# Initialize Property VDB (create directory if needed)
os.makedirs(PROPERTY_VDB_PATH, exist_ok=True)
property_vdb = Chroma(
    persist_directory=PROPERTY_VDB_PATH,
    embedding_function=embeddings,
    collection_name=PROPERTY_COLLECTION_NAME
)
print(f"[INGESTION] Property VDB loaded from {PROPERTY_VDB_PATH}")

# --- Data Schemas ---
class RawTriplet(BaseModel):
    subject: str
    relation: str
    object: str
    subject_type: str = Field(..., description="Inferred type of subject")
    object_type: str = Field(..., description="Inferred type of object")

class ExtractionOutput(BaseModel):
    triplets: List[RawTriplet] = Field(default_factory=list)

class VerificationResult(BaseModel):
    is_valid: bool = Field(..., description="Whether the relation is valid and supported by text")
    reason: str = Field(..., description="Specific reasoning for validation or rejection")


class BatchVerificationResult(BaseModel):
    """Result of batch verification - decisions for all triplets in one call."""
    decisions: List[str] = Field(
        ...,
        description="List of 'keep' or 'drop' decisions for each triplet in order"
    )

class StandardizedTriplet(BaseModel):
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    is_raw_relation: bool = False  # True if kept raw (not standardized)

# --- Prompts ---
EXTRACT_PROMPT = """You are an intelligent relation extraction agent specialized in product analysis.

CRITICAL: Output ONLY valid JSON. Do NOT include any thinking, reasoning, explanations, or XML tags. Start your response directly with the JSON object.

Your task is to analyze product descriptions and extract ALL meaningful relationships for product ID: {product_id}.

IMPORTANT EXTRACTION RULES:

1. PRIORITY: Extract ALL properties and features directly linked to the PRODUCT (use {product_id} as subject):
   - {product_id} → HAS_PROCESSOR → Snapdragon 8 Gen 3
   - {product_id} → HAS_RAM → 12GB
   - {product_id} → HAS_DISPLAY → 6.8 inch AMOLED
   - {product_id} → HAS_REFRESH_RATE → 120Hz  (NOT Display → HAS_REFRESH_RATE → 120Hz)
   - {product_id} → HAS_BATTERY → 5000mAh
   - {product_id} → HAS_RESOLUTION → 200MP
   - {product_id} → SUPPORTS → 45W fast charging

2. ONLY create component-as-subject relationships if the component has ADDITIONAL details beyond what's linked to the product:
   - If text says "Snapdragon 8 Gen 3 processor with 8 cores":
     * {product_id} → HAS_PROCESSOR → Snapdragon 8 Gen 3
     * Snapdragon 8 Gen 3 → HAS_CORES → 8
   - If text only says "120Hz refresh rate":
     * {product_id} → HAS_REFRESH_RATE → 120Hz (DON'T create Display node)

3. Use the actual product ID ({product_id}) as the subject, NOT generic "Product"

4. **CRITICAL - BE SPECIFIC WITH RELATION NAMES:**
   - Use UPPERCASE_UNDERSCORE format for relations
   - Create SPECIFIC relation names that describe the EXACT component/property
   - ❌ NEVER use generic relations like: HAS_FEATURE, HAS_PROPERTY, HAS_ATTRIBUTE, HAS_CHARACTERISTIC
   - ✅ ALWAYS use specific relations that name the component:
     * HAS_LUMBAR_SUPPORT (not HAS_FEATURE for lumbar support)
     * HAS_ARMREST_TYPE (not HAS_FEATURE for armrests)
     * HAS_BACK_MATERIAL (not HAS_FEATURE for mesh back)
     * HAS_CASTER_TYPE (not HAS_FEATURE for casters)
     * HAS_HEIGHT_ADJUSTMENT (not HAS_FEATURE for gas lift)
     * HAS_AIR_UNIT (not HAS_FEATURE for Max Air unit)
     * HAS_UPPER_MATERIAL (not HAS_FEATURE for mesh upper)
   - The relation name should identify WHAT the property is, not just that it's "a feature"

5. Assign precise node types: subject_type="product" for the product, object_type based on semantic meaning

**FASHION DOMAIN - COMMON RELATIONS:**
- HAS_MATERIAL → leather, rubber, cotton, mesh, polyester
- HAS_PATTERN → floral, striped, solid, printed
- HAS_FIT → slim fit, regular fit, loose fit
- HAS_CLOSURE → zipper, button, hook, velcro
- HAS_SOLE_MATERIAL → rubber, EVA, leather (footwear)
- HAS_FABRIC → cotton, silk, denim, wool (clothing)

**DO NOT EXTRACT (Common Noise):**
- ❌ Vague claims: "appears to be high-quality", "durable hardware" → SKIP unless specific material
- ❌ Packaging: "presented in protective packaging", "comes in gift box" → SKIP
- ❌ Generic suitability: "versatile for occasions" → SKIP (unless specific like "wedding attire")
- ❌ Marketing fluff: "comfort-focused construction" → SKIP unless specific feature

**VALIDATION RULES:**
- Product category: {category} - only extract category-appropriate features
- Trust metadata over description (e.g., if product color is "black" but description says "white", use black)
- Skip irrelevant features (e.g., "compartments" for flip flops, "battery" for wallets)

Extract EVERY specification and feature as product properties:
- Display specs (size, resolution, refresh rate, type)
- Processor, RAM, storage
- Battery capacity, charging speed
- Camera specs
- Materials, dimensions, weight
- Technologies, standards, compatibility

### TEXT TO ANALYZE:
{text}

Output JSON format:
{format_instructions}
"""

# NOTE: STANDARDIZE_PROMPT removed

VERIFY_PROMPT = """You are a fact-checking agent that validates the consistency and accuracy of extracted relations.

CRITICAL: Output ONLY valid JSON. Do NOT include any thinking, reasoning, explanations, or XML tags. Start your response directly with the JSON object.

Given a relation triplet (subject, relation, object) and the original text context, verify the following:

1. Does the source text explicitly support this relation?
2. Is the relation semantically coherent?
3. Are subject and object correctly identified?
4. Is there any contradiction with the source text?

Mark as valid only if ALL checks pass. Provide specific reasoning for rejection if invalid.

### TEXT CONTEXT:
{text}

### RELATION TO VERIFY:
Subject: {subject} (Type: {subject_type})
Relation: {relation}
Object: {object} (Type: {object_type})

Output JSON format:
{format_instructions}
"""

BATCH_VERIFY_PROMPT = """You are a fact-checking agent verifying multiple extracted relations for a product.

CRITICAL: Output ONLY valid JSON. Do NOT include any thinking, reasoning, explanations, or XML tags. Start your response directly with the JSON object.

For EACH triplet below, decide:
- "keep" if the relation is explicitly supported by the product description
- "drop" if it's a hallucination, contradiction, not supported by text, or nonsensical

### PRODUCT DESCRIPTION:
{description}

### TRIPLETS TO VERIFY (in order):
{triplets_list}

You MUST output exactly {num_triplets} decisions, one for each triplet in the same order.

Output JSON format:
{format_instructions}
"""

# --- Helper: Extract JSON from LLM output (handles thinking text) ---
def extract_json_from_output(text: str) -> str:
    """Extract JSON from LLM output that may contain <think> tags or other text."""
    import re
    import json as json_lib

    if not text:
        return "{}"

    # Strategy 1: Remove all <think>...</think> blocks (greedy to handle nested content)
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)

    # Strategy 2: If <think> exists without closing tag, take everything after </think> or after last >
    if '<think>' in text.lower() and '</think>' not in text.lower():
        # Try to find content after the thinking section ends (look for JSON start)
        json_start_match = re.search(r'[{\[]', text)
        if json_start_match:
            cleaned = text[json_start_match.start():]

    # Remove any remaining XML-like tags
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    cleaned = cleaned.strip()

    # Strategy 3: Find JSON object or array patterns
    # Try to find a complete JSON object
    json_obj_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    json_arr_pattern = r'\[(?:[^\[\]]|(?:\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]))*\]'

    # First try to find JSON object (most common case)
    matches = list(re.finditer(json_obj_pattern, cleaned, re.DOTALL))
    if matches:
        # Try from the last match (often the actual output, not examples in prompt)
        for match in reversed(matches):
            try:
                candidate = match.group(0)
                json_lib.loads(candidate)
                return candidate
            except:
                continue

    # Try array pattern (for batch results)
    arr_matches = list(re.finditer(json_arr_pattern, cleaned, re.DOTALL))
    if arr_matches:
        for match in reversed(arr_matches):
            try:
                candidate = match.group(0)
                json_lib.loads(candidate)
                return candidate
            except:
                continue

    # Strategy 4: Try the cleaned text as-is
    try:
        json_lib.loads(cleaned)
        return cleaned
    except:
        pass

    # Strategy 5: Last resort - look for JSON in original text
    for pattern in [json_obj_pattern, json_arr_pattern]:
        matches = list(re.finditer(pattern, text, re.DOTALL))
        for match in reversed(matches):
            try:
                candidate = match.group(0)
                json_lib.loads(candidate)
                return candidate
            except:
                continue

    # Return empty object if nothing found
    return "{}"

# --- Chains ---
extract_parser = JsonOutputParser(pydantic_object=ExtractionOutput)
extract_prompt = ChatPromptTemplate.from_messages([
    ("system", EXTRACT_PROMPT)
])
extract_chain = extract_prompt.partial(
    format_instructions=extract_parser.get_format_instructions()
) | RunnableLambda(get_next_extractor) | (lambda x: extract_json_from_output(x.content)) | extract_parser

# NOTE: Standardize Chain removed

verify_parser = JsonOutputParser(pydantic_object=VerificationResult)
verify_prompt = ChatPromptTemplate.from_messages([
    ("system", VERIFY_PROMPT)
])
verify_chain = verify_prompt.partial(
    format_instructions=verify_parser.get_format_instructions()
) | RunnableLambda(get_next_verifier) | (lambda x: extract_json_from_output(x.content)) | verify_parser

# Batch verification chain - verifies all triplets from a product in one call
batch_verify_parser = JsonOutputParser(pydantic_object=BatchVerificationResult)
batch_verify_prompt = ChatPromptTemplate.from_messages([
    ("system", BATCH_VERIFY_PROMPT)
])
batch_verify_chain = batch_verify_prompt.partial(
    format_instructions=batch_verify_parser.get_format_instructions()
) | RunnableLambda(get_next_verifier) | (lambda x: extract_json_from_output(x.content)) | batch_verify_parser

# --- Helper Functions ---
def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def adaptive_chunk_description(description: str) -> List[str]:
    length = len(description)
    if length < 200:
        return [description]
    elif length < 400:
        return split_into_sentences(description)
    else:
        sentences = split_into_sentences(description)
        if len(sentences) <= 3:
            return sentences
        else:
            grouped = []
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    grouped.append(f"{sentences[i]} {sentences[i+1]}")
                else:
                    grouped.append(sentences[i])
            return grouped

def add_relation_to_custom_vdb(relation_name: str, subject: str, object: str):
    """Add a new raw relation to custom VDB (learning mode)."""
    # Check if dry-run mode is enabled
    dry_run = os.getenv("DEBUG_DRY_RUN") == "true"

    try:
        description = f"Relation learned from product data: {subject} {relation_name} {object}"
        example = f"{subject} → {object}"

        if dry_run:
            print(f"  [VDB DRY-RUN] Would add new relation: {relation_name}")
        else:
            doc = Document(
                page_content=relation_name,
                metadata={
                    "relation": relation_name,
                    "description": description,
                    "category": "learned_from_products",
                    "examples": f'["{example}"]',
                    "type": "learned_relation",
                    "source": "extract_relation_ingestion"
                }
            )

            custom_relations_vdb.add_documents([doc])
            print(f"  [VDB] Added new relation: {relation_name}")
    except Exception as e:
        print(f"  [VDB ERROR] Failed to add relation: {e}")


def add_property_to_property_vdb(relation: str, property_value: str, product_id: str):
    """Add property to Property VDB for search semantic matching.

    Format: {Type}:{Value} (e.g., Color:Red, Size:Large, Brand:Nike)

    Args:
        relation: The relation name (e.g., HAS_COLOR, HAS_SIZE)
        property_value: The property value (e.g., Red, Large, Nike)
        product_id: The product ID for metadata
    """
    dry_run = os.getenv("DEBUG_DRY_RUN") == "true"

    try:
        # Extract type from relation name
        property_type = extract_type_from_relation(relation)
        property_string = f"{property_type}:{property_value}"

        # Check for near-duplicates to avoid VDB bloat
        existing = property_vdb.similarity_search_with_relevance_score(property_string, k=1)
        if existing and existing[0][1] > 0.98:
            # Near-exact match already exists
            return

        if dry_run:
            print(f"  [PROP VDB DRY-RUN] Would add: {property_string}")
        else:
            doc = Document(
                page_content=property_string,
                metadata={
                    "property_type": property_type.lower(),
                    "property_value": property_value,
                    "property_string": property_string,
                    "product_id": product_id,
                    "source": "extract_relation_ingestion"
                }
            )
            property_vdb.add_documents([doc])
            print(f"  [PROP VDB] Added: {property_string}")

    except Exception as e:
        print(f"  [PROP VDB ERROR] Failed to add property: {e}")


def extract_type_from_relation(relation: str) -> str:
    """Extract property type from relation name.

    Examples:
        HAS_COLOR -> Color
        HAS_SIZE -> Size
        HAS_REFRESH_RATE -> Refresh Rate
        SOLD_BY -> Store
        MADE_BY -> Brand
    """
    relation_upper = relation.upper()

    # Special cases for non-HAS_ relations
    if relation_upper in ("SOLD_BY", "AVAILABLE_AT"):
        return "Store"
    if relation_upper == "MADE_BY":
        return "Brand"
    if relation_upper == "BELONGS_TO":
        return "Category"
    if relation_upper == "COMPATIBLE_WITH":
        return "Compatibility"
    if relation_upper == "SUPPORTS":
        return "Feature"

    # Standard HAS_X pattern
    if relation_upper.startswith("HAS_"):
        type_part = relation[4:]  # Remove "HAS_"
        # Convert REFRESH_RATE to Refresh Rate
        return type_part.replace("_", " ").title()

    # Fallback: use the relation name itself, title-cased
    return relation.replace("_", " ").title()


def normalize_node_type(node_type: str, node_value: str, relation: str = "") -> str:
    """Normalize node type based on relation name.

    For subject_type: Keep 'product' if it's the product node.
    For object_type: Extract type from relation name (e.g., HAS_SIZE -> Size).
    """
    # If it's explicitly a product, keep it
    if node_type.lower() == "product":
        return "product"

    # If relation is provided, extract type from it
    if relation:
        return extract_type_from_relation(relation)

    return "Property"

def format_triplet_output(triplet: StandardizedTriplet, product_id: str, product_name: str, document_id: str, chunk_id: int, category: str = "Product") -> dict:
    sanitized_category = category.replace(' ', '_').replace('-', '_')

    return {
        "source": {
            "name": triplet.subject,
            "type": sanitized_category,
            "metadata": {
                "product_id": product_id,
                "document_id": document_id,
                "chunk_id": f"sentence_{chunk_id}"
            },
            "properties": {
                "prod_name": product_name,
                "product_id": product_id
            }
        },
        "relation": {
            "type": triplet.relation,
            "metadata": {
                "source": document_id
            }
        },
        "target": {
            "name": triplet.object,
            "type": triplet.object_type
        }
    }

# --- Pipeline Workers ---

async def agent1_extract_worker(
    chunk: str,
    chunk_id: int,
    product_id: str,
    category: str,
    queue1: asyncio.Queue,
    debug_file
):
    """Agent1: Extract relations from a sentence chunk (runs in parallel)."""
    import time
    start_time = time.time()

    max_retries = 2
    for attempt in range(max_retries):
        try:
            retry_suffix = ""
            if attempt > 0:
                retry_suffix = "\n\nIMPORTANT: You previously failed this extraction by outputting thinking/reasoning text instead of JSON. Output ONLY valid JSON."

            result = await extract_chain.ainvoke({
                "text": chunk + retry_suffix,
                "product_id": product_id,
                "category": category
            })
            
            if result is None:
                raise Exception("LLM returned None or invalid JSON")

            extraction = ExtractionOutput(**result)

            elapsed = time.time() - start_time
            debug_file.write(f"\n{'='*80}\n")
            debug_file.write(f"[AGENT1] Chunk {chunk_id} | Time: {elapsed:.2f}s | Attempt: {attempt + 1}\n")
            debug_file.write(f"[AGENT1] Extracted {len(extraction.triplets)} triplets:\n")
            for idx, triplet in enumerate(extraction.triplets, 1):
                debug_file.write(f"  {idx}. ({triplet.subject}:{triplet.subject_type}) -[{triplet.relation}]-> ({triplet.object}:{triplet.object_type})\n")
            debug_file.flush()

            for triplet in extraction.triplets:
                await queue1.put({
                    "triplet": triplet,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk
                })

            print(f"  [Agent1] Chunk {chunk_id}: Extracted {len(extraction.triplets)} triplets (attempt {attempt + 1}, {elapsed:.2f}s)")
            break

        except Exception as e:
            elapsed = time.time() - start_time
            if attempt < max_retries - 1:
                print(f"  [Agent1 RETRY] Chunk {chunk_id}: {str(e)[:100]}... (attempt {attempt + 1})")
                debug_file.write(f"\n[AGENT1 RETRY] Chunk {chunk_id} | Error: {e}\n")
                debug_file.flush()
            else:
                print(f"  [Agent1 ERROR] Chunk {chunk_id}: Failed - {str(e)[:100]}...")
                debug_file.write(f"\n[AGENT1 ERROR] Chunk {chunk_id} | Final error: {e}\n")
                debug_file.flush()
                if "NoneType" in str(e):
                    break


async def semantic_standardize_worker(
    worker_id: int,
    queue1: asyncio.Queue,
    queue2: asyncio.Queue,
    stop_event: asyncio.Event,
    debug_file
):
    """
    Worker: Semantic Similarity Check (Replaces Agent2).
    Compares raw relation embedding to VDB.
    - If score > 0.85 -> Use VDB relation.
    - If score <= 0.85 -> Use raw relation and ADD to VDB.
    """
    print(f"  [Worker{worker_id}] Semantic Similarity Standardizer started")

    while True:
        item = None
        try:
            # Wait for item from Queue1 with timeout
            item = await asyncio.wait_for(queue1.get(), timeout=0.1)

            import time
            step_start = time.time()

            triplet = item["triplet"]
            chunk_id = item["chunk_id"]
            chunk_text = item["chunk_text"]

            raw_relation = triplet.relation
            final_relation = raw_relation
            is_raw = True
            decision = "USE_RAW"
            similarity_score = 0.0
            matched_vdb_relation = None

            # --- Semantic Search Step with retry ---
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # Perform similarity search with relevance score
                    # Note: LangChain Chroma returns 0-1 score where 1 is best (higher = more similar)
                    results = custom_relations_vdb.similarity_search_with_relevance_score(raw_relation, k=1)

                    if results:
                        doc, score = results[0]
                        similarity_score = score
                        matched_vdb_relation = doc.metadata.get("relation", doc.page_content)

                        if score > SIMILARITY_THRESHOLD:
                            final_relation = matched_vdb_relation
                            is_raw = False
                            decision = "USE_STANDARD"
                    break  # Success, exit retry loop

                except Exception as search_error:
                    if attempt < max_retries - 1:
                        debug_file.write(f"[SEMANTIC RETRY] Worker{worker_id} | Chunk {chunk_id} | Attempt {attempt + 1} | Error: {search_error}\n")
                        debug_file.flush()
                        await asyncio.sleep(0.1)  # Brief delay before retry
                    else:
                        # Final attempt failed - use raw relation and continue
                        debug_file.write(f"[SEMANTIC ERROR] Worker{worker_id} | Chunk {chunk_id} | Using raw relation after {max_retries} attempts | Error: {search_error}\n")
                        debug_file.flush()
                        # Keep is_raw=True, final_relation=raw_relation (defaults)

            search_time = time.time() - step_start

            # Log decision
            debug_file.write(f"\n[SEMANTIC STANDARDIZE] Worker{worker_id} | Chunk {chunk_id}\n")
            debug_file.write(f"  Raw Relation: {raw_relation}\n")
            debug_file.write(f"  Top Match: {matched_vdb_relation} (Score: {similarity_score:.4f})\n")
            debug_file.write(f"  Decision: {decision} (Threshold: {SIMILARITY_THRESHOLD})\n")
            debug_file.write(f"  Final Relation: {final_relation}\n")
            debug_file.write(f"  Time: {search_time:.4f}s\n")
            debug_file.flush()

            # --- VDB Update Step ---
            # If we kept the raw relation, we add it to the VDB for future consistency
            if is_raw:
                add_relation_to_custom_vdb(final_relation, triplet.subject, triplet.object)
                debug_file.write(f"  Action: Added '{final_relation}' to VDB\n")

            debug_file.flush()

            # Step 4: Normalize node types - extract type from relation for object
            normalized_subject_type = normalize_node_type(triplet.subject_type, triplet.subject)
            normalized_object_type = normalize_node_type(triplet.object_type, triplet.object, final_relation)

            # Create standardized triplet
            std_triplet = StandardizedTriplet(
                subject=triplet.subject,
                subject_type=normalized_subject_type,
                relation=final_relation,
                object=triplet.object,
                object_type=normalized_object_type,
                is_raw_relation=is_raw
            )

            # Add to Queue2 for verification
            await queue2.put({
                "triplet": std_triplet,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text
            })

            queue1.task_done()

        except asyncio.TimeoutError:
            # No items in queue, check if we should stop
            if stop_event.is_set() and queue1.empty():
                print(f"  [Worker{worker_id}] Semantic Standardizer stopping")
                break
        except Exception as e:
            print(f"  [Worker{worker_id}] Semantic Error: {e}")
            debug_file.write(f"[SEMANTIC FATAL] Worker{worker_id} | Error: {e}\n")
            debug_file.flush()
            # Pass triplet through with raw relation to avoid data loss
            if item is not None:
                triplet = item["triplet"]
                chunk_id = item["chunk_id"]
                chunk_text = item["chunk_text"]
                # Create triplet with raw relation and pass through
                std_triplet = StandardizedTriplet(
                    subject=triplet.subject,
                    subject_type=normalize_node_type(triplet.subject_type, triplet.subject),
                    relation=triplet.relation,
                    object=triplet.object,
                    object_type=extract_type_from_relation(triplet.relation),
                    is_raw_relation=True
                )
                await queue2.put({
                    "triplet": std_triplet,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text
                })
                debug_file.write(f"[SEMANTIC FALLBACK] Passed through with raw relation: {triplet.relation}\n")
                debug_file.flush()
            queue1.task_done()

async def verification_worker(
    worker_id: int,
    queue2: asyncio.Queue,
    queue3: asyncio.Queue,
    stop_event: asyncio.Event,
    debug_file
):
    """Verification worker: Validates relations (runs in parallel, pulls from Queue2)."""
    print(f"  [VerifyWorker{worker_id}] Verification worker started")

    while True:
        try:
            item = await asyncio.wait_for(queue2.get(), timeout=0.1)

            import time
            verify_start = time.time()

            triplet = item["triplet"]
            chunk_text = item["chunk_text"]
            chunk_id = item["chunk_id"]

            max_verify_retries = 2
            verification = None

            for verify_attempt in range(max_verify_retries):
                try:
                    verify_text = chunk_text
                    if verify_attempt > 0:
                        verify_text += "\n\nIMPORTANT: Previous verification attempt failed. Output ONLY valid JSON."

                    verify_result = await verify_chain.ainvoke({
                        "text": verify_text,
                        "subject": triplet.subject,
                        "subject_type": triplet.subject_type,
                        "relation": triplet.relation,
                        "object": triplet.object,
                        "object_type": triplet.object_type
                    })
                    
                    if verify_result is None:
                        raise Exception("LLM returned None or invalid JSON")
                        
                    verification = VerificationResult(**verify_result)
                    verify_time = time.time() - verify_start

                    debug_file.write(f"[VERIFICATION] Worker{worker_id} | Chunk {chunk_id} | Time: {verify_time:.2f}s | Attempt: {verify_attempt + 1}\n")
                    debug_file.write(f"  Triplet: ({triplet.subject}) -[{triplet.relation}]-> ({triplet.object})\n")
                    debug_file.write(f"  Result: {'YES' if verification.is_valid else 'NO'}\n")
                    debug_file.write(f"  Reason: {verification.reason}\n\n")
                    debug_file.flush()
                    break

                except Exception as e:
                    verify_time = time.time() - verify_start
                    if verify_attempt < max_verify_retries - 1:
                        print(f"  [VerifyWorker{worker_id}] Retry {verify_attempt + 1}: {str(e)[:100]}...")
                        debug_file.write(f"[VERIFICATION RETRY] Worker{worker_id} | Chunk {chunk_id} | Error: {e}\n")
                        debug_file.flush()
                    else:
                        print(f"  [VerifyWorker{worker_id}] Failed after {max_verify_retries} attempts")
                        debug_file.write(f"[VERIFICATION ERROR] Worker{worker_id} | Chunk {chunk_id} | Error: {e}\n")
                        debug_file.flush()

            if verification and verification.is_valid:
                await queue3.put({
                    "triplet": triplet,
                    "chunk_id": chunk_id
                })
            elif verification:
                print(f"  [VerifyWorker{worker_id}] Rejected: {triplet.relation} - {verification.reason}")

            queue2.task_done()

        except asyncio.TimeoutError:
            if stop_event.is_set() and queue2.empty():
                print(f"  [VerifyWorker{worker_id}] Verification worker stopping")
                break

async def file_output_worker(
    queue3: asyncio.Queue,
    stop_event: asyncio.Event,
    product_id: str,
    product_name: str,
    document_id: str,
    queue: Optional[UnifiedIngestionQueue],
    entity_type_tracker: Dict[str, str],
    category: str = "Product"
) -> int:
    """File Output worker: Sequential writing from Queue3."""
    print("  [FileOutput] File output worker started")

    triplet_count = 0

    while True:
        try:
            item = await asyncio.wait_for(queue3.get(), timeout=0.1)

            triplet = item["triplet"]
            chunk_id = item["chunk_id"]

            if triplet.subject in entity_type_tracker:
                if entity_type_tracker[triplet.subject] != triplet.subject_type:
                    triplet.subject_type = entity_type_tracker[triplet.subject]
            else:
                entity_type_tracker[triplet.subject] = triplet.subject_type

            if triplet.object in entity_type_tracker:
                if entity_type_tracker[triplet.object] != triplet.object_type:
                    triplet.object_type = entity_type_tracker[triplet.object]
            else:
                entity_type_tracker[triplet.object] = triplet.object_type

            triplet_dict = format_triplet_output(triplet, product_id, product_name, document_id, chunk_id, category)

            # Add property to Property VDB for search semantic matching
            add_property_to_property_vdb(triplet.relation, triplet.object, product_id)

            if queue:
                await queue.add_triplet(triplet_dict, source_type="description")
                triplet_count += 1
                print(f"  [FileOutput] Queued: ({triplet.subject})-[{triplet.relation}]->({triplet.object})")
            else:
                print(f"  [WARN] No queue provided, skipping triplet for {product_id}")

            queue3.task_done()

        except asyncio.TimeoutError:
            if stop_event.is_set() and queue3.empty():
                print(f"  [FileOutput] File output worker stopping. Total triplets: {triplet_count}")
                break

    return triplet_count


async def batch_verify_triplets(
    triplets: List[StandardizedTriplet],
    description: str,
    debug_file
) -> List[StandardizedTriplet]:
    """Batch verify all triplets from a product in a single LLM call.

    Args:
        triplets: List of standardized triplets to verify
        description: Full product description for context
        debug_file: Debug file handle for logging

    Returns:
        List of verified triplets (only those marked as 'keep')
    """
    import time

    if not triplets:
        return []

    # Format triplets for the prompt
    triplets_list = "\n".join([
        f"{i+1}. ({t.subject}) -[{t.relation}]-> ({t.object})"
        for i, t in enumerate(triplets)
    ])

    verify_start = time.time()
    max_retries = 2
    verified_triplets = []

    for attempt in range(max_retries):
        try:
            result = await batch_verify_chain.ainvoke({
                "description": description,
                "triplets_list": triplets_list,
                "num_triplets": len(triplets)
            })

            if result is None:
                raise Exception("LLM returned None or invalid JSON")

            batch_result = BatchVerificationResult(**result)
            decisions = batch_result.decisions

            # Validate we got the right number of decisions
            if len(decisions) != len(triplets):
                debug_file.write(f"[BATCH VERIFY WARNING] Expected {len(triplets)} decisions, got {len(decisions)}\n")
                # Pad or truncate decisions
                if len(decisions) < len(triplets):
                    decisions = decisions + ["keep"] * (len(triplets) - len(decisions))
                else:
                    decisions = decisions[:len(triplets)]

            verify_time = time.time() - verify_start

            # Log batch verification
            debug_file.write(f"\n{'='*80}\n")
            debug_file.write(f"[BATCH VERIFICATION] Time: {verify_time:.2f}s | Triplets: {len(triplets)} | Attempt: {attempt + 1}\n")

            kept_count = 0
            dropped_count = 0

            for i, (triplet, decision) in enumerate(zip(triplets, decisions)):
                decision_lower = decision.lower().strip()
                if decision_lower == "keep":
                    verified_triplets.append(triplet)
                    kept_count += 1
                    debug_file.write(f"  {i+1}. KEEP: ({triplet.subject}) -[{triplet.relation}]-> ({triplet.object})\n")
                else:
                    dropped_count += 1
                    debug_file.write(f"  {i+1}. DROP: ({triplet.subject}) -[{triplet.relation}]-> ({triplet.object})\n")

            debug_file.write(f"[BATCH VERIFY RESULT] Kept: {kept_count}, Dropped: {dropped_count}\n")
            debug_file.flush()

            print(f"  [BatchVerify] Verified {len(triplets)} triplets: {kept_count} kept, {dropped_count} dropped ({verify_time:.2f}s)")
            break

        except Exception as e:
            verify_time = time.time() - verify_start
            if attempt < max_retries - 1:
                print(f"  [BatchVerify RETRY] Attempt {attempt + 1}: {str(e)[:100]}...")
                debug_file.write(f"[BATCH VERIFY RETRY] Attempt {attempt + 1} | Error: {e}\n")
                debug_file.flush()
            else:
                # On final failure, keep all triplets (fail-open)
                print(f"  [BatchVerify ERROR] Failed after {max_retries} attempts, keeping all triplets")
                debug_file.write(f"[BATCH VERIFY ERROR] Final error: {e} | Keeping all triplets\n")
                debug_file.flush()
                verified_triplets = triplets

    return verified_triplets


# --- Main Function ---

async def extract_relations_from_description(
    description: str,
    product_id: str,
    product_name: str,
    document_id: str,
    queue: Optional[UnifiedIngestionQueue],
    debug_file,
    category: str = "Product"
) -> int:
    """Extract relations from product description using batch verification.

    Pipeline:
    1. Agent1 extracts triplets from description chunks (parallel)
    2. Semantic Standardizer normalizes relations via VDB similarity
    3. Batch Verification - ALL triplets verified in ONE LLM call per product
    4. Output verified triplets and add properties to Property VDB
    """
    if not description or not description.strip():
        return 0

    print(f"\n[INGESTION] Processing product {product_id}")

    debug_file.write(f"\n{'='*80}\n")
    debug_file.write(f"[START] Product: {product_id} | Document: {document_id}\n")
    debug_file.write(f"[START] Timestamp: {datetime.utcnow().isoformat()}Z\n")
    debug_file.flush()

    sentences = adaptive_chunk_description(description)
    print(f"[INGESTION] Chunked into {len(sentences)} chunks")

    # --- Phase 1: Extraction ---
    queue1 = asyncio.Queue()  # Agent1 → Semantic Standardizer
    collected_triplets: List[StandardizedTriplet] = []  # Collect all standardized triplets
    stop_event = asyncio.Event()

    # Start Semantic Standardizer worker that collects into list instead of queue
    async def collecting_semantic_worker():
        """Modified semantic worker that collects triplets instead of passing to queue."""
        print(f"  [Worker0] Semantic Similarity Standardizer started (collecting mode)")

        while True:
            item = None
            try:
                item = await asyncio.wait_for(queue1.get(), timeout=0.1)

                import time
                step_start = time.time()

                triplet = item["triplet"]
                chunk_id = item["chunk_id"]

                raw_relation = triplet.relation
                final_relation = raw_relation
                is_raw = True
                decision = "USE_RAW"
                similarity_score = 0.0
                matched_vdb_relation = None

                # Semantic Search Step with retry
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        results = custom_relations_vdb.similarity_search_with_relevance_score(raw_relation, k=1)

                        if results:
                            doc, score = results[0]
                            similarity_score = score
                            matched_vdb_relation = doc.metadata.get("relation", doc.page_content)

                            if score > SIMILARITY_THRESHOLD:
                                final_relation = matched_vdb_relation
                                is_raw = False
                                decision = "USE_STANDARD"
                        break

                    except Exception as search_error:
                        if attempt < max_retries - 1:
                            debug_file.write(f"[SEMANTIC RETRY] Attempt {attempt + 1} | Error: {search_error}\n")
                            debug_file.flush()
                            await asyncio.sleep(0.1)
                        else:
                            debug_file.write(f"[SEMANTIC ERROR] Using raw relation | Error: {search_error}\n")
                            debug_file.flush()

                search_time = time.time() - step_start

                # Log decision
                debug_file.write(f"\n[SEMANTIC STANDARDIZE] Chunk {chunk_id}\n")
                debug_file.write(f"  Raw: {raw_relation} | Match: {matched_vdb_relation} (Score: {similarity_score:.4f})\n")
                debug_file.write(f"  Decision: {decision} | Final: {final_relation} | Time: {search_time:.4f}s\n")
                debug_file.flush()

                # VDB Update Step
                if is_raw:
                    add_relation_to_custom_vdb(final_relation, triplet.subject, triplet.object)

                # Normalize node types
                normalized_subject_type = normalize_node_type(triplet.subject_type, triplet.subject)
                normalized_object_type = normalize_node_type(triplet.object_type, triplet.object, final_relation)

                # Create and collect standardized triplet
                std_triplet = StandardizedTriplet(
                    subject=triplet.subject,
                    subject_type=normalized_subject_type,
                    relation=final_relation,
                    object=triplet.object,
                    object_type=normalized_object_type,
                    is_raw_relation=is_raw
                )
                collected_triplets.append(std_triplet)

                queue1.task_done()

            except asyncio.TimeoutError:
                if stop_event.is_set() and queue1.empty():
                    print(f"  [Worker0] Semantic Standardizer stopping. Collected {len(collected_triplets)} triplets")
                    break
            except Exception as e:
                print(f"  [Worker0] Semantic Error: {e}")
                debug_file.write(f"[SEMANTIC FATAL] Error: {e}\n")
                debug_file.flush()
                # Pass triplet through with raw relation to avoid data loss
                if item is not None:
                    triplet = item["triplet"]
                    std_triplet = StandardizedTriplet(
                        subject=triplet.subject,
                        subject_type=normalize_node_type(triplet.subject_type, triplet.subject),
                        relation=triplet.relation,
                        object=triplet.object,
                        object_type=extract_type_from_relation(triplet.relation),
                        is_raw_relation=True
                    )
                    collected_triplets.append(std_triplet)
                queue1.task_done()

    # Start workers
    semantic_worker_task = asyncio.create_task(collecting_semantic_worker())

    # Start Agent1 extraction tasks
    extraction_tasks = [
        asyncio.create_task(agent1_extract_worker(chunk, idx, product_id, category, queue1, debug_file))
        for idx, chunk in enumerate(sentences)
    ]

    await asyncio.gather(*extraction_tasks, return_exceptions=True)
    print("[INGESTION] Agent1 extraction complete")

    await queue1.join()
    stop_event.set()
    await semantic_worker_task
    print(f"[INGESTION] Semantic Standardizer complete. Collected {len(collected_triplets)} triplets")

    # --- Phase 2: Batch Verification (ONE LLM call for ALL triplets) ---
    if collected_triplets:
        verified_triplets = await batch_verify_triplets(collected_triplets, description, debug_file)
    else:
        verified_triplets = []

    print(f"[INGESTION] Batch verification complete. {len(verified_triplets)} triplets verified")

    # --- Phase 3: Output and Property VDB ---
    entity_type_tracker: Dict[str, str] = {}
    triplet_count = 0

    for i, triplet in enumerate(verified_triplets):
        # Entity type consistency tracking
        if triplet.subject in entity_type_tracker:
            if entity_type_tracker[triplet.subject] != triplet.subject_type:
                triplet.subject_type = entity_type_tracker[triplet.subject]
        else:
            entity_type_tracker[triplet.subject] = triplet.subject_type

        if triplet.object in entity_type_tracker:
            if entity_type_tracker[triplet.object] != triplet.object_type:
                triplet.object_type = entity_type_tracker[triplet.object]
        else:
            entity_type_tracker[triplet.object] = triplet.object_type

        triplet_dict = format_triplet_output(triplet, product_id, product_name, document_id, i, category)

        # Add property to Property VDB for search semantic matching
        add_property_to_property_vdb(triplet.relation, triplet.object, product_id)

        if queue:
            await queue.add_triplet(triplet_dict, source_type="description")
            triplet_count += 1
            print(f"  [Output] Queued: ({triplet.subject})-[{triplet.relation}]->({triplet.object})")
        else:
            print(f"  [WARN] No queue provided, skipping triplet for {product_id}")

    print(f"[INGESTION] Complete. Wrote {triplet_count} valid triplets")
    debug_file.write(f"\n{'='*80}\n")
    debug_file.write(f"[COMPLETE] Total triplets: {triplet_count}\n")
    debug_file.flush()

    return triplet_count

async def extract_relations_from_directory(
    directory_path: str,
    queue: Optional[UnifiedIngestionQueue] = None
) -> int:
    import glob
    import json

    json_files = glob.glob(os.path.join(directory_path, "*.json"))

    if not json_files:
        print("⚠ No JSON files found for relation extraction")
        return 0

    print(f"Found {len(json_files)} JSON files for relation extraction\n")
    
    if queue is None:
        print("[WARN] No queue provided to extract_relations_from_directory, getting global instance.")
        queue = await get_global_queue()

    os.makedirs("output", exist_ok=True)
    debug_file = open("output/extraction_debug.log", "w", encoding="utf-8")

    total_triplets = 0

    try:
        for i, filepath in enumerate(json_files, 1):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    if len(data) == 0: continue
                    data = data[0]

                description = data.get('description') or data.get('descrption', '')
                product_id = data.get('product_id', 'UNKNOWN')
                product_name = data.get('prod_name') or data.get('product_name', 'Unknown')
                category = data.get('category', 'Product')
                metadata = data.get('metadata', {})
                document_id = metadata.get('document_id', product_id)

                if not description.strip():
                    continue

                count = await extract_relations_from_description(
                    description,
                    product_id,
                    product_name,
                    document_id,
                    queue,
                    debug_file,
                    category
                )

                total_triplets += count

            except Exception as e:
                print(f"✗ Error processing {filepath}: {e}")
                continue

    finally:
        debug_file.close()

    return total_triplets


# --- Example Usage ---
async def main_test():
    import glob

    parser = argparse.ArgumentParser(description="Extract relations from product descriptions")
    parser.add_argument("input", nargs="?", default=None,
                       help="Input directory or single JSON file")
    parser.add_argument("--debug-output", default="debug.json",
                       help="Output file for debug logs")
    parser.add_argument("--product-id", default="TEST-001",
                       help="Product ID for test mode")
    parser.add_argument("--document-id", default="doc_test_001",
                       help="Document ID for test mode")

    args = parser.parse_args()

    print("="*80)
    print("RELATION EXTRACTION DEBUG PIPELINE (SEMANTIC STANDARDIZATION)")
    print("="*80)

    queue = await get_global_queue()

    with open(args.debug_output, 'w') as debug_f:
        debug_f.write(f"Extraction Debug Log\n")
        
        total_triplets = 0

        if args.input:
            if os.path.isdir(args.input):
                pattern = os.path.join(args.input, "*.json")
                json_files = sorted(glob.glob(pattern))

                if not json_files:
                    await shutdown_global_queue()
                    exit(1)

                for filepath in json_files:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    metadata = data.get('metadata', {})
                    description = data.get('description') or data.get('descrption', '')
                    product_id = data.get('product_id', 'UNKNOWN')
                    document_id = metadata.get('document_id', product_id)
                    product_name = data.get('prod_name') or data.get('product_name', 'Unknown')
                    category = data.get('category', 'Product')

                    if not description.strip(): continue

                    count = await extract_relations_from_description(
                        description, product_id, product_name, document_id, queue, debug_f, category
                    )
                    total_triplets += count

            elif os.path.isfile(args.input):
                with open(args.input, 'r') as f:
                    data = json.load(f)
                metadata = data.get('metadata', {})
                description = data.get('description') or data.get('descrption', '')
                product_id = data.get('product_id', args.product_id)
                product_name = data.get('prod_name') or data.get('product_name', 'Unknown')
                document_id = metadata.get('document_id', args.document_id)
                category = data.get('category', 'Product')

                total_triplets = await extract_relations_from_description(
                    description, product_id, product_name, document_id, queue, debug_f, category
                )
        else:
            test_description = """
            Premium flagship smartphone with 6.8-inch Dynamic AMOLED display.
            Features 120Hz refresh rate.
            Powered by Snapdragon 8 Gen 3 processor.
            """
            total_triplets = await extract_relations_from_description(
                test_description, args.product_id, "Test Product", args.document_id, queue, debug_f
            )

    print("\n" + "="*80)
    print(f"EXTRACTION COMPLETE. Total triplets queued: {total_triplets}")
    print("="*80)
    
    await shutdown_global_queue()

if __name__ == "__main__":
    asyncio.run(main_test())