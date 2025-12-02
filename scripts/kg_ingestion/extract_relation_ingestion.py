"""
Extract Relation Ingestion - Debug Version with JSON Output
(Modified: Step 2 uses Semantic Similarity instead of LLM)

This module provides a streaming queue-based pipeline for extracting relations
from product descriptions and outputting them to JSONL files for analysis.

Pipeline: Agent1 (parallel) → Queue1 → VDB Similarity Check (workers) → Queue2 →
          Verification (parallel workers) → Queue3 → File Output (sequential)

Output files:
- Triplets file (JSONL): Final validated triplets
- Debug file: Intermediate processing steps for analysis
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

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Updated path as requested
CUSTOM_RELATIONS_VDB_PATH = r"\\wsl.localhost\Ubuntu\home\sabhi\BeeKurse\scripts\kg_ingestion\custom_relations_vdb"
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

# --- Helper: Extract JSON from LLM output (handles thinking text) ---
def extract_json_from_output(text: str) -> str:
    import re
    import json as json_lib

    # Strategy 1: Remove XML tags first
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    cleaned = cleaned.strip()

    # Strategy 2: Find all JSON-like blocks
    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    matches = list(re.finditer(json_pattern, cleaned, re.DOTALL))

    if matches:
        for match in reversed(matches):
            try:
                candidate = match.group(0)
                json_lib.loads(candidate)  # Validate
                return candidate
            except:
                continue

    # Strategy 3: Try the cleaned text as-is
    try:
        json_lib.loads(cleaned)
        return cleaned
    except:
        pass

    return text.strip()

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

def normalize_node_type(node_type: str, node_value: str) -> str:
    return "property"

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
            "type": triplet.object_type,
            "metadata": {
                "product_id": product_id,
                "document_id": document_id,
                "chunk_id": f"sentence_{chunk_id}"
            },
            "properties": {
                "product_name": product_name
            }
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
        try:
            # Wait for item from Queue1 with timeout
            item = await asyncio.wait_for(queue1.get(), timeout=0.1)

            import time
            step_start = time.time()

            triplet = item["triplet"]
            chunk_id = item["chunk_id"]
            chunk_text = item["chunk_text"]

            raw_relation = triplet.relation

            # --- Semantic Search Step ---
            # Perform similarity search with relevance score
            # Note: LangChain Chroma usually returns cosine similarity (0-1) or distance based on config.
            # 'similarity_search_with_relevance_score' attempts to return a 0-1 score where 1 is best.
            
            results = custom_relations_vdb.similarity_search_with_relevance_score(raw_relation, k=1)
            
            final_relation = raw_relation
            is_raw = True
            decision = "USE_RAW"
            similarity_score = 0.0
            matched_vdb_relation = None

            if results:
                doc, score = results[0]
                similarity_score = score
                matched_vdb_relation = doc.metadata.get("relation", doc.page_content)
                
                if score > SIMILARITY_THRESHOLD:
                    final_relation = matched_vdb_relation
                    is_raw = False
                    decision = "USE_STANDARD"
            
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

            # Step 4: Normalize node types
            normalized_subject_type = normalize_node_type(triplet.subject_type, triplet.subject)
            normalized_object_type = normalize_node_type(triplet.object_type, triplet.object)

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
    if not description or not description.strip():
        return 0

    print(f"\n[INGESTION] Processing product {product_id}")
    
    debug_file.write(f"\n{'='*80}\n")
    debug_file.write(f"[START] Product: {product_id} | Document: {document_id}\n")
    debug_file.write(f"[START] Timestamp: {datetime.utcnow().isoformat()}Z\n")
    debug_file.flush()

    sentences = adaptive_chunk_description(description)
    print(f"[INGESTION] Chunked into {len(sentences)} chunks")

    queue1 = asyncio.Queue()  # Agent1 → Semantic Standardizer
    queue2 = asyncio.Queue()  # Semantic Standardizer → Verification
    queue3 = asyncio.Queue()  # Verification → File output
    stop_event = asyncio.Event()

    entity_type_tracker: Dict[str, str] = {}

    # Start Semantic Standardizer worker (1 sequential to avoid VDB race conditions)
    rag_worker = asyncio.create_task(
        semantic_standardize_worker(0, queue1, queue2, stop_event, debug_file)
    )

    # Verification workers
    verify_workers = [
        asyncio.create_task(verification_worker(i, queue2, queue3, stop_event, debug_file))
        for i in range(NUM_VERIFICATION_WORKERS)
    ]

    # File output worker
    output_worker = asyncio.create_task(
        file_output_worker(queue3, stop_event, product_id, product_name, document_id, queue, entity_type_tracker, category)
    )

    # Start Agent1 extraction tasks
    extraction_tasks = [
        asyncio.create_task(agent1_extract_worker(chunk, idx, product_id, category, queue1, debug_file))
        for idx, chunk in enumerate(sentences)
    ]

    await asyncio.gather(*extraction_tasks, return_exceptions=True)
    print("[INGESTION] Agent1 extraction complete")

    await queue1.join()
    print("[INGESTION] Semantic Standardizer complete")

    await queue2.join()
    print("[INGESTION] Verification complete")

    await queue3.join()
    print("[INGESTION] File output queue processed")

    stop_event.set()
    await asyncio.gather(rag_worker, *verify_workers, return_exceptions=True)

    triplet_count = await output_worker

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