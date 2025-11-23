"""
Extract Relation Ingestion - Debug Version with JSON Output

This module provides a streaming queue-based pipeline for extracting relations
from product descriptions and outputting them to JSONL files for analysis.

Pipeline: Agent1 (parallel) → Queue1 → RAG/Agent2/VDB (workers) → Queue2 →
          Verification (parallel workers) → Queue3 → File Output (sequential)

Output files:
- Triplets file (JSONL): Final validated triplets with structured JSON objects containing:
  {source: {name, type, metadata, properties}, relation: {type, metadata}, target: {name, type, metadata, properties}}
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
CUSTOM_RELATIONS_VDB_PATH = "../custom_relations_vdb"
COLLECTION_NAME = "ecommerce_relations"
NUM_VERIFICATION_WORKERS = 3  # Number of parallel verification workers

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

standardizer_clients = [
    ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        api_key=key,
        temperature=0.0,
        top_p=0.95,
        max_completeion_tokens=65536,
        streaming=True
    ) for key in API_KEYS
]

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
llm_standardizer_cycler = itertools.cycle(standardizer_clients)
llm_verifier_cycler = itertools.cycle(verifier_clients)

# --- Helper functions to get the next client from the pool ---
def get_next_extractor(_): # <-- FIX: Added '_' to accept argument
    client = next(llm_extractor_cycler)
    # print(f"[LLM] Using API Key ending in: ...{client.api_key[-4:]} for Extractor")
    return client

def get_next_standardizer(_): # <-- FIX: Added '_' to accept argument
    client = next(llm_standardizer_cycler)
    # print(f"[LLM] Using API Key ending in: ...{client.api_key[-4:]} for Standardizer")
    return client

def get_next_verifier(_): # <-- FIX: Added '_' to accept argument
    client = next(llm_verifier_cycler)
    # print(f"[LLM] Using API Key ending in: ...{client.api_key[-4:]} for Verifier")
    return client


# --- Load Custom Relations VDB ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
custom_relations_vdb = Chroma(
    persist_directory=CUSTOM_RELATIONS_VDB_PATH,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)
print("[INGESTION] Custom Relations VDB loaded")

# --- Data Schemas ---
class RawTriplet(BaseModel):
    subject: str
    relation: str
    object: str
    subject_type: str = Field(..., description="Inferred type of subject")
    object_type: str = Field(..., description="Inferred type of object")

class ExtractionOutput(BaseModel):
    triplets: List[RawTriplet] = Field(default_factory=list)

class RelationStandardizerProposal(BaseModel):
    decision: str = Field(..., description="Either 'USE_STANDARD' or 'USE_RAW'")
    standard_relation: str = Field(None, description="Standard relation name (only if decision is USE_STANDARD)")
    reasoning: str = Field(..., description="Brief explanation for the decision")

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

STANDARDIZE_PROMPT = """You are a relation standardization expert for an e-commerce knowledge graph.

CRITICAL: Output ONLY valid JSON. Do NOT include any thinking, reasoning, explanations, or XML tags. Start your response directly with the JSON object.

### RELATIONSHIP TO STANDARDIZE:
Subject: {subject} (Type: {subject_type})
Relation: {raw_relation}
Object: {object} (Type: {object_type})

### AVAILABLE STANDARD RELATIONS (from our growing VDB):
{relations_context}

### YOUR TASK:
Determine if any VDB relation means THE SAME THING as "{raw_relation}" for this specific {subject}→{object} relationship.

**IMPORTANT:** You are ONLY standardizing the RELATION TYPE. The subject and object names remain unchanged.

**Decision Rules:**
1. **USE_STANDARD** if a VDB relation is semantically equivalent AND fits this relationship just as well or better
2. **USE_RAW** if:
   - No good semantic match exists
   - The raw relation is more specific/precise for this domain
   - The raw relation better captures the relationship meaning

**Examples:**
- VDB has "hasComponent", raw is "HAS_PROCESSOR" for (Product→Processor): USE_RAW (more specific)
- VDB has "HAS_DISPLAY", raw is "HAS_SCREEN" for (Product→Display): USE_STANDARD (same meaning)
- VDB has "relatedTo", raw is "HAS_REFRESH_RATE" for (Product→120Hz): USE_RAW (more precise)

### OUTPUT FORMAT:
If using standard relation:
{{
  "decision": "USE_STANDARD",
  "standard_relation": "<relation_name>",
  "reasoning": "Brief explanation"
}}

If keeping raw relation:
{{
  "decision": "USE_RAW",
  "reasoning": "Brief explanation"
}}

{format_instructions}
"""

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
    """
    Extract valid JSON from LLM output, even if wrapped in thinking text.

    Tries multiple strategies:
    1. Remove XML thinking tags first
    2. Look for JSON block patterns: {...}
    3. Extract and validate JSON blocks (tries last match first)
    4. Fall back to whole text as-is

    This handles:
    - Pure JSON output
    - XML tags like <think>...</think>
    - Free-form thinking text followed by JSON
    - Thinking text with JSON embedded
    """
    import re
    import json as json_lib

    # Strategy 1: Remove XML tags first
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    cleaned = cleaned.strip()

    # Strategy 2: Find all JSON-like blocks
    # Match outermost braces with nested content
    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    matches = list(re.finditer(json_pattern, cleaned, re.DOTALL))

    if matches:
        # Try matches from last to first (most recent is likely the final answer)
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

    # Strategy 4: Last resort - return original text (will fail in parser)
    return text.strip()

# --- Chains (Modified for API Key Rotation) ---
extract_parser = JsonOutputParser(pydantic_object=ExtractionOutput)
extract_prompt = ChatPromptTemplate.from_messages([
    ("system", EXTRACT_PROMPT)
])
extract_chain = extract_prompt.partial(
    format_instructions=extract_parser.get_format_instructions()
) | RunnableLambda(get_next_extractor) | (lambda x: extract_json_from_output(x.content)) | extract_parser

standardize_parser = JsonOutputParser(pydantic_object=RelationStandardizerProposal)
standardize_prompt = ChatPromptTemplate.from_messages([
    ("system", STANDARDIZE_PROMPT)
])
standardize_chain = standardize_prompt.partial(
    format_instructions=standardize_parser.get_format_instructions()
) | RunnableLambda(get_next_standardizer) | (lambda x: extract_json_from_output(x.content)) | standardize_parser

verify_parser = JsonOutputParser(pydantic_object=VerificationResult)
verify_prompt = ChatPromptTemplate.from_messages([
    ("system", VERIFY_PROMPT)
])
verify_chain = verify_prompt.partial(
    format_instructions=verify_parser.get_format_instructions()
) | RunnableLambda(get_next_verifier) | (lambda x: extract_json_from_output(x.content)) | verify_parser

# --- Helper Functions ---
def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences while preserving decimals (e.g., 6.8).
    Splits only when punctuation is followed by whitespace and a capital letter.
    """
    # Split on sentence boundaries: .!? followed by space and capital letter
    # This avoids splitting on decimals like "6.8 inch"
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def adaptive_chunk_description(description: str) -> List[str]:
    """
    Adaptive chunking strategy based on description length.

    Strategy:
    - Short (< 200 chars): Process whole description (no chunking)
    - Medium (200-400 chars): Sentence-based chunking (current approach)
    - Long (> 400 chars): Sentence-based with grouping for better context

    This optimizes for:
    - Template-based fashion products (45% of data, < 200 chars)
    - Standard descriptions (35% of data, 200-400 chars)
    - Detailed descriptions (20% of data, > 400 chars)
    """
    length = len(description)

    # Short descriptions: process whole (no context loss, faster)
    if length < 200:
        return [description]

    # Medium descriptions: sentence-based (current approach)
    elif length < 400:
        return split_into_sentences(description)

    # Long descriptions: sentence-based with grouping for context
    else:
        sentences = split_into_sentences(description)
        # Group every 2 sentences for better context in complex descriptions
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

def get_custom_relations_context(relation: str, top_k: int = 10) -> str:
    """Retrieve relevant relation examples from custom relations VDB."""
    try:
        results = custom_relations_vdb.similarity_search(relation, k=top_k)
        context_lines = []
        for doc in results:
            relation_name = doc.metadata.get('relation', 'UNKNOWN')
            description = doc.metadata.get('description', '')
            examples = doc.metadata.get('examples', '[]')
            context_lines.append(
                f"RELATION: {relation_name}\n"
                f"  Description: {description}\n"
                f"  Examples: {examples}"
            )
        return "\n\n".join(context_lines) if context_lines else "No matching relations found."
    except Exception as e:
        return f"Error retrieving context: {e}"

def add_relation_to_custom_vdb(relation_name: str, subject: str, object: str):
    """Add a new raw relation to custom VDB (learning mode)."""
    # Check if dry-run mode is enabled
    dry_run = os.getenv("DEBUG_DRY_RUN") == "true"

    try:
        description = f"Relation learned from product data: {subject} {relation_name} {object}"
        example = f"{subject} → {object}"

        if dry_run:
            # DRY-RUN: Just print what would be added
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
    """
    Normalize node types - use generic 'property' for all extracted description targets.
    This is simpler, faster, and more consistent than intelligent classification.
    """
    # Always return "property" for extracted description values
    # This matches the user's requirement for generic :property type
    return "property"

def format_triplet_output(triplet: StandardizedTriplet, product_id: str, product_name: str, document_id: str, chunk_id: int, category: str = "Product") -> dict:
    """
    Format triplet as a JSON object with source node, relation, and target node.
    Uses category as node label and includes prod_name/product_id properties to match process_jsons format.
    """
    # Sanitize category for valid Cypher identifier
    sanitized_category = category.replace(' ', '_').replace('-', '_')

    return {
        "source": {
            "name": triplet.subject,
            "type": sanitized_category,  # Use category as node type (e.g., Clothing, Electronics)
            "metadata": {
                "product_id": product_id,
                "document_id": document_id,
                "chunk_id": f"sentence_{chunk_id}"
            },
            "properties": {
                "prod_name": product_name,  # Match process_jsons format
                "product_id": product_id    # Include product_id in properties
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
            # Add warning on retry
            retry_suffix = ""
            if attempt > 0:
                retry_suffix = "\n\nIMPORTANT: You previously failed this extraction by outputting thinking/reasoning text instead of JSON. This time, you MUST output ONLY valid JSON with no explanations, no thinking process, and no additional text. Start your response directly with the JSON object."

            result = await extract_chain.ainvoke({
                "text": chunk + retry_suffix,
                "product_id": product_id,
                "category": category
            })
            
            # [FIX] Add check for None response from chain
            if result is None:
                raise Exception("LLM returned None or invalid JSON")

            extraction = ExtractionOutput(**result)

            # Log to debug file: Initial assignment after Agent1
            elapsed = time.time() - start_time
            debug_file.write(f"\n{'='*80}\n")
            debug_file.write(f"[AGENT1] Chunk {chunk_id} | Time: {elapsed:.2f}s | Attempt: {attempt + 1}\n")
            debug_file.write(f"[AGENT1] Text: {chunk[:100]}...\n")
            debug_file.write(f"[AGENT1] Extracted {len(extraction.triplets)} triplets:\n")
            for idx, triplet in enumerate(extraction.triplets, 1):
                debug_file.write(f"  {idx}. ({triplet.subject}:{triplet.subject_type}) -[{triplet.relation}]-> ({triplet.object}:{triplet.object_type})\n")
            debug_file.flush()

            # Add each triplet to Queue1 with chunk context
            for triplet in extraction.triplets:
                await queue1.put({
                    "triplet": triplet,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk
                })

            print(f"  [Agent1] Chunk {chunk_id}: Extracted {len(extraction.triplets)} triplets (attempt {attempt + 1}, {elapsed:.2f}s)")
            break  # Success - exit retry loop

        except Exception as e:
            elapsed = time.time() - start_time
            if attempt < max_retries - 1:
                print(f"  [Agent1 RETRY] Chunk {chunk_id}: {str(e)[:100]}... (attempt {attempt + 1}, retrying...)")
                debug_file.write(f"\n[AGENT1 RETRY] Chunk {chunk_id} | Time: {elapsed:.2f}s | Attempt: {attempt + 1}\n")
                debug_file.write(f"[AGENT1 RETRY] Error: {e}\n")
                debug_file.flush()
            else:
                print(f"  [Agent1 ERROR] Chunk {chunk_id}: Failed after {max_retries} attempts - {str(e)[:100]}...")
                debug_file.write(f"\n[AGENT1 ERROR] Chunk {chunk_id} | Time: {elapsed:.2f}s | All attempts failed\n")
                debug_file.write(f"[AGENT1 ERROR] Final error: {e}\n")
                debug_file.flush()
                # [FIX] Handle NoneType error by returning an empty list
                if "NoneType" in str(e):
                    print("  [Agent1 FIX] NoneType error caught, returning 0 triplets.")
                    break # Break loop, 0 triplets extracted


async def rag_standardize_vdb_worker(
    worker_id: int,
    queue1: asyncio.Queue,
    queue2: asyncio.Queue,
    stop_event: asyncio.Event,
    debug_file
):
    """Worker: RAG → Agent2 → VDB (sequential per item, pulls from Queue1)."""
    print(f"  [Worker{worker_id}] RAG/Standardize/VDB worker started")

    while True:
        try:
            # Wait for item from Queue1 with timeout
            item = await asyncio.wait_for(queue1.get(), timeout=0.1)

            import time
            step_start = time.time()

            triplet = item["triplet"]
            chunk_id = item["chunk_id"]
            chunk_text = item["chunk_text"]

            # Step 1: RAG - Get relations context
            rag_start = time.time()
            relations_context = get_custom_relations_context(triplet.relation)
            rag_time = time.time() - rag_start

            # Step 2: Agent2 - Standardize (with retry)
            max_std_retries = 2
            proposal = None
            std_time = 0

            for std_attempt in range(max_std_retries):
                try:
                    std_start = time.time()

                    # Add retry warning if needed
                    retry_context = relations_context
                    if std_attempt > 0:
                        retry_context += "\n\nIMPORTANT: Previous standardization attempt failed due to invalid output format. You MUST output ONLY valid JSON with the exact schema specified. No thinking, no explanations, just JSON."

                    std_result = await standardize_chain.ainvoke({
                        "raw_relation": triplet.relation,
                        "subject": triplet.subject,
                        "subject_type": triplet.subject_type,
                        "object": triplet.object,
                        "object_type": triplet.object_type,
                        "relations_context": retry_context
                    })
                    
                    if std_result is None:
                        raise Exception("LLM returned None or invalid JSON")
                        
                    proposal = RelationStandardizerProposal(**std_result)
                    std_time = time.time() - std_start

                    # Success - log and break
                    total_time = time.time() - step_start
                    debug_file.write(f"\n[RAG+AGENT2] Worker{worker_id} | Chunk {chunk_id} | Relation: {triplet.relation} | Attempt: {std_attempt + 1}\n")
                    debug_file.write(f"  Timing: RAG={rag_time:.2f}s, Standardize={std_time:.2f}s, Total={total_time:.2f}s\n")
                    debug_file.write(f"  RAG Context (top matches):\n")
                    context_lines = relations_context.split('\n')[:6]
                    for line in context_lines:
                        debug_file.write(f"    {line}\n")
                    debug_file.write(f"  Agent2 Decision:\n")
                    debug_file.write(f"    Raw Relation: {triplet.relation}\n")
                    debug_file.write(f"    Decision: {proposal.decision}\n")
                    if proposal.decision == "USE_STANDARD":
                        debug_file.write(f"    Standard Relation: {proposal.standard_relation}\n")
                    debug_file.write(f"    Reasoning: {proposal.reasoning}\n")
                    debug_file.flush()
                    break  # Success - exit retry loop

                except Exception as e:
                    std_time = time.time() - std_start
                    if std_attempt < max_std_retries - 1:
                        print(f"  [Worker{worker_id}] Standardization retry {std_attempt + 1}: {str(e)[:100]}...")
                        debug_file.write(f"  [STANDARDIZATION RETRY] Attempt {std_attempt + 1} failed: {str(e)[:200]}...\n")
                        debug_file.flush()
                    else:
                        # Final failure
                        print(f"  [Worker{worker_id}] Standardization failed after {max_std_retries} attempts, keeping raw")
                        debug_file.write(f"  [STANDARDIZATION ERROR] All {max_std_retries} attempts failed. Keeping raw.\n")
                        debug_file.write(f"  Final error: {e}\n\n")
                        debug_file.flush()

            # Decide: use standard or keep raw based on LLM decision
            if proposal and proposal.decision == "USE_STANDARD" and proposal.standard_relation:
                final_relation = proposal.standard_relation
                is_raw = False
                debug_file.write(f"    Choice: STANDARD ({final_relation})\n\n")
            else:
                final_relation = triplet.relation
                is_raw = True
                debug_file.write(f"    Choice: RAW ({final_relation})\n\n")
            debug_file.flush()

            # Step 3: VDB - Add if raw relation
            if is_raw:
                add_relation_to_custom_vdb(final_relation, triplet.subject, triplet.object)

            # Step 4: Normalize node types (classify properties as "property")
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
                print(f"  [Worker{worker_id}] RAG/Standardize/VDB worker stopping")
                break

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
            # Wait for item from Queue2 with timeout
            item = await asyncio.wait_for(queue2.get(), timeout=0.1)

            import time
            verify_start = time.time()

            triplet = item["triplet"]
            chunk_text = item["chunk_text"]
            chunk_id = item["chunk_id"]

            # Verify the relation (with retry)
            max_verify_retries = 2
            verification = None

            for verify_attempt in range(max_verify_retries):
                try:
                    # Add retry warning if needed
                    verify_text = chunk_text
                    if verify_attempt > 0:
                        verify_text += "\n\nIMPORTANT: Previous verification attempt failed. Output ONLY valid JSON following the exact schema. No explanations or thinking text."

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

                    # Success - log and break
                    debug_file.write(f"[VERIFICATION] Worker{worker_id} | Chunk {chunk_id} | Time: {verify_time:.2f}s | Attempt: {verify_attempt + 1}\n")
                    debug_file.write(f"  Triplet: ({triplet.subject}:{triplet.subject_type}) -[{triplet.relation}]-> ({triplet.object}:{triplet.object_type})\n")
                    debug_file.write(f"  Result: {'YES' if verification.is_valid else 'NO'}\n")
                    debug_file.write(f"  Reason: {verification.reason}\n\n")
                    debug_file.flush()
                    break  # Success - exit retry loop

                except Exception as e:
                    verify_time = time.time() - verify_start
                    if verify_attempt < max_verify_retries - 1:
                        print(f"  [VerifyWorker{worker_id}] Verification retry {verify_attempt + 1}: {str(e)[:100]}...")
                        debug_file.write(f"[VERIFICATION RETRY] Worker{worker_id} | Chunk {chunk_id} | Attempt {verify_attempt + 1} failed: {str(e)[:200]}...\n")
                        debug_file.flush()
                    else:
                        # Final failure - discard triplet (safer than accepting unverified)
                        print(f"  [VerifyWorker{worker_id}] Verification failed after {max_verify_retries} attempts, discarding triplet")
                        debug_file.write(f"[VERIFICATION ERROR] Worker{worker_id} | Chunk {chunk_id} | All {max_verify_retries} attempts failed. Discarding triplet.\n")
                        debug_file.write(f"  Final error: {e}\n\n")
                        debug_file.flush()

            # Add to queue if verification passed
            if verification and verification.is_valid:
                # Add to Queue3 for file output
                await queue3.put({
                    "triplet": triplet,
                    "chunk_id": chunk_id
                })
            elif verification:
                print(f"  [VerifyWorker{worker_id}] Rejected: {triplet.relation} - {verification.reason}")

            queue2.task_done()

        except asyncio.TimeoutError:
            # No items in queue, check if we should stop
            if stop_event.is_set() and queue2.empty():
                print(f"  [VerifyWorker{worker_id}] Verification worker stopping")
                break

async def file_output_worker(
    queue3: asyncio.Queue,
    stop_event: asyncio.Event,
    product_id: str,
    product_name: str,
    document_id: str,
    queue: Optional[UnifiedIngestionQueue], # <-- MODIFIED: Accept queue
    entity_type_tracker: Dict[str, str],
    category: str = "Product"
) -> int:
    """
    File Output worker: Sequential writing from Queue3 with entity consistency enforcement.
    [MODIFIED] Now sends formatted triplets to the UnifiedIngestionQueue.
    """
    print("  [FileOutput] File output worker started (mode: sending to queue)")

    triplet_count = 0

    while True:
        try:
            # Wait for item from Queue3 with timeout
            item = await asyncio.wait_for(queue3.get(), timeout=0.1)

            triplet = item["triplet"]
            chunk_id = item["chunk_id"]

            # Enforce entity consistency: ensure same entity name has same type
            # Check subject
            if triplet.subject in entity_type_tracker:
                # Entity seen before - use consistent type
                if entity_type_tracker[triplet.subject] != triplet.subject_type:
                    print(f"  [FileOutput] Normalizing subject '{triplet.subject}': {triplet.subject_type} -> {entity_type_tracker[triplet.subject]}")
                    triplet.subject_type = entity_type_tracker[triplet.subject]
            else:
                # First time seeing this entity - record its type
                entity_type_tracker[triplet.subject] = triplet.subject_type

            # Check object
            if triplet.object in entity_type_tracker:
                # Entity seen before - use consistent type
                if entity_type_tracker[triplet.object] != triplet.object_type:
                    print(f"  [FileOutput] Normalizing object '{triplet.object}': {triplet.object_type} -> {entity_type_tracker[triplet.object]}")
                    triplet.object_type = entity_type_tracker[triplet.object]
            else:
                # First time seeing this entity - record its type
                entity_type_tracker[triplet.object] = triplet.object_type

            # [MODIFIED] Format and send to queue
            triplet_dict = format_triplet_output(triplet, product_id, product_name, document_id, chunk_id, category)
            
            if queue:
                await queue.add_triplet(triplet_dict, source_type="description")
                triplet_count += 1
                print(f"  [FileOutput] Queued triplet: ({triplet.subject})-[{triplet.relation}]->({triplet.object})")
            else:
                print(f"  [WARN] No queue provided, skipping triplet for {product_id}")


            queue3.task_done()

        except asyncio.TimeoutError:
            # No items in queue, check if we should stop
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
    queue: Optional[UnifiedIngestionQueue], # <-- MODIFIED: Accept queue
    debug_file,
    category: str = "Product"
) -> int:
    """
    Extract relations from product description using streaming pipeline.

    Args:
        description: Product description text
        product_id: Product identifier
        product_name: Product name (for KG properties)
        document_id: Document identifier
        queue: The UnifiedIngestionQueue instance
        debug_file: Open file handle for debug output
        category: Product category (used as node label)

    Returns:
        Number of triplets written
    """
    if not description or not description.strip():
        return 0

    print(f"\n[INGESTION] Processing product {product_id}")
    print(f"[INGESTION] Description length: {len(description)} chars")

    debug_file.write(f"\n{'='*80}\n")
    debug_file.write(f"[START] Product: {product_id} | Document: {document_id}\n")
    debug_file.write(f"[START] Timestamp: {datetime.utcnow().isoformat()}Z\n")
    debug_file.write(f"[START] Description:\n{description}\n")
    debug_file.write(f"{'='*80}\n")
    debug_file.flush()

    # Step 1: Adaptive chunking based on description length
    sentences = adaptive_chunk_description(description)
    print(f"[INGESTION] Chunked into {len(sentences)} chunks (adaptive strategy)")
    debug_file.write(f"\n[CHUNKING] {len(sentences)} chunks (desc length: {len(description)} chars)\n")
    debug_file.flush()

    # Create queues
    queue1 = asyncio.Queue()  # Agent1 → RAG/Agent2/VDB workers
    queue2 = asyncio.Queue()  # RAG/Agent2/VDB → Verification workers
    queue3 = asyncio.Queue()  # Verification → File output
    stop_event = asyncio.Event()

    # Entity consistency tracker: maps entity name → entity type
    # Ensures the same entity has the same type across all triplets
    entity_type_tracker: Dict[str, str] = {}

    # Start workers
    # RAG/Standardize/VDB worker (1 sequential worker to avoid VDB race conditions)
    rag_worker = asyncio.create_task(
        rag_standardize_vdb_worker(0, queue1, queue2, stop_event, debug_file)
    )

    # Verification workers (configurable)
    verify_workers = [
        asyncio.create_task(verification_worker(i, queue2, queue3, stop_event, debug_file))
        for i in range(NUM_VERIFICATION_WORKERS)
    ]

    # File output worker (single sequential worker)
    output_worker = asyncio.create_task(
        file_output_worker(queue3, stop_event, product_id, product_name, document_id, queue, entity_type_tracker, category)
    )

    # Start Agent1 extraction tasks (parallel)
    extraction_tasks = [
        asyncio.create_task(agent1_extract_worker(chunk, idx, product_id, category, queue1, debug_file))
        for idx, chunk in enumerate(sentences)
    ]

    # Wait for all extractions to complete
    await asyncio.gather(*extraction_tasks, return_exceptions=True)
    print("[INGESTION] Agent1 extraction complete")

    # Wait for Queue1 to be processed
    await queue1.join()
    print("[INGESTION] RAG/Standardize/VDB complete")

    # Wait for Queue2 to be processed
    await queue2.join()
    print("[INGESTION] Verification complete")

    # Wait for Queue3 to be processed
    await queue3.join()
    print("[INGESTION] File output queue processed")

    # Signal workers to stop
    stop_event.set()

    # Wait for workers to finish
    await asyncio.gather(rag_worker, *verify_workers, return_exceptions=True)

    # Get final count from output worker
    triplet_count = await output_worker

    print(f"[INGESTION] Complete. Wrote {triplet_count} valid triplets")
    debug_file.write(f"\n{'='*80}\n")
    debug_file.write(f"[COMPLETE] Product: {product_id}\n")
    debug_file.write(f"[COMPLETE] Total triplets: {triplet_count}\n")
    debug_file.write(f"[COMPLETE] Timestamp: {datetime.utcnow().isoformat()}Z\n")
    debug_file.write(f"{'='*80}\n\n")
    debug_file.flush()

    return triplet_count


# [DELETED] Synchronous wrapper extract_relations_sync removed


async def extract_relations_from_directory(
    directory_path: str,
    queue: Optional[UnifiedIngestionQueue] = None  # <-- MODIFIED: Accept queue
) -> int:
    """
    Extract relations from all JSON files in directory.

    Args:
        directory_path: Path to directory with JSON files
        queue: The UnifiedIngestionQueue instance

    Returns:
        Total number of triplets extracted
    """
    import glob
    import json

    # Find all JSON files
    json_files = glob.glob(os.path.join(directory_path, "*.json"))

    if not json_files:
        print("⚠ No JSON files found for relation extraction")
        return 0

    print(f"Found {len(json_files)} JSON files for relation extraction\n")
    
    # [MODIFIED] Get queue if not provided
    if queue is None:
        print("[WARN] No queue provided to extract_relations_from_directory, getting global instance.")
        queue = await get_global_queue()


    # Open output files
    os.makedirs("output", exist_ok=True)
    # [DELETED] triplets_file removed
    debug_file = open("output/extraction_debug.log", "w", encoding="utf-8")

    total_triplets = 0

    try:
        for i, filepath in enumerate(json_files, 1):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Handle JSON array format: [{...}] -> {...}
                if isinstance(data, list):
                    if len(data) == 0:
                        print(f"[{i}/{len(json_files)}] Skipping empty array")
                        continue
                    data = data[0]  # Extract first object from array

                # Extract fields (handle variations)
                description = data.get('description') or data.get('descrption', '')
                product_id = data.get('product_id', 'UNKNOWN')
                product_name = data.get('prod_name') or data.get('product_name', 'Unknown')
                category = data.get('category', 'Product')
                metadata = data.get('metadata', {})
                document_id = metadata.get('document_id', product_id)

                if not description.strip():
                    print(f"[{i}/{len(json_files)}] Skipping {product_name} (no description)")
                    continue

                print(f"\n{'='*60}")
                print(f"[{i}/{len(json_files)}] Processing: {product_name}")
                print(f"Product ID: {product_id}")
                print(f"Category: {category}")
                print(f"Description: {description[:100]}...")
                print(f"{'='*60}")

                # Extract relations for this product
                count = await extract_relations_from_description(
                    description,
                    product_id,
                    product_name,
                    document_id,
                    queue, # <-- MODIFIED: Pass queue
                    debug_file,
                    category
                )

                total_triplets += count
                print(f"  ✓ Extracted {count} triplets from this product")

            except Exception as e:
                print(f"✗ Error processing {filepath}: {e}")
                continue

    finally:
        # [DELETED] triplets_file.close() removed
        debug_file.close()

    print(f"\n{'='*60}")
    print(f"✓ EXTRACTION COMPLETE")
    print(f"  Total triplets queued: {total_triplets}")
    print(f"  Debug Output: output/extraction_debug.log")
    print(f"{'='*60}")

    return total_triplets


# --- Example Usage (MODIFIED for async and queue) ---
async def main_test():
    """Async main function for standalone testing."""
    import glob

    parser = argparse.ArgumentParser(description="Extract relations from product descriptions to JSON files")
    parser.add_argument("input", nargs="?", default=None,
                       help="Input directory containing product JSON files or single JSON file, or leave empty for test mode")
    parser.add_argument("--debug-output", default="debug.json",
                       help="Output file for debug logs in JSON format (default: debug.json)")
    parser.add_argument("--product-id", default="TEST-001",
                       help="Product ID for test mode (default: TEST-001)")
    parser.add_argument("--document-id", default="doc_test_001",
                       help="Document ID for test mode (default: doc_test_001)")

    args = parser.parse_args()

    print("="*80)
    print("RELATION EXTRACTION DEBUG PIPELINE (STANDALONE TEST)")
    print("="*80)
    print(f"Debug output: {args.debug_output}")

    # Get the global queue for this test run
    queue = await get_global_queue()

    with open(args.debug_output, 'w') as debug_f:

        debug_f.write(f"Extraction Debug Log\n")
        debug_f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
        debug_f.write(f"{'='*80}\n\n")

        total_triplets = 0

        # If input provided, process it
        if args.input:
            # Check if it's a directory
            if os.path.isdir(args.input):
                print(f"Input directory: {args.input}\n")

                # Find all JSON files in directory
                pattern = os.path.join(args.input, "*.json")
                json_files = sorted(glob.glob(pattern))

                if not json_files:
                    print(f"Error: No JSON files found in {args.input}")
                    await shutdown_global_queue()
                    exit(1)

                print(f"Found {len(json_files)} product files\n")

                for filepath in json_files:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    # Extract metadata (at same level as product data)
                    metadata = data.get('metadata', {})

                    # Extract description and identifiers (handle field name variations)
                    description = data.get('description') or data.get('descrption', '')
                    product_id = data.get('product_id', 'UNKNOWN')
                    document_id = metadata.get('document_id', product_id)  # Use product_id if no document_id
                    product_name = data.get('prod_name') or data.get('product_name', 'Unknown')
                    category = data.get('category', 'Product')

                    if not description.strip():
                        print(f"Skipping {product_name} - no description")
                        continue

                    print(f"\n{'='*80}")
                    print(f"Processing: {product_name} ({product_id})")
                    print(f"{'='*80}")

                    count = await extract_relations_from_description(
                        description,
                        product_id,
                        product_name,
                        document_id,
                        queue, # Pass queue
                        debug_f,
                        category
                    )

                    total_triplets += count
                    print(f"Extracted {count} triplets from {product_name}")

            # Single file
            elif os.path.isfile(args.input):
                print(f"Input file: {args.input}\n")

                with open(args.input, 'r') as f:
                    data = json.load(f)

                # Extract metadata (at same level as product data)
                metadata = data.get('metadata', {})

                # Extract description and identifiers (handle field name variations)
                description = data.get('description') or data.get('descrption', '')
                product_id = data.get('product_id', args.product_id)
                product_name = data.get('prod_name') or data.get('product_name', 'Unknown')
                document_id = metadata.get('document_id', args.document_id)
                category = data.get('category', 'Product')

                total_triplets = await extract_relations_from_description(
                    description,
                    product_id,
                    product_name,
                    document_id,
                    queue, # Pass queue
                    debug_f,
                    category
                )

            else:
                print(f"Error: '{args.input}' is not a valid file or directory")
                await shutdown_global_queue()
                exit(1)

        # Test mode - use hardcoded description
        else:
            print("Test mode: Using hardcoded description\n")

            test_description = """
            Premium flagship smartphone with 6.8-inch Dynamic AMOLED display.
            Features 120Hz refresh rate for smooth scrolling.
            Powered by Snapdragon 8 Gen 3 processor with 12GB RAM.
            Quad camera system includes 200MP main sensor.
            5000mAh battery supports 45W fast charging.
            """

            total_triplets = await extract_relations_from_description(
                test_description,
                args.product_id,
                "Test Product",  # product_name for test mode
                args.document_id,
                queue, # Pass queue
                debug_f
            )

    print("\n" + "="*80)
    print(f"EXTRACTION COMPLETE (STANDALONE TEST)")
    print("="*80)
    print(f"Total triplets queued: {total_triplets}")
    print(f"Debug log saved to: {args.debug_output}")
    print("Waiting for queue to finalize...")
    
    # Shut down the queue to process all items
    await shutdown_global_queue()
    
    print("Standalone test complete.")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main_test())