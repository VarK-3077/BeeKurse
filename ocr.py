import os
import sys
import json
import uuid
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import io

import cv2
import numpy as np
from PIL import Image
import torch

try:
    import fitz as pymupdf
except Exception as e:
    raise ImportError(f"PyMuPDF (pip install PyMuPDF) is required. Import error: {e}")

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
    from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
except ImportError as e:
    print(f"[ERROR] Missing required imports: {e}")
    print("Install: pip install 'olmocr>=0.4.0' 'transformers>=4.48.3' torch torchvision PyMuPDF pillow opencv-python numpy")
    raise


# ==================== CONFIGURATION ====================

class Config:
    """Configuration for e-commerce document processing"""
    # Minimal text detection
    MIN_TEXT_LENGTH_FOR_IMAGE = 50
    MIN_WORDS_FOR_IMAGE = 10

    # Image handling
    IMAGE_OUTPUT_DIR = "./extracted_images"
    IMAGE_QUALITY = 95
    IMAGE_DPI = 150
    MIN_IMAGE_SIZE = 100

    # Models
    OCR_MODEL = "allenai/olmOCR-2-7B-1025-FP8"
    REASONING_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    # Token caps (conservative)
    OCR_MAX_NEW_TOKENS = 1024
    REASONING_MAX_NEW_TOKENS = 512
    REASONING_MAX_PROMPT_CHARS = 3000


# ==================== ID GENERATION ====================

def generate_document_id(file_path: str) -> str:
    timestamp = datetime.utcnow().isoformat()
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()[:8]
    return f"DOC_{file_hash}_{timestamp.replace(':', '').replace('-', '').replace('.', '')[:14]}"


def generate_product_id() -> str:
    return f"PID-{uuid.uuid4()}"


def generate_image_filename(document_id: str, page_no: int, image_index: int) -> str:
    return f"{document_id}_P{page_no:04d}_IMG{image_index:03d}.png"


# ==================== IMAGE PREPROCESSING ====================

def load_image(path: str, page: int = 0) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    try:
        pil = Image.open(path)
        if getattr(pil, "n_frames", 1) > 1:
            pil.seek(page)
        pil = pil.convert("RGB")
        arr = np.array(pil)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as e:
        raise FileNotFoundError(f"Could not read image '{path}': {e}")


def denoise_image(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        raise ValueError("Input image is None or empty")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10,
                                           templateWindowSize=7, searchWindowSize=21)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        print(f"[WARN] Denoising failed: {e}. Returning original.")
        return img


def deskew_with_hough(img: np.ndarray, debug: bool = False) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180.0, threshold=100,
                           minLineLength=max(int(img.shape[1]*0.15), 30), maxLineGap=20)

    angles = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1
            angle = 90.0 if dx == 0 else np.degrees(np.arctan2(dy, dx))
            if abs(angle) <= 45:
                angles.append(angle)

    if len(angles) >= 1:
        median_angle = float(np.median(angles))
    else:
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh < 255))
        if coords.shape[0] == 0:
            median_angle = 0.0
        else:
            rect = cv2.minAreaRect(coords)
            rect_angle = rect[-1]
            median_angle = -(90 + rect_angle) if rect_angle < -45 else -rect_angle

    if abs(median_angle) > 45:
        median_angle = 0.0

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if debug:
        print(f"[DEBUG] Deskew angle: {median_angle:.3f} degrees")
    return rotated


def preprocess_image_for_ocr(image_path: str, debug: bool = False) -> str:
    img = load_image(image_path)
    if debug:
        print(f"[INFO] Loaded image shape: {img.shape}")

    den = denoise_image(img)
    desk = deskew_with_hough(den, debug=debug)

    temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    preprocessed_path = os.path.join(temp_dir, f"{base}_prep_{uuid.uuid4().hex[:8]}.png")
    cv2.imwrite(preprocessed_path, desk)

    if debug:
        print(f"[INFO] Preprocessed image saved to: {preprocessed_path}")

    return preprocessed_path


# ==================== OCR MODEL MANAGEMENT ====================

class OLMOCRProcessor:
    _instance = None
    _model = None
    _processor = None
    _model_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OLMOCRProcessor, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, model_name: str = Config.OCR_MODEL, debug: bool = False):
        if cls._model is not None and cls._model_name == model_name:
            if debug:
                print(f"[INFO] OCR Model already loaded: {model_name}")
            return

        if debug:
            print(f"[INFO] Initializing olmOCR model: {model_name}")

        if "2-7B-1025" in model_name or "2-7B-0725" in model_name:
            processor_model = "Qwen/Qwen2.5-VL-7B-Instruct"
        else:
            processor_model = "Qwen/Qwen2-VL-7B-Instruct"

        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32

        try:
            if "FP8" in model_name:
                try:
                    cls._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
                    )
                    cls._model_name = model_name
                except Exception:
                    fallback = model_name.replace("-FP8", "")
                    cls._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        fallback, torch_dtype=dtype, device_map="auto", trust_remote_code=True
                    )
                    cls._model_name = fallback
            else:
                cls._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
                )
                cls._model_name = model_name

            cls._processor = AutoProcessor.from_pretrained(processor_model, trust_remote_code=True)

            if debug:
                print(f"[INFO] OCR model loaded: {cls._model_name}")

        except Exception as e:
            print(f"[ERROR] Failed to initialize OCR model: {e}")
            raise

    @classmethod
    def get_model(cls):
        if cls._model is None:
            raise RuntimeError("OCR Model not initialized. Call initialize() first.")
        return cls._model

    @classmethod
    def get_processor(cls):
        if cls._processor is None:
            raise RuntimeError("OCR Processor not initialized. Call initialize() first.")
        return cls._processor

    @classmethod
    def reset(cls):
        cls._model = None
        cls._processor = None
        cls._model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==================== REASONING MODEL MANAGEMENT ====================

class ReasoningModelProcessor:
    _instance = None
    _model = None
    _tokenizer = None
    _model_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ReasoningModelProcessor, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, model_name: str = Config.REASONING_MODEL, debug: bool = False):
        if cls._model is not None and cls._model_name == model_name:
            if debug:
                print(f"[INFO] Reasoning Model already loaded: {model_name}")
            return

        if debug:
            print(f"[INFO] Initializing reasoning model: {model_name}")

        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32

        try:
            cls._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # Ensure pad token exists
            if cls._tokenizer.pad_token is None:
                if getattr(cls._tokenizer, "eos_token", None) is not None:
                    cls._tokenizer.pad_token = cls._tokenizer.eos_token
                else:
                    cls._tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            cls._model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
            )

            # Resize embeddings if token added
            try:
                cls._model.resize_token_embeddings(len(cls._tokenizer))
            except Exception:
                pass

            cls._model_name = model_name
            if debug:
                print(f"[INFO] Reasoning model loaded: {cls._model_name}")

        except Exception as e:
            print(f"[ERROR] Failed to initialize reasoning model: {e}")
            raise

    @classmethod
    def get_model(cls):
        if cls._model is None:
            raise RuntimeError("Reasoning model not initialized. Call initialize() first.")
        return cls._model

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            raise RuntimeError("Reasoning tokenizer not initialized. Call initialize() first.")
        return cls._tokenizer

    @classmethod
    def reset(cls):
        cls._model = None
        cls._tokenizer = None
        cls._model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==================== SAFE GENERATE WRAPPER ====================

def safe_generate(model, *args, max_new_tokens: int = 256, device: Optional[torch.device] = None, **kwargs):
    """
    Wrapper around model.generate to handle OOM and retry with smaller token limits.
    Returns the generated ids tensor.
    """
    try:
        return model.generate(*args, max_new_tokens=max_new_tokens, **kwargs)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            print("[WARN] CUDA OOM during generation. Trying smaller max_new_tokens and clearing cache.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # Retry with half tokens
            smaller = max(64, int(max_new_tokens // 2))
            try:
                return model.generate(*args, max_new_tokens=smaller, **kwargs)
            except RuntimeError as e2:
                if "out of memory" in str(e2).lower():
                    # Final fallback: move model to cpu and try small generation (expensive but safe)
                    try:
                        print("[WARN] Moving model to CPU for final retry (slow).")
                        model_cpu = model.to("cpu")
                        return model_cpu.generate(*args, max_new_tokens=smaller, **kwargs)
                    except Exception:
                        pass
                raise
        else:
            raise


# ==================== OCR: extract text from image ====================

def extract_text_from_image(image_path: str, debug: bool = False) -> str:
    processor = OLMOCRProcessor.get_processor()
    model = OLMOCRProcessor.get_model()

    if debug:
        print(f"[INFO] Extracting text from: {image_path}")

    image = Image.open(image_path).convert("RGB")
    max_dim = max(image.size)
    if max_dim > 1288:
        scale = 1288 / max_dim
        image = image.resize(tuple(int(d * scale) for d in image.size), Image.LANCZOS)
        if debug:
            print(f"[INFO] Resized image to {image.size}")

    prompt = build_no_anchoring_v4_yaml_prompt()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image", "image": image}
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

    # Safe device move (only tensors)
    device = next(model.parameters()).device
    safe_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            safe_inputs[k] = v.to(device)
        else:
            safe_inputs[k] = v

    # compute prompt length if input_ids present
    prompt_length = safe_inputs.get("input_ids").shape[1] if safe_inputs.get("input_ids") is not None else 0

    # generate with safe wrapper
    outputs = safe_generate(model, **safe_inputs, max_new_tokens=Config.OCR_MAX_NEW_TOKENS, device=device)

    # decode - if we have prompt_length, remove it
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        # fallback: try to decode raw tensor to string
        try:
            decoded = outputs.cpu().numpy().tolist()
            return str(decoded)
        except Exception:
            return ""

    if prompt_length > 0:
        # sometimes outputs includes entire sequence; attempt to decode tokens after prompt_length
        try:
            new_tokens = outputs[:, prompt_length:].cpu().numpy().tolist()
            text_output = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        except Exception:
            text_output = tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)
    else:
        text_output = tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)

    markdown_content = text_output[0] if text_output else ""
    if debug:
        print(f"[INFO] OCR extracted {len(markdown_content)} chars")
    return markdown_content


def is_minimal_text(text: str) -> bool:
    if not text or len(text.strip()) < Config.MIN_TEXT_LENGTH_FOR_IMAGE:
        return True
    words = [w for w in text.split() if len(w) > 1]
    return len(words) < Config.MIN_WORDS_FOR_IMAGE


# ==================== PDF PROCESSING ====================

def extract_images_from_pdf_page(pdf_doc, page_num: int, document_id: str,
                                 output_dir: str, debug: bool = False) -> List[Dict[str, Any]]:
    page = pdf_doc[page_num]
    image_list = page.get_images(full=True)
    extracted_images = []

    for img_index, img_info in enumerate(image_list):
        try:
            xref = img_info[0]
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")
            image = Image.open(io.BytesIO(image_bytes))
            if image.width < Config.MIN_IMAGE_SIZE or image.height < Config.MIN_IMAGE_SIZE:
                if debug:
                    print(f"[INFO] Skipping small embedded image on page {page_num+1}: {image.size}")
                continue
            os.makedirs(output_dir, exist_ok=True)
            image_filename = generate_image_filename(document_id, page_num + 1, img_index)
            image_path = os.path.join(output_dir, image_filename)
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            image.save(image_path, 'PNG', quality=Config.IMAGE_QUALITY)
            extracted_images.append({
                "image_path": image_path,
                "image_filename": image_filename,
                "width": image.width,
                "height": image.height,
                "format": image_ext,
                "extraction_source": "embedded_image"
            })
            if debug:
                print(f"[INFO] Saved embedded image: {image_filename}")
        except Exception as e:
            if debug:
                print(f"[WARN] Failed to extract embedded image {img_index} on page {page_num+1}: {e}")
            continue

    return extracted_images


def extract_text_from_pdf(pdf_path: str, document_id: str, image_output_dir: str,
                          debug: bool = False) -> Dict[str, Any]:
    if debug:
        print(f"[INFO] Processing PDF: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    num_pages = doc.page_count if hasattr(doc, "page_count") else len(doc)

    all_text = []
    all_images = []

    for page_num in range(num_pages):
        if debug:
            print(f"[INFO] Processing page {page_num + 1}/{num_pages}")

        extracted_images = extract_images_from_pdf_page(doc, page_num, document_id, image_output_dir, debug=debug)

        # render page for OCR
        page = doc[page_num]
        zoom = Config.IMAGE_DPI / 72.0
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        image_data = pix.tobytes("ppm")
        image = Image.open(io.BytesIO(image_data))

        max_dim = max(image.size)
        if max_dim > 1288:
            scale = 1288 / max_dim
            image = image.resize(tuple(int(d * scale) for d in image.size), Image.LANCZOS)

        # OCR page image
        try:
            processor = OLMOCRProcessor.get_processor()
            model = OLMOCRProcessor.get_model()
        except RuntimeError:
            # If model not initialized, raise a clear error
            raise RuntimeError("OCR model not initialized. Call OLMOCRProcessor.initialize() before extracting PDF text.")

        prompt = build_no_anchoring_v4_yaml_prompt()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image}
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

        # Safe device move (only tensors)
        device = next(model.parameters()).device
        safe_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                safe_inputs[k] = v.to(device)
            else:
                safe_inputs[k] = v

        prompt_length = safe_inputs.get("input_ids").shape[1] if safe_inputs.get("input_ids") is not None else 0

        outputs = safe_generate(model, **safe_inputs, max_new_tokens=Config.OCR_MAX_NEW_TOKENS, device=device)

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            markdown_content = ""
        else:
            if prompt_length > 0:
                try:
                    new_tokens = outputs[:, prompt_length:].cpu().numpy().tolist()
                    text_output = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                except Exception:
                    text_output = tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)
            else:
                text_output = tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)
            markdown_content = text_output[0] if text_output else ""

        # If minimal text, save rendered page as fallback image
        if is_minimal_text(markdown_content):
            try:
                fallback_idx = len(extracted_images)
                image_filename = generate_image_filename(document_id, page_num + 1, fallback_idx)
                image_path = os.path.join(image_output_dir, image_filename)
                save_img = image.convert('RGB') if image.mode in ('RGBA', 'LA', 'P') else image
                os.makedirs(image_output_dir, exist_ok=True)
                save_img.save(image_path, 'PNG', quality=Config.IMAGE_QUALITY)
                extracted_images.append({
                    "image_path": image_path,
                    "image_filename": image_filename,
                    "width": save_img.size[0],
                    "height": save_img.size[1],
                    "format": "png",
                    "extraction_source": "rendered_page_fallback"
                })
                if debug:
                    print(f"[INFO] Saved fallback image for page {page_num + 1}: {image_filename}")
            except Exception as ee:
                if debug:
                    print(f"[WARN] Could not save rendered fallback image for page {page_num + 1}: {ee}")

        all_text.append(markdown_content)
        all_images.extend(extracted_images)

    doc.close()
    full_text = "\n\n".join([t for t in all_text if t.strip()])

    return {
        "full_text": full_text,
        "extracted_images": all_images,
        "num_pages": num_pages
    }


# ==================== STRUCTURED EXTRACTION ====================

def _messages_to_text_fallback(messages: List[Dict[str, Any]]) -> str:
    """Fallback to convert messages list to a single prompt string."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content_str = []
            for c in content:
                if isinstance(c, dict):
                    # pick 'text' if present
                    txt = c.get("text") or c.get("caption") or json.dumps(c)
                    content_str.append(str(txt))
                else:
                    content_str.append(str(c))
            content_flat = "\n".join(content_str)
        else:
            content_flat = str(content)
        parts.append(f"{role.upper()}:\n{content_flat}")
    return "\n\n".join(parts)

def create_extraction_prompt(extracted_text: str, image_filenames: List[str]) -> str:
    images_list = ", ".join(image_filenames) if image_filenames else "No images extracted"

    # Updated example to reflect empty strings for missing numbers and color extraction
    example1_text = "Black sequined blouse, Price: ₹899, Size: S, M. Description: Available in red and black."
    example1_json = {
        "prod_name": "Sequined Blouse",
        "price": "₹899",
        "quantity": "1",
        "quantityunit": "piece",
        "size": "S, M",
        "store": "",
        "dimensions": "",
        "brand": "",
        "colour": "red, black", 
        "description": "Available in red and black.",
        "category": "Clothing",
        "subcategory": "Blouse",
        "imageid": "",
        "product_id": "some-generated-id",
        "rating": "",
        "stock": ""
    }

    prompt = f"""
You are an expert e-commerce information extractor. Read the EXTRACTED TEXT and AVAILABLE IMAGES list and output a single JSON object (no commentary) with EXACTLY the following keys:
prod_name, price, quantity, quantityunit, size, store, dimensions, brand, colour, description, category, subcategory, imageid, product_id, rating, stock

Rules:
- Output ONLY one valid JSON object. No markdown, no explanation.
- **dimensions**: format as "LxWxH unit" (e.g., "10x5x2 cm").
- **colour**: specific item colour. If not in the table, extract available colours from the description (e.g., "available in red, white").
- **Store**: The text at the very top/header is USUALLY the 'store'. If not, try to infer from context, else "".
- **Brand**: The first word of the product name is often the 'brand' .If not explicitly written, infer from product name or description (e.g., 'Nike Running Shoes' -> brand: 'Nike', 'Levi's Jeans' -> 'Levi's').
- **rating**: float between 0 and 5, try to extract from table and description, else "" (e.g. rated 4.3 -> 4.3, rated four stars -> 4.0, rating of 3.6 -> 3.6).
- **Category Inference**: If not explicitly written, INFER category and subcategory from the product name (e.g., 'Shirt' -> Category: 'Clothing', Subcategory: 'Shirt').
- **stock**: integer number of items in stock, if not in table, try to extract from other information available, else "".
- imageid: choose a primary image filename from AVAILABLE IMAGES when appropriate, otherwise "".
- product_id: generate a new and unique product_id using the format "PID-" followed by a UUID.

FEW-SHOT EXAMPLE (DO NOT OUTPUT THIS):
TEXT: {example1_text}
OUTPUT JSON: {json.dumps(example1_json, ensure_ascii=False)}

AVAILABLE IMAGES: {images_list}

EXTRACTED TEXT:
{extracted_text}

Now produce ONLY the JSON object with the exact keys above.
""".strip()

    # truncate prompt if too long
    if len(prompt) > Config.REASONING_MAX_PROMPT_CHARS:
        prompt = prompt[:Config.REASONING_MAX_PROMPT_CHARS] + "\n\n[... text truncated ...]"

    return prompt

def extract_structured_data(extracted_text: str, image_filenames: List[str],
                           debug: bool = False) -> Dict[str, Any]:
    model = ReasoningModelProcessor.get_model()
    tokenizer = ReasoningModelProcessor.get_tokenizer()
    device = next(model.parameters()).device

    if debug:
        print("[INFO] Extracting structured data with reasoning model...")

    prompt = create_extraction_prompt(extracted_text or "", image_filenames)

    # Build messages (some tokenizers/processors support apply_chat_template)
    messages = [
        {"role": "system", "content": "You are a precise JSON extraction system. Output only valid JSON."},
        {"role": "user", "content": prompt}
    ]

    # Try to use tokenizer.apply_chat_template if available; otherwise fallback
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = _messages_to_text_fallback(messages)
    else:
        text = _messages_to_text_fallback(messages)

    # Tokenize with truncation
    max_model_len = getattr(tokenizer, "model_max_length", 2048)
    # Leave margin for generation
    max_input_len = max(1, min(max_model_len - 32, int(max_model_len * 0.9)))
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Generate with safe wrapper
    try:
        outputs = safe_generate(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=Config.REASONING_MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    except Exception as e:
        print(f"[ERROR] Reasoning model generation failed: {e}")
        # Fallback empty template
        return {
            "prod_name": "",
            "price": "",
            "quantity": "1",
            "quantityunit": "piece",
            "size": "",
            "store": "",
            "dimensions": "",
            "brand": "",
            "colour": "",
            "description": extracted_text[:500] if extracted_text else "",
            "category": "",
            "subcategory": "",
            "imageid": image_filenames[0] if image_filenames else "",
            "product_id": generate_product_id(),
            "rating": "",
            "stock": ""
        }

    # Compute prompt_length to remove from outputs if present
    prompt_length = input_ids.shape[1] if input_ids is not None else 0

    # If outputs include the prompt, remove prompt tokens; else decode whole output
    try:
        if prompt_length > 0:
            gen_ids = outputs[0][prompt_length:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        else:
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception:
        # fallback decode entire output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if debug:
        print(f"[DEBUG] Raw model response (first 1000 chars): {response[:1000]}")

    # Robust JSON extraction
    try:
        cleaned = response.strip()
        # strip fences
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'```$', '', cleaned, flags=re.IGNORECASE).strip()
        # find first {...} block
        m = re.search(r'({[\s\S]*})', cleaned)
        if m:
            json_str = m.group(1)
        else:
            json_str = cleaned

        structured_data = json.loads(json_str)

        # Post-process defaults and normalize keys
        if not structured_data.get("product_id"):
            structured_data["product_id"] = generate_product_id()

        # normalize numeric fields
        for k in ("rating", "stock"):
            if k in structured_data:
                try:
                    structured_data[k] = int(structured_data[k])
                except Exception:
                    structured_data[k] = ""
            else:
                structured_data[k] = ""

        # ensure quantity is string
        structured_data.setdefault("quantity", "1")
        structured_data["quantity"] = str(structured_data.get("quantity", "1"))

        # ensure all required fields exist
        required_fields = ["prod_name", "price", "quantity", "quantityunit", "size", "store", "dimensions",
                           "brand", "colour", "description", "category", "subcategory", "imageid", "product_id",
                           "rating", "stock"]
        for rf in required_fields:
            if rf not in structured_data:
                structured_data[rf] = "" if rf not in ("rating", "stock") else 0

        return structured_data

    except Exception as e:
        if debug:
            print(f"[ERROR] Failed to parse JSON from reasoning model: {e}")
            print(f"[ERROR] Raw response snippet: {response[:1000]}")
        # fallback template
        return {
            "prod_name": "",
            "price": "",
            "quantity": "1",
            "quantityunit": "piece",
            "size": "",
            "store": "",
            "dimensions": "",
            "brand": "",
            "colour": "",
            "description": extracted_text[:500] if extracted_text else "",
            "category": "",
            "subcategory": "",
            "imageid": image_filenames[0] if image_filenames else "",
            "product_id": generate_product_id(),
            "rating": "",
            "stock": ""
        }


# ==================== MAIN PROCESSING ====================

def process_document(file_path: str, output_dir: str = "./ecommerce_output",
                     image_output_dir: Optional[str] = None,
                     debug: bool = True) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    doc_id = generate_document_id(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()

    if image_output_dir is None:
        image_output_dir = os.path.join(Config.IMAGE_OUTPUT_DIR, doc_id)
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if debug:
        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")
        print(f"Document ID: {doc_id}")
        print(f"{'='*60}\n")

    # Initialize models
    if debug:
        print("[INFO] Initializing OCR model...")
    OLMOCRProcessor.initialize(model_name=Config.OCR_MODEL, debug=debug)

    if debug:
        print("[INFO] Initializing reasoning model...")
    ReasoningModelProcessor.initialize(model_name=Config.REASONING_MODEL, debug=debug)

    try:
        # Extract text and images
        if file_ext == '.pdf':
            result = extract_text_from_pdf(file_path, doc_id, image_output_dir, debug=debug)
            extracted_text = result["full_text"]
            extracted_images = result["extracted_images"]
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']:
            preprocessed_path = preprocess_image_for_ocr(file_path, debug=debug)
            extracted_text = extract_text_from_image(preprocessed_path, debug=debug)
            extracted_images = []
            try:
                os.remove(preprocessed_path)
            except Exception:
                pass
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    except Exception as e:
        raise RuntimeError(f"Failed during extraction: {e}")

    if debug:
        print(f"\n[INFO] Extracted {len(extracted_text or '')} characters of text")
        print(f"[INFO] Found {len(extracted_images)} images")

    # Get structured data from reasoning model
    image_filenames = [img["image_filename"] for img in extracted_images]
    structured_data = extract_structured_data(extracted_text or "", image_filenames, debug=debug)

    output_path = os.path.join(output_dir, f"{doc_id}_product.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    if debug:
        print(f"\n{'='*60}")
        print("Product Data JSON:")
        print(json.dumps(structured_data, indent=2, ensure_ascii=False))
        print(f"\n✓ Output saved to: {output_path}")
        print(f"✓ Images saved to: {image_output_dir}")
        print(f"{'='*60}\n")

    return output_path


def process_batch(file_paths: List[str], output_dir: str = "./ecommerce_output",
                 image_output_dir: Optional[str] = None, debug: bool = True) -> List[str]:
    if debug:
        print(f"\n[INFO] Batch processing {len(file_paths)} files...")

    # Initialize models once
    OLMOCRProcessor.initialize(model_name=Config.OCR_MODEL, debug=debug)
    ReasoningModelProcessor.initialize(model_name=Config.REASONING_MODEL, debug=debug)

    results = []
    for i, file_path in enumerate(file_paths, 1):
        try:
            if debug:
                print(f"\n[{i}/{len(file_paths)}] Processing: {file_path}")

            output = process_document(file_path, output_dir, image_output_dir, debug=debug)
            results.append(output)
            print(f"✓ Successfully processed: {file_path}")

        except Exception as e:
            print(f"✗ Failed to process {file_path}: {e}")
            import traceback
            traceback.print_exc()

    if debug:
        print(f"\n[INFO] Batch complete. Processed {len(results)}/{len(file_paths)} files")

    return results


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Configure CUDA if available
    if torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    if len(sys.argv) < 2:
        print("Usage: python ecom_extractor_qwen.py <document.pdf|image.png>")
        sys.exit(1)

    input_path = sys.argv[1]
    out = process_document(
        input_path,
        output_dir="./ecommerce_output",
        image_output_dir=None,
        debug=True
    )
    print(f"\n✓ Output: {out}")
