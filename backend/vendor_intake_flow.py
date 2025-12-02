"""
Vendor intake flow.

Handles:
- Registration gate (vendors must be registered to proceed)
- Session mode locking (add vs update) for a short window
- Missing-field prompting loop (name + price/quantity/stock)
- Similarity checks against existing inventory for additions and updates
- Intake queue logging for downstream SQL/KG/VDB ingestion

The flow is designed to be stateless for the transport layer; all per-vendor
state lives in memory inside the SessionManager.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from config.config import Config
from search_agent.database.sql_client import SQLClient

config = Config


VITAL_FIELDS = {"name", "price", "quantity", "stock"}


def _now() -> float:
    return time.time()


def _clean_text(text: str) -> str:
    return (text or "").strip()


def _extract_name(text: str, mode: Optional[str]) -> Optional[str]:
    if not text:
        return None

    cleaned = text.strip()
    if mode in {"add", "update"}:
        cleaned = re.sub(r"^(add|update)\b[:\- ]*", "", cleaned, flags=re.IGNORECASE).strip()

    # Strip trailing fields like price/quantity/stock hints to keep only the product name
    keyword_match = re.search(r"\b(price|cost|mrp|qty|quantity|stock)\b", cleaned, re.IGNORECASE)
    if keyword_match:
        cleaned = cleaned[: keyword_match.start()].strip()

    return cleaned or None


def _extract_price(text: str) -> Optional[float]:
    match = re.search(r"(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d{1,2})?)", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _extract_integer(text: str) -> Optional[int]:
    match = re.search(r"(\d+)", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


@dataclass
class SessionState:
    user_id: str
    mode: Optional[str] = None  # "add" or "update"
    session_start: float = field(default_factory=_now)
    locked_until: float = 0.0
    awaiting_missing_fields: bool = False
    missing_fields: List[str] = field(default_factory=list)
    retry_count: int = 0
    last_payload: Dict[str, Any] = field(default_factory=dict)
    pending_confirmation: Optional[Dict[str, Any]] = None

    def reset(self):
        self.mode = None
        self.session_start = _now()
        self.locked_until = 0.0
        self.awaiting_missing_fields = False
        self.missing_fields = []
        self.retry_count = 0
        self.last_payload = {}
        self.pending_confirmation = None


class VendorRegistry:
    """Simple file-backed registry for vendor phone numbers."""

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)

    def _load(self) -> Dict[str, List[str]]:
        if not self.registry_path.exists():
            return {"registered": []}
        try:
            with self.registry_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"registered": []}

    def is_registered(self, user_id: str) -> bool:
        data = self._load()
        registered = set(data.get("registered", []))
        # Normalize numbers by stripping +, -, spaces
        normalized = user_id.replace("+", "").replace("-", "").replace(" ", "")
        return normalized in {u.replace("+", "").replace("-", "").replace(" ", "") for u in registered}


class InventoryIntake:
    """Encapsulates inventory addition/update decisions and logging."""

    def __init__(self, sql_client: SQLClient):
        self.sql = sql_client

    def find_similar(self, name: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        conn = self.sql._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT product_id, prod_name FROM product_table")
        rows = cur.fetchall()
        scored = []
        for row in rows:
            prod_id, prod_name = row["product_id"], row["prod_name"]
            if not prod_name:
                continue
            score = _similarity(name, prod_name)
            scored.append((prod_id, prod_name, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def enqueue_intake(self, user_id: str, mode: str, payload: Dict[str, Any], matched_product_id: Optional[str] = None) -> None:
        """Persist intake intent for downstream ingestion."""
        conn = self.sql._get_connection()
        cur = conn.cursor()

        table = config.INTAKE_QUEUE_TABLE
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                mode TEXT,
                matched_product_id TEXT,
                payload_json TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            f"INSERT INTO {table} (user_id, mode, matched_product_id, payload_json) VALUES (?, ?, ?, ?)",
            (user_id, mode, matched_product_id, json.dumps(payload)),
        )
        conn.commit()


class VendorIntakeFlow:
    """Stateful flow manager for vendor intake sessions."""

    def __init__(self, sql_client: Optional[SQLClient] = None, registry: Optional[VendorRegistry] = None):
        self.sql_client = sql_client or SQLClient()
        self.registry = registry or VendorRegistry(config.VENDOR_REGISTRY_FILE)
        self.sessions: Dict[str, SessionState] = {}
        self.intake = InventoryIntake(self.sql_client)
        self.lock_seconds = max(
            config.VENDOR_SESSION_LOCK_MIN_SECONDS,
            min(config.VENDOR_SESSION_LOCK_SECONDS, config.VENDOR_SESSION_LOCK_MAX_SECONDS),
        )

    # --------------------- Public API ---------------------
    def handle(self, user_id: str, message: str, attachments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Main entry point. Returns a response dict ready for WhatsApp/HTTP."""

        attachments = attachments or []
        session = self.sessions.get(user_id) or SessionState(user_id=user_id)
        self.sessions[user_id] = session

        if not self.registry.is_registered(user_id):
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            "You need to register before adding or updating inventory. "
                            f"Sign up here: {config.VENDOR_REGISTRATION_URL}"
                        ),
                    }
                ]
            }

        # If lock expired, force re-selection unless user is in the middle of a prompt
        if session.locked_until and _now() > session.locked_until:
            session.mode = None

        # Handle pending confirmation for similar products
        if session.pending_confirmation:
            return self._handle_confirmation(session, message)

        # Detect explicit mode switches
        detected_mode = self._detect_mode(message)
        if detected_mode:
            session.mode = detected_mode
            session.session_start = _now()
            session.locked_until = session.session_start + self.lock_seconds

        if session.mode not in {"add", "update"}:
            return self._prompt_mode(session)

        # Run payload extraction and validation
        payload, missing_fields = self._extract_payload(message, attachments, session)

        if missing_fields:
            return self._handle_missing_fields(session, missing_fields, payload)

        # Similarity checks
        similar = self.intake.find_similar(payload["name"]) if payload.get("name") else []
        if session.mode == "add" and similar and similar[0][2] >= config.VENDOR_SIMILARITY_THRESHOLD:
            session.pending_confirmation = {
                "action": "similar_add",
                "matches": similar,
                "payload": payload,
            }
            return self._format_similar_prompt(similar, addition=True)

        if session.mode == "update":
            if not similar or similar[0][2] < config.VENDOR_UPDATE_MIN_SIMILARITY:
                return {
                    "messages": [
                        {
                            "type": "text",
                            "text": (
                                "This product isn't in the inventory. "
                                "Share more details or say 'add as new' to create it."
                            ),
                        }
                    ]
                }
            session.pending_confirmation = {
                "action": "update_match",
                "matches": similar,
                "payload": payload,
            }
            return self._format_similar_prompt(similar, addition=False)

        # No confirmations required; enqueue directly
        return self._complete_intake(session, payload, matched_product_id=None)

    # --------------------- Internal helpers ---------------------
    def _detect_mode(self, text: str) -> Optional[str]:
        lower = text.lower()
        if "update" in lower:
            return "update"
        if "add" in lower or "addition" in lower or "new product" in lower:
            return "add"
        return None

    def _prompt_mode(self, session: SessionState) -> Dict[str, Any]:
        session.mode = None
        session.pending_confirmation = None
        return {
            "messages": [
                {
                    "type": "text",
                    "text": (
                        "Are we adding a new product or updating an existing one? "
                        "Reply with 'add' or 'update'."
                    ),
                }
            ]
        }

    def _extract_payload(
        self, message: str, attachments: List[Dict[str, Any]], session: SessionState
    ) -> Tuple[Dict[str, Any], List[str]]:
        text = _clean_text(message)
        name_candidate = _extract_name(text.split("\n")[0] if text else "", session.mode)
        price = _extract_price(text)
        quantity = _extract_integer(text)
        stock = _extract_integer(text)

        payload = {
            "name": name_candidate or session.last_payload.get("name"),
            "price": price if price is not None else session.last_payload.get("price"),
            "quantity": quantity if quantity is not None else session.last_payload.get("quantity"),
            "stock": stock if stock is not None else session.last_payload.get("stock"),
            "attachments": [att.get("type", "image") for att in attachments],
            "raw": text,
        }

        missing = []
        if not payload.get("name"):
            missing.append("name")
        # Need at least one of price/quantity/stock
        if payload.get("price") is None:
            missing.append("price")
        if payload.get("quantity") is None and payload.get("stock") is None:
            missing.append("quantity/stock")

        return payload, missing

    def _handle_missing_fields(
        self, session: SessionState, missing_fields: List[str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        session.awaiting_missing_fields = True
        session.missing_fields = missing_fields
        session.retry_count += 1
        session.last_payload.update(payload)

        if session.retry_count >= 2:
            session.awaiting_missing_fields = False
            session.retry_count = 0
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            "I still couldn't find these fields: "
                            f"{', '.join(missing_fields)}. "
                            "Please resend with clear labels (e.g., Name, Price, Quantity)."
                        ),
                    }
                ]
            }

        prompt_fields = ", ".join(missing_fields)
        return {
            "messages": [
                {
                    "type": "text",
                    "text": (
                        f"I need a bit more info to proceed. Missing: {prompt_fields}. "
                        "Share them in one message — if you have a photo with handwritten details, send it too."
                    ),
                }
            ]
        }

    def _format_similar_prompt(self, matches: List[Tuple[str, str, float]], addition: bool) -> Dict[str, Any]:
        intro = "Found something very similar in your inventory." if addition else "Is this the item you want to update?"
        lines = [intro]
        for idx, (pid, name, score) in enumerate(matches, start=1):
            lines.append(f"{idx}. {name} (ID: {pid}, similarity {score:.2f})")
        lines.append("Reply with the number/ID to use it, or say 'add new' to continue with your version.")
        return {"messages": [{"type": "text", "text": "\n".join(lines)}]}

    def _handle_confirmation(self, session: SessionState, message: str) -> Dict[str, Any]:
        pending = session.pending_confirmation or {}
        lower = message.lower().strip()
        matches = pending.get("matches", [])

        if lower in {"add new", "add as new", "new"}:
            session.pending_confirmation = None
            return self._complete_intake(session, pending.get("payload", {}), matched_product_id=None)

        # Accept number or product_id
        chosen_id = None
        if matches:
            if lower.isdigit():
                idx = int(lower) - 1
                if 0 <= idx < len(matches):
                    chosen_id = matches[idx][0]
            else:
                for pid, _name, _score in [(m[0], m[1], m[2]) for m in matches]:
                    if pid.lower() in lower:
                        chosen_id = pid
                        break

        if chosen_id:
            session.pending_confirmation = None
            # Switch mode to update if user selected an existing item during addition
            if pending.get("action") == "similar_add":
                session.mode = "update"
            return self._complete_intake(session, pending.get("payload", {}), matched_product_id=chosen_id)

        # Fallback prompt
        return self._format_similar_prompt(matches, addition=pending.get("action") == "similar_add")

    def _complete_intake(
        self, session: SessionState, payload: Dict[str, Any], matched_product_id: Optional[str]
    ) -> Dict[str, Any]:
        start = _now()
        self.intake.enqueue_intake(
            user_id=session.user_id,
            mode=session.mode or "add",
            payload=payload,
            matched_product_id=matched_product_id,
        )
        ingestion_time = _now() - start
        session_time = _now() - session.session_start
        done_after = max(session_time, ingestion_time)

        session.awaiting_missing_fields = False
        session.retry_count = 0
        session.pending_confirmation = None

        completion = (
            f"Input processing done in {done_after:.1f}s. "
            "We're pushing it to SQL/KG/VDB in the background."
        )
        thanks = (
            "Thanks for being our partner! Choose next action: add or update. "
            "Next time, just pick one of these to continue quickly."
        )

        session.locked_until = _now() + self.lock_seconds
        return {
            "messages": [
                {"type": "text", "text": completion},
                {"type": "text", "text": thanks},
            ]
        }