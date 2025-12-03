"""
Progress Tracker for KG Ingestion Pipeline

Tracks which files have been processed so the pipeline can resume from where it left off
if interrupted.
"""

import json
import os
from datetime import datetime
from typing import Set, Dict, Any

PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "processed_files.json")


def load_progress() -> Dict[str, Any]:
    """Load progress from JSON file. Returns empty structure if file doesn't exist."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[PROGRESS] Warning: Could not load progress file: {e}")

    return {
        "last_updated": None,
        "phase": "not_started",
        "attribute_files_completed": [],
        "relation_files_completed": []
    }


def save_progress(progress: Dict[str, Any]):
    """Save progress to JSON file."""
    progress["last_updated"] = datetime.now().isoformat()
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"[PROGRESS] Warning: Could not save progress file: {e}")


def get_completed_attribute_files() -> Set[str]:
    """Get set of filenames that have completed attribute processing."""
    progress = load_progress()
    return set(progress.get("attribute_files_completed", []))


def get_completed_relation_files() -> Set[str]:
    """Get set of filenames that have completed relation extraction."""
    progress = load_progress()
    return set(progress.get("relation_files_completed", []))


def mark_attribute_file_complete(filename: str):
    """Mark a file as having completed attribute processing."""
    progress = load_progress()
    if filename not in progress.get("attribute_files_completed", []):
        progress.setdefault("attribute_files_completed", []).append(filename)
        progress["phase"] = "processing_attributes"
        save_progress(progress)
        print(f"[PROGRESS] Attribute processing complete: {filename}")


def mark_relation_file_complete(filename: str):
    """Mark a file as having completed relation extraction."""
    progress = load_progress()
    if filename not in progress.get("relation_files_completed", []):
        progress.setdefault("relation_files_completed", []).append(filename)
        progress["phase"] = "extracting_relations"
        save_progress(progress)
        print(f"[PROGRESS] Relation extraction complete: {filename}")


def mark_phase_complete(phase: str):
    """Mark a phase as complete."""
    progress = load_progress()
    progress["phase"] = phase
    save_progress(progress)
    print(f"[PROGRESS] Phase complete: {phase}")


def reset_progress():
    """Reset all progress (for fresh start)."""
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("[PROGRESS] Progress file removed for fresh start")


def print_progress_summary():
    """Print a summary of current progress."""
    progress = load_progress()
    attr_count = len(progress.get("attribute_files_completed", []))
    rel_count = len(progress.get("relation_files_completed", []))
    phase = progress.get("phase", "not_started")
    last_updated = progress.get("last_updated", "never")

    print(f"\n[PROGRESS SUMMARY]")
    print(f"  Phase: {phase}")
    print(f"  Attribute files completed: {attr_count}")
    print(f"  Relation files completed: {rel_count}")
    print(f"  Last updated: {last_updated}\n")
