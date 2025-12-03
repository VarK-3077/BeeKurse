#!/usr/bin/env python3
"""
Auto-test script for OCR inventory pipeline.
Tests the full flow: image -> OCR -> parse -> inventory update
"""
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Test image path
TEST_IMAGE = PROJECT_ROOT / "ocr_test.jpeg"

def test_step(name, func):
    """Run a test step and report results."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print("="*60)
    try:
        result = func()
        print(f"✅ PASSED: {name}")
        return True, result
    except Exception as e:
        print(f"❌ FAILED: {name}")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_ocr_import():
    """Test 1: Import OCR module."""
    from ocr import extract_text_from_image, OLMOCRProcessor
    print("  OCR module imported successfully")
    return True


def test_terminal_bot_import():
    """Test 2: Import terminal_bot functions."""
    from backend.terminal_bot import (
        parse_inventory_list_text,
        parse_inventory_item,
        process_vendor_inventory_image,
        _ocr_available
    )
    print(f"  Terminal bot imported successfully")
    print(f"  OCR available: {_ocr_available}")
    return _ocr_available


def test_ocr_extraction():
    """Test 3: Extract text from test image."""
    from ocr import extract_text_from_image, OLMOCRProcessor

    if not TEST_IMAGE.exists():
        raise FileNotFoundError(f"Test image not found: {TEST_IMAGE}")

    # Initialize OCR processor first
    print("  Initializing OCR processor (this may take a while on first run)...")
    OLMOCRProcessor.initialize(debug=True)

    print(f"  Processing image: {TEST_IMAGE}")
    ocr_text = extract_text_from_image(str(TEST_IMAGE), debug=True)

    if not ocr_text:
        raise ValueError("OCR returned empty text")

    print(f"  Extracted {len(ocr_text)} characters")
    print(f"\n--- OCR OUTPUT ---")
    print(ocr_text)
    print("--- END OCR OUTPUT ---\n")

    return ocr_text


def test_parse_inventory(ocr_text):
    """Test 4: Parse inventory list from OCR text."""
    from backend.terminal_bot import parse_inventory_list_text

    parsed = parse_inventory_list_text(ocr_text)

    update_items = parsed.get("update", [])
    add_items = parsed.get("add", [])

    print(f"  Parsed {len(update_items)} UPDATE items")
    print(f"  Parsed {len(add_items)} ADD items")

    if update_items:
        print("\n  UPDATE Items:")
        for item in update_items:
            print(f"    - {item['name']}: ₹{item['price']}/{item['unit']}, Stock: {item['stock']} {item['stock_unit']}")

    if add_items:
        print("\n  ADD Items:")
        for item in add_items:
            print(f"    - {item['name']}: ₹{item['price']}/{item['unit']}, Stock: {item['stock']} {item['stock_unit']}")

    if len(update_items) == 0 and len(add_items) == 0:
        raise ValueError("No items parsed from OCR text")

    return parsed


def test_process_vendor_image():
    """Test 5: Full vendor image processing."""
    from backend.terminal_bot import process_vendor_inventory_image

    test_vendor = "917893127444"  # Test vendor phone

    success, message, data = process_vendor_inventory_image(str(TEST_IMAGE), test_vendor)

    print(f"  Success: {success}")
    print(f"\n--- MESSAGE ---")
    print(message)
    print("--- END MESSAGE ---\n")

    if data:
        print(f"  Update count: {data.get('update_count', 0)}")
        print(f"  Add count: {data.get('add_count', 0)}")

    return success, message, data


def test_inventory_intake_import():
    """Test 6: Import inventory intake module."""
    from backend.vendor_intake_flow import InventoryIntake
    from search_agent.database.sql_client import SQLClient
    from search_agent import config

    print(f"  SQL DB Path: {config.SQL_DB_PATH}")
    print("  InventoryIntake imported successfully")

    return True


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "="*60)
    print("  OCR INVENTORY PIPELINE AUTO-TEST")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Test image: {TEST_IMAGE}")
    print(f"Image exists: {TEST_IMAGE.exists()}")

    results = []
    ocr_text = None

    # Test 1: OCR import
    passed, _ = test_step("Import OCR module", test_ocr_import)
    results.append(("Import OCR module", passed))
    if not passed:
        print("\n⚠️  OCR module not available. Install olmocr in Python 3.11+ environment.")
        return results

    # Test 2: Terminal bot import
    passed, ocr_available = test_step("Import terminal_bot", test_terminal_bot_import)
    results.append(("Import terminal_bot", passed))

    if not ocr_available:
        print("\n⚠️  OCR not available in terminal_bot. Check imports.")
        return results

    # Test 3: OCR extraction
    passed, ocr_text = test_step("OCR text extraction", test_ocr_extraction)
    results.append(("OCR text extraction", passed))

    if not passed or not ocr_text:
        print("\n⚠️  OCR extraction failed. Cannot continue.")
        return results

    # Test 4: Parse inventory
    passed, parsed = test_step("Parse inventory list", lambda: test_parse_inventory(ocr_text))
    results.append(("Parse inventory list", passed))

    # Test 5: Full vendor image processing
    passed, _ = test_step("Process vendor image", test_process_vendor_image)
    results.append(("Process vendor image", passed))

    # Test 6: Inventory intake import
    passed, _ = test_step("Import inventory intake", test_inventory_intake_import)
    results.append(("Import inventory intake", passed))

    # Summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

    print(f"\n  Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed! Pipeline is ready.")
    else:
        print("\n⚠️  Some tests failed. See details above.")

    return results


if __name__ == "__main__":
    run_all_tests()
