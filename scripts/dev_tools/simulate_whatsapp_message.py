"""
Simulate a WhatsApp message to test the full flow locally
Bypasses Meta and directly calls the webhook
"""
import requests
import json

print("="*80)
print("SIMULATING WHATSAPP MESSAGE".center(80))
print("="*80)

# Sample WhatsApp webhook payload (what Meta would send)
webhook_payload = {
    "entry": [{
        "changes": [{
            "value": {
                "messages": [{
                    "from": "917893127444",
                    "type": "text",
                    "text": {
                        "body": "Hello!"
                    }
                }]
            }
        }]
    }]
}

print(f"\n📤 Simulating WhatsApp message:")
print(f"   From: 917893127444")
print(f"   Message: \"Hello!\"")

# Send to local webhook
webhook_url = "http://localhost:8000/webhook"

print(f"\n🔄 Sending to webhook: {webhook_url}")

try:
    response = requests.post(webhook_url, json=webhook_payload, timeout=30)

    print(f"\n📊 Webhook Response:")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")

    if response.ok:
        print(f"\n✅ SUCCESS!")
        print(f"\n💬 The webhook should have:")
        print(f"   1. Received the message")
        print(f"   2. Sent it to Strontium backend")
        print(f"   3. Got a chat response")
        print(f"   4. Sent it back to WhatsApp API")
        print(f"\n📱 Check the backend logs above to see what happened!")
    else:
        print(f"\n❌ Webhook returned error")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
