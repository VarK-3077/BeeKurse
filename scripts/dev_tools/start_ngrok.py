"""
Start ngrok tunnel and display public URL
"""
import os
from pyngrok import ngrok
import time
from dotenv import load_dotenv

load_dotenv()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
if not VERIFY_TOKEN:
    raise ValueError("VERIFY_TOKEN environment variable is required")

print("="*80)
print("STARTING NGROK TUNNEL".center(80))
print("="*80)

# Start ngrok tunnel on port 8000
print("\n🚀 Starting ngrok tunnel on port 8000...")
public_url = ngrok.connect(8000)

print(f"\n✅ Ngrok tunnel started successfully!")
print(f"\n{'='*80}")
print(f"PUBLIC URL: {public_url}")
print(f"{'='*80}")
print(f"\n📋 Webhook Configuration for Meta:")
print(f"   Callback URL: {public_url}/webhook")
print(f"   Verify Token: {VERIFY_TOKEN}")
print(f"\n{'='*80}")
print(f"\n⏳ Tunnel is active. Press Ctrl+C to stop...")
print(f"{'='*80}\n")

try:
    # Keep running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\n🛑 Stopping ngrok tunnel...")
    ngrok.disconnect(public_url)
    print("✅ Tunnel stopped")
