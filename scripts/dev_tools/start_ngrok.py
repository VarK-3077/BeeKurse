"""
Start ngrok tunnel and display public URL
"""
from pyngrok import ngrok
import time

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
print(f"   Verify Token: my_verify_token_123")
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
