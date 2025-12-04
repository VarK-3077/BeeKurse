#!/bin/bash
# Start Cart & Wishlist Server with ngrok tunnel
# Usage: ./start_cart_server.sh

cd "$(dirname "$0")/.."

echo "========================================"
echo "🛒 Starting BeeKurse Cart Server"
echo "========================================"

# Activate virtual environment first (ngrok is installed there)
source ~/kurse/bin/activate

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check for CART_NGROK_URL
if [ -z "$CART_NGROK_URL" ]; then
    echo "⚠️  CART_NGROK_URL not set in .env file"
    echo "   Add: CART_NGROK_URL=https://your-domain.ngrok-free.dev"
    echo ""
fi

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found. Please install ngrok first:"
    echo "   https://ngrok.com/download"
    echo ""
    echo "Starting server without ngrok tunnel..."
    echo "Cart API will be available at: http://localhost:8002"
    echo ""
    python backend/cart_api.py
    exit 0
fi

# Start the cart server in background
echo "Starting cart API server on port 8002..."
python backend/cart_api.py &
CART_PID=$!
sleep 2

# Start ngrok
echo ""
echo "Starting ngrok tunnel..."
echo ""
ngrok http 8002 --url $CART_NGROK_URL --pooling-enabled=true
# Cleanup on exit
kill $CART_PID 2>/dev/null
