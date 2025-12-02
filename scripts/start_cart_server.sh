#!/bin/bash
# Start Cart & Wishlist Server with ngrok tunnel
# Usage: ./start_cart_server.sh

cd "$(dirname "$0")/.."

echo "========================================"
echo "🛒 Starting BeeKurse Cart Server"
echo "========================================"

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found. Please install ngrok first:"
    echo "   https://ngrok.com/download"
    echo ""
    echo "Starting server without ngrok tunnel..."
    echo "Cart API will be available at: http://localhost:8002"
    echo ""
    source ~/kurse/bin/activate
    python backend/cart_api.py
    exit 0
fi

# Start the cart server in background
echo "Starting cart API server on port 8002..."
source ~/kurse/bin/activate
python backend/cart_api.py &
CART_PID=$!
sleep 2

# Start ngrok
echo ""
echo "Starting ngrok tunnel..."
echo ""
ngrok http 8002 --url https://unribbed-affluently-kody.ngrok-free.dev --pooling-enabled=true
# Cleanup on exit
kill $CART_PID 2>/dev/null
