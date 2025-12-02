#!/bin/bash
# Start Vendor System (API + JSON Server + Frontend)
# Usage: ./start_vendor_system.sh [--with-frontend]

cd "$(dirname "$0")/.."

echo "========================================"
echo "🏪 Starting BeeKurse Vendor System"
echo "========================================"

# Activate virtual environment
if [ -d ~/kurse ]; then
    source ~/kurse/bin/activate
    echo "✅ Virtual environment activated"
fi

# Check if json-server is installed
if ! command -v json-server &> /dev/null; then
    echo "⚠️  json-server not found. Installing..."
    npm install -g json-server
fi

# Create db.json if it doesn't exist
DB_JSON="data/vendor_data/db.json"
if [ ! -f "$DB_JSON" ]; then
    echo "📄 Creating initial database file..."
    mkdir -p data/vendor_data
    echo '{"users": [], "products": []}' > "$DB_JSON"
fi

# Start JSON Server in background (port 3001)
echo ""
echo "🗄️  Starting JSON Server on port 3001..."
json-server --watch "$DB_JSON" --port 3001 &
JSON_PID=$!
sleep 2

# Start Vendor API in background (port 8000)
echo ""
echo "🚀 Starting Vendor API on port 8000..."
python backend/vendor_api.py &
API_PID=$!
sleep 2

echo ""
echo "========================================"
echo "✅ Vendor System Started!"
echo "========================================"
echo "📍 Vendor API:    http://localhost:8000"
echo "📍 JSON Server:   http://localhost:3001"
echo "📍 Test DB:       data/databases/sql/vendor_test.db"
echo ""
echo "Endpoints:"
echo "  POST /token          - Login"
echo "  POST /register       - Register vendor"
echo "  GET  /users/me       - Get profile"
echo "  POST /files/         - Upload product"
echo "  GET  /products/me    - List my products"
echo "========================================"

# Check if --with-frontend flag is passed
if [ "$1" = "--with-frontend" ]; then
    echo ""
    echo "🎨 Starting Frontend dev server..."
    cd vendor_frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..
    echo "📍 Frontend:      http://localhost:5173"
fi

# Wait for user to stop
echo ""
echo "Press Ctrl+C to stop all services..."

# Cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $JSON_PID 2>/dev/null
    kill $API_PID 2>/dev/null
    [ -n "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

trap cleanup INT TERM

# Keep script running
wait
