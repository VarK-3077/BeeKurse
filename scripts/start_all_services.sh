#!/bin/bash

# BeeKurse - All Services Startup Script (Unified Gateway Mode)
# This script starts all required backend services with a SINGLE ngrok tunnel
#
# Architecture:
#   - Unified Gateway (Port 8000) - Routes all external requests
#   - Strontium API (Port 5001) - Search/Chat backend
#   - Gallery Frontend (Port 5400) - React app (proxied through gateway)
#   - Vendor Frontend (Port 3000) - Vendor registration/dashboard
#   - Single ngrok tunnel on port 8000

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Set PYTHONPATH to include project root for module imports
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Log directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# PID file to track running processes
PID_FILE="$LOG_DIR/services.pid"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[BeeKurse]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for a service to be ready
wait_for_service() {
    local port=$1
    local name=$2
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if check_port $port; then
            print_success "$name is ready on port $port"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done

    print_error "$name failed to start on port $port"
    return 1
}

# Function to stop all services
stop_services() {
    print_status "Stopping all services..."

    if [ -f "$PID_FILE" ]; then
        while read pid; do
            if kill -0 $pid 2>/dev/null; then
                kill $pid 2>/dev/null || true
                print_status "Stopped process $pid"
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi

    # Kill any remaining processes on our ports
    for port in 8000 5001 5400 3000; do
        local pid=$(lsof -t -i:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill $pid 2>/dev/null || true
            print_status "Killed process on port $port"
        fi
    done

    # Kill ngrok processes
    pkill -f "ngrok http" 2>/dev/null || true

    print_success "All services stopped"
}

# Function to start Unified Gateway (Port 8000)
# This replaces the separate whatsapp_bot.py and cart_api.py
start_unified_gateway() {
    print_status "Starting Unified Gateway on port 8000..."

    if check_port 8000; then
        print_warning "Port 8000 already in use, skipping Unified Gateway"
        return 1
    fi

    cd "$PROJECT_ROOT"
    python3 -m backend.unified_gateway > "$LOG_DIR/unified_gateway.log" 2>&1 &
    echo $! >> "$PID_FILE"

    wait_for_service 8000 "Unified Gateway"
}

# Function to start Strontium API Backend (Port 5001)
start_strontium_api() {
    print_status "Starting Strontium API Backend on port 5001..."

    if check_port 5001; then
        print_warning "Port 5001 already in use, skipping Strontium API"
        return 1
    fi

    cd "$PROJECT_ROOT"
    python3 -m backend.strontium_api > "$LOG_DIR/strontium_api.log" 2>&1 &
    echo $! >> "$PID_FILE"

    wait_for_service 5001 "Strontium API"
}

# Function to start Gallery Frontend (Port 5400)
# This is proxied through the unified gateway at /gallery
start_gallery_frontend() {
    print_status "Starting Gallery Frontend on port 5400..."

    if check_port 5400; then
        print_warning "Port 5400 already in use, skipping Gallery Frontend"
        return 1
    fi

    cd "$PROJECT_ROOT"

    # Check if gallery_api.py exists, if not use whatsapp-frontend
    if [ -f "backend/gallery_api.py" ]; then
        python3 -m backend.gallery_api > "$LOG_DIR/gallery_api.log" 2>&1 &
        echo $! >> "$PID_FILE"
        wait_for_service 5400 "Gallery API"
    else
        # Run whatsapp-frontend on port 5400
        cd "$PROJECT_ROOT/whatsapp-frontend"
        if [ -d "node_modules" ]; then
            npm run dev -- --port 5400 > "$LOG_DIR/gallery_frontend.log" 2>&1 &
            echo $! >> "$PID_FILE"
            wait_for_service 5400 "Gallery Frontend"
        else
            print_warning "whatsapp-frontend dependencies not installed. Run: cd whatsapp-frontend && npm install"
            return 1
        fi
    fi
}

# Function to start Vendor Frontend (Port 3000)
# This serves the vendor registration and dashboard
start_vendor_frontend() {
    print_status "Starting Vendor Frontend on port 3000..."

    if check_port 3000; then
        print_warning "Port 3000 already in use, skipping Vendor Frontend"
        return 1
    fi

    cd "$PROJECT_ROOT"

    # Run vendor_frontend on port 3000
    if [ -d "$PROJECT_ROOT/vendor_frontend" ]; then
        cd "$PROJECT_ROOT/vendor_frontend"
        if [ -d "node_modules" ]; then
            npm run dev -- --port 3000 --host > "$LOG_DIR/vendor_frontend.log" 2>&1 &
            echo $! >> "$PID_FILE"
            wait_for_service 3000 "Vendor Frontend"
        else
            print_warning "vendor_frontend dependencies not installed. Run: cd vendor_frontend && npm install"
            return 1
        fi
    else
        print_warning "vendor_frontend directory not found"
        return 1
    fi

    cd "$PROJECT_ROOT"
}

# Function to start single ngrok tunnel for Unified Gateway (port 8000)
start_ngrok() {
    print_status "Starting ngrok tunnel for Unified Gateway (port 8000)..."

    # Check if ngrok is installed
    if ! command -v ngrok &> /dev/null; then
        print_warning "ngrok not installed. Please install from https://ngrok.com/download"
        print_warning "Skipping ngrok tunnel..."
        return 1
    fi

    # Get ngrok URL from environment
    local ngrok_url="${NGROK_URL:-}"

    if [ -n "$ngrok_url" ]; then
        print_status "Using static ngrok URL: $ngrok_url"
        ngrok http 8000 --url "$ngrok_url" > "$LOG_DIR/ngrok.log" 2>&1 &
    else
        ngrok http 8000 > "$LOG_DIR/ngrok.log" 2>&1 &
    fi
    echo $! >> "$PID_FILE"

    sleep 3
    print_success "ngrok tunnel started for port 8000"

    # Try to get the ngrok URL
    local tunnel_url=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"https://[^"]*' | head -1 | cut -d'"' -f4)
    if [ -n "$tunnel_url" ]; then
        print_success "ngrok URL: $tunnel_url"
        echo "$tunnel_url" > "$PROJECT_ROOT/data/ngrok_url.txt"
        echo ""
        print_success "All services available at:"
        echo "  - Webhook:    $tunnel_url/webhook"
        echo "  - Cart:       $tunnel_url/cart/{user_id}"
        echo "  - Wishlist:   $tunnel_url/wishlist/{user_id}"
        echo "  - Cart Page:  $tunnel_url/view/cart/{user_id}"
        echo "  - Gallery:    $tunnel_url/gallery?user={user_id}"
        echo "  - Vendor:     $tunnel_url/vendor"
    fi
}

# Display usage information
usage() {
    echo ""
    echo -e "${CYAN}BeeKurse Unified Gateway Startup Script${NC}"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start       Start all services (default)"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show status of all services"
    echo "  logs        Show log files location"
    echo ""
    echo "Services started:"
    echo "  - Unified Gateway     (Port 8000) - Routes all external requests"
    echo "  - Strontium API       (Port 5001) - Search/Chat backend"
    echo "  - Gallery Frontend    (Port 5400) - React app (proxied through gateway)"
    echo "  - Vendor Frontend     (Port 3000) - Vendor registration/dashboard"
    echo "  - Single ngrok tunnel (Port 8000 -> public URL)"
    echo ""
    echo "All services available at single ngrok URL:"
    echo "  - Webhook:    /webhook"
    echo "  - Cart API:   /cart/{user_id}"
    echo "  - Cart Page:  /view/cart/{user_id}"
    echo "  - Gallery:    /gallery?user={user_id}"
    echo "  - Vendor:     /vendor"
    echo ""
}

# Show status of all services
show_status() {
    echo ""
    echo -e "${CYAN}BeeKurse Services Status${NC}"
    echo "=========================="

    for port in 8000 5001 5400 3000; do
        if check_port $port; then
            local pid=$(lsof -t -i:$port 2>/dev/null | head -1)
            case $port in
                8000) print_success "Unified Gateway (Port $port): Running (PID: $pid)" ;;
                5001) print_success "Strontium API (Port $port): Running (PID: $pid)" ;;
                5400) print_success "Gallery Frontend (Port $port): Running (PID: $pid)" ;;
                3000) print_success "Vendor Frontend (Port $port): Running (PID: $pid)" ;;
            esac
        else
            case $port in
                8000) print_error "Unified Gateway (Port $port): Not running" ;;
                5001) print_error "Strontium API (Port $port): Not running" ;;
                5400) print_error "Gallery Frontend (Port $port): Not running" ;;
                3000) print_error "Vendor Frontend (Port $port): Not running" ;;
            esac
        fi
    done

    # Check ngrok
    if pgrep -f "ngrok http" > /dev/null; then
        print_success "ngrok: Running"
        local tunnel_url=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"https://[^"]*' | head -1 | cut -d'"' -f4)
        if [ -n "$tunnel_url" ]; then
            echo -e "         URL: $tunnel_url"
            echo ""
            echo "  Endpoints:"
            echo "    Webhook:   $tunnel_url/webhook"
            echo "    Cart:      $tunnel_url/view/cart/{user_id}"
            echo "    Gallery:   $tunnel_url/gallery?user={user_id}"
            echo "    Vendor:    $tunnel_url/vendor"
        fi
    else
        print_warning "ngrok: Not running"
    fi

    echo ""
}

# Show log file locations
show_logs() {
    echo ""
    echo -e "${CYAN}Log Files Location${NC}"
    echo "==================="
    echo "Directory: $LOG_DIR"
    echo ""
    echo "Files:"
    ls -la "$LOG_DIR"/*.log 2>/dev/null || echo "No log files yet"
    echo ""
    echo "To view logs in real-time:"
    echo "  tail -f $LOG_DIR/unified_gateway.log"
    echo "  tail -f $LOG_DIR/strontium_api.log"
    echo "  tail -f $LOG_DIR/gallery_frontend.log"
    echo "  tail -f $LOG_DIR/vendor_frontend.log"
    echo "  tail -f $LOG_DIR/ngrok.log"
    echo ""
}

# Main function to start all services
start_all() {
    echo ""
    echo -e "${CYAN}=============================================${NC}"
    echo -e "${CYAN}  BeeKurse - Unified Gateway Mode           ${NC}"
    echo -e "${CYAN}  Single ngrok URL for all services         ${NC}"
    echo -e "${CYAN}=============================================${NC}"
    echo ""

    # Clear old PID file
    rm -f "$PID_FILE"

    # Start services in order
    # 1. Start Strontium API first (search/chat backend)
    start_strontium_api
    sleep 1

    # 2. Start Gallery Frontend (proxied through gateway)
    start_gallery_frontend
    sleep 1

    # 3. Start Vendor Frontend (registration/dashboard)
    start_vendor_frontend
    sleep 1

    # 4. Start Unified Gateway (replaces whatsapp_bot + cart_api)
    start_unified_gateway
    sleep 1

    # 5. Start single ngrok tunnel
    start_ngrok

    echo ""
    echo -e "${GREEN}=============================================${NC}"
    echo -e "${GREEN}  All Services Started Successfully          ${NC}"
    echo -e "${GREEN}=============================================${NC}"
    echo ""

    show_status

    echo "Press Ctrl+C to stop all services, or run: $0 stop"
    echo ""

    # Wait for interrupt
    trap stop_services SIGINT SIGTERM
    wait
}

# Handle command line arguments
case "${1:-start}" in
    start)
        start_all
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 2
        start_all
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        print_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
