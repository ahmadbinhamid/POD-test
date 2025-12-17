#!/bin/bash
# =============================================================================
# POD OCR Distributed Services Launcher
# =============================================================================
set -e  # Exit on error

# Environment Variables
export HF_HOME=/workspace/.cache
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Add common Python/conda paths
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
if [ -d "/opt/conda/bin" ]; then
    export PATH="/opt/conda/bin:$PATH"
fi
if [ -d "$HOME/.local/bin" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Configuration
PROJECT_DIR="/workspace/POD_OCR"
LOG_DIR="$PROJECT_DIR/logs"
MODEL_PATH="/workspace/.cache/huggingface/models--OpenGVLab--InternVL2-8B/snapshots/6fb9ad6924f69424e57fab2ab061d707688f0296"

# Color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# =============================================================================
# Utility Functions
# =============================================================================
print_header() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         POD OCR Distributed Services Launcher           ║${NC}"
    echo -e "${CYAN}║                                                          ║${NC}"
    echo -e "${CYAN}║          Running All Services from Built Executables     ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════╝${NC}"
    echo
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[ℹ]${NC} $1"
}

print_progress() {
    echo -e "${BLUE}[→]${NC} $1"
}

# Function to check if service is ready with progress indicator
check_service_health() {
    local service_name=$1
    local url=$2
    local max_attempts=$3
    local attempt=1
    
    print_info "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            print_success "$service_name is ready! (${url}) [${attempt}/${max_attempts}s]"
            return 0
        fi
        
        # Show progress every 5 attempts
        if [ $((attempt % 5)) -eq 0 ]; then
            echo -e "    ${YELLOW}Progress: ${attempt}/${max_attempts} seconds${NC}"
        fi
        
        sleep 1
        ((attempt++))
    done
    
    print_error "$service_name failed to start after $max_attempts seconds"
    echo -e "    ${RED}Check log file: $LOG_DIR/lmdeploy_serve.log${NC}"
    return 1
}

# Function to cleanup existing processes
cleanup_processes() {
    print_info "Cleaning up existing processes..."
    
    # Kill built executables
    if pgrep -f "dist_lmdeploy/my_app/my_app" > /dev/null; then
        pkill -f "dist_lmdeploy/my_app/my_app" 2>/dev/null || true
        echo -e "    ${MAGENTA}Stopped LMDeploy executable${NC}"
    fi
    
    if pgrep -f "dist_pixtral/my_app/my_app" > /dev/null; then
        pkill -f "dist_pixtral/my_app/my_app" 2>/dev/null || true
        echo -e "    ${MAGENTA}Stopped Pixtral executable${NC}"
    fi
    
    if pgrep -f "dist_app/my_app/my_app" > /dev/null; then
        pkill -f "dist_app/my_app/my_app" 2>/dev/null || true
        echo -e "    ${MAGENTA}Stopped App executable${NC}"
    fi
    
    # Also kill any lmdeploy serve processes
    if pgrep -f "lmdeploy serve" > /dev/null; then
        pkill -f "lmdeploy serve" 2>/dev/null || true
        echo -e "    ${MAGENTA}Stopped lmdeploy serve processes${NC}"
    fi
    
    sleep 2
    print_success "Process cleanup completed"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if running in correct directory
    if [ ! -d "$PROJECT_DIR" ]; then
        print_error "Project directory not found: $PROJECT_DIR"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    print_success "Working directory: $PROJECT_DIR"
    
    # Check for lmdeploy command
    if ! command -v lmdeploy &> /dev/null; then
        print_error "lmdeploy command not found in PATH"
        echo -e "    ${RED}Current PATH: $PATH${NC}"
        echo -e "    ${YELLOW}Try running: which lmdeploy${NC}"
        exit 1
    fi
    LMDEPLOY_PATH=$(which lmdeploy)
    print_success "LMDeploy found: $LMDEPLOY_PATH"
    
    # Check for built executables
    if [ ! -f "./dist_app/my_app/my_app" ]; then
        print_error "dist_app/my_app/my_app not found. Run build_all.sh first."
        exit 1
    fi
    print_success "App build found: ./dist_app/my_app/my_app"
    
    if [ ! -f "./dist_pixtral/my_app/my_app" ]; then
        print_error "dist_pixtral/my_app/my_app not found. Run build_all.sh first."
        exit 1
    fi
    print_success "Pixtral build found: ./dist_pixtral/my_app/my_app"
    
    if [ ! -f "./dist_lmdeploy/my_app/my_app" ]; then
        print_error "dist_lmdeploy/my_app/my_app not found. Run build_all.sh first."
        exit 1
    fi
    print_success "LMDeploy build found: ./dist_lmdeploy/my_app/my_app"
    
    # Check for model path
    if [ ! -d "$MODEL_PATH" ]; then
        print_error "Model not found: $MODEL_PATH"
        exit 1
    fi
    print_success "Model found: $MODEL_PATH"
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    print_success "Log directory ready: $LOG_DIR"
    
    return 0
}

# Function to handle cleanup on exit
cleanup() {
    echo
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}           Stopping All Applications...                  ${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ ! -z "$APP_PID" ]; then
        kill $APP_PID 2>/dev/null && echo -e "    ${MAGENTA}Stopped App (PID: $APP_PID)${NC}" || true
    fi
    
    if [ ! -z "$PIXTRAL_PID" ]; then
        kill $PIXTRAL_PID 2>/dev/null && echo -e "    ${MAGENTA}Stopped Pixtral (PID: $PIXTRAL_PID)${NC}" || true
    fi
    
    if [ ! -z "$LMDEPLOY_EXEC_PID" ]; then
        kill $LMDEPLOY_EXEC_PID 2>/dev/null && echo -e "    ${MAGENTA}Stopped LMDeploy Exec (PID: $LMDEPLOY_EXEC_PID)${NC}" || true
    fi
    
    if [ ! -z "$LMDEPLOY_SERVE_PID" ]; then
        kill $LMDEPLOY_SERVE_PID 2>/dev/null && echo -e "    ${MAGENTA}Stopped LMDeploy Serve (PID: $LMDEPLOY_SERVE_PID)${NC}" || true
    fi
    
    print_success "All applications stopped"
    exit 0
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    print_header
    
    # Step 1: Check prerequisites
    check_prerequisites
    
    # Step 2: Cleanup existing processes
    cleanup_processes
    
    # Step 3: Start services
    echo
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}                    Starting Services                     ${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    # Start LMDeploy serve first
    print_progress "Starting LMDeploy API Server (port 23333)..."
    
    # Use nohup and explicit shell to ensure proper backgrounding
    nohup lmdeploy serve api_server "$MODEL_PATH" \
        --model-name InternVL2-8B \
        --server-name 0.0.0.0 \
        --server-port 23333 \
        --tp 1 \
        --backend turbomind > "$LOG_DIR/lmdeploy_serve.log" 2>&1 &
    
    LMDEPLOY_SERVE_PID=$!
    echo -e "    ${CYAN}PID: $LMDEPLOY_SERVE_PID${NC}"
    echo -e "    ${YELLOW}Log: tail -f $LOG_DIR/lmdeploy_serve.log${NC}"
    
    # Wait for LMDeploy serve to be ready
    if ! check_service_health "LMDeploy API Server" "http://localhost:23333/v1/models" 120; then
        print_error "Failed to start LMDeploy API Server"
        echo -e "${YELLOW}Last 20 lines of log:${NC}"
        tail -20 "$LOG_DIR/lmdeploy_serve.log"
        exit 1
    fi
    
    # Start LMDeploy executable
    print_progress "Starting LMDeploy executable..."
    ./dist_lmdeploy/my_app/my_app > "$LOG_DIR/lmdeploy_exec.log" 2>&1 &
    LMDEPLOY_EXEC_PID=$!
    echo -e "    ${CYAN}PID: $LMDEPLOY_EXEC_PID${NC}"
    sleep 3
    
    # Start Pixtral
    print_progress "Starting Pixtral app (port 3203)..."
    ./dist_pixtral/my_app/my_app > "$LOG_DIR/pixtral.log" 2>&1 &
    PIXTRAL_PID=$!
    echo -e "    ${CYAN}PID: $PIXTRAL_PID${NC}"
    sleep 3
    
    # Start main app
    print_progress "Starting main app (port 8080)..."
    ./dist_app/my_app/my_app > "$LOG_DIR/app.log" 2>&1 &
    APP_PID=$!
    echo -e "    ${CYAN}PID: $APP_PID${NC}"
    
    # Final status
    echo
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║             All Applications Started!                   ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${CYAN}Service PIDs:${NC}"
    echo -e "  ${YELLOW}•${NC} LMDeploy Serve:  ${CYAN}$LMDEPLOY_SERVE_PID${NC}"
    echo -e "  ${YELLOW}•${NC} LMDeploy Exec:   ${CYAN}$LMDEPLOY_EXEC_PID${NC}"
    echo -e "  ${YELLOW}•${NC} Pixtral:         ${CYAN}$PIXTRAL_PID${NC}"
    echo -e "  ${YELLOW}•${NC} Main App:        ${CYAN}$APP_PID${NC}"
    echo
    echo -e "${CYAN}Service Endpoints:${NC}"
    echo -e "  ${YELLOW}•${NC} LMDeploy API:    ${BLUE}http://localhost:23333${NC}"
    echo -e "  ${YELLOW}•${NC} Pixtral:         ${BLUE}http://localhost:3203${NC}"
    echo -e "  ${YELLOW}•${NC} Main App:        ${BLUE}http://localhost:8080${NC}"
    echo
    echo -e "${CYAN}Logs:${NC}"
    echo -e "  ${YELLOW}•${NC} LMDeploy Serve:  ${MAGENTA}$LOG_DIR/lmdeploy_serve.log${NC}"
    echo -e "  ${YELLOW}•${NC} LMDeploy Exec:   ${MAGENTA}$LOG_DIR/lmdeploy_exec.log${NC}"
    echo -e "  ${YELLOW}•${NC} Pixtral:         ${MAGENTA}$LOG_DIR/pixtral.log${NC}"
    echo -e "  ${YELLOW}•${NC} Main App:        ${MAGENTA}$LOG_DIR/app.log${NC}"
    echo
    echo -e "${YELLOW}Press CTRL+C to stop all applications${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Set up trap for CTRL+C
    trap cleanup SIGINT SIGTERM
    
    # Wait for all background processes
    wait $APP_PID $PIXTRAL_PID $LMDEPLOY_EXEC_PID $LMDEPLOY_SERVE_PID
}

# Run main function
main