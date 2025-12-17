#!/bin/bash
# =============================================================================
# POD OCR - Model Download Script
# =============================================================================
# Downloads the InternVL2-8B model from HuggingFace
# Run this before creating the deployment package
# =============================================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
MODEL_REPO="OpenGVLab/InternVL2-8B"
CACHE_DIR="/workspace/.cache/huggingface"
VENV_PATH="/workspace/POD_OCR/venv"

print_header() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         POD OCR - Model Download Script                 ║${NC}"
    echo -e "${CYAN}║                                                          ║${NC}"
    echo -e "${CYAN}║         Downloading InternVL2-8B Model                   ║${NC}"
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

check_internet() {
    print_info "Checking internet connectivity..."
    
    if ! ping -c 1 google.com &> /dev/null; then
        print_error "No internet connection detected"
        echo -e "    ${RED}This script requires internet to download the model${NC}"
        exit 1
    fi
    
    print_success "Internet connection OK"
}

check_venv() {
    print_info "Checking virtual environment..."
    
    if [ ! -d "$VENV_PATH" ]; then
        print_error "Virtual environment not found at: $VENV_PATH"
        echo -e "    ${YELLOW}Create it with: python3 -m venv $VENV_PATH${NC}"
        exit 1
    fi
    
    print_success "Virtual environment found"
}

install_dependencies() {
    print_progress "Installing/verifying dependencies..."
    
    # Activate venv
    source "$VENV_PATH/bin/activate"
    
    # Install huggingface-hub if needed
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        print_info "Installing huggingface-hub..."
        pip install huggingface-hub -q
    fi
    
    print_success "Dependencies ready"
}

download_model() {
    print_progress "Downloading model: $MODEL_REPO"
    echo -e "    ${YELLOW}This will download ~15-20GB of data${NC}"
    echo -e "    ${YELLOW}Estimated time: 15-30 minutes${NC}"
    echo
    
    # Activate venv
    source "$VENV_PATH/bin/activate"
    
    # Download with progress
    python3 << PYTHON_SCRIPT
import os
from huggingface_hub import snapshot_download
from tqdm import tqdm
import sys

print("Starting download...")
print(f"Model: $MODEL_REPO")
print(f"Cache: $CACHE_DIR")
print()

try:
    # Download model with progress bar
    model_path = snapshot_download(
        repo_id="$MODEL_REPO",
        cache_dir="$CACHE_DIR",
        resume_download=True,
        local_files_only=False
    )
    
    print()
    print(f"✓ Model downloaded successfully!")
    print(f"  Location: {model_path}")
    
    # Get model size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    
    size_gb = total_size / (1024**3)
    print(f"  Size: {size_gb:.2f} GB")
    
except KeyboardInterrupt:
    print()
    print("✗ Download cancelled by user")
    sys.exit(1)
except Exception as e:
    print()
    print(f"✗ Error downloading model: {e}")
    sys.exit(1)
PYTHON_SCRIPT
    
    if [ $? -eq 0 ]; then
        print_success "Model download complete"
    else
        print_error "Model download failed"
        exit 1
    fi
}

verify_model() {
    print_progress "Verifying model files..."
    
    # Find model directory
    MODEL_DIR=$(find "$CACHE_DIR" -maxdepth 3 -type d -name "models--OpenGVLab--InternVL2-8B" \
            -not -path "*/.locks/*" | head -1)
    
    if [ -z "$MODEL_DIR" ]; then
        print_error "Model directory not found"
        exit 1
    fi
    
    print_success "Model directory found: $MODEL_DIR"
    
    # Check for key files
    SNAPSHOT_DIR=$(find "$MODEL_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)
    
    if [ -z "$SNAPSHOT_DIR" ]; then
        print_error "Model snapshot not found"
        exit 1
    fi
    
    print_success "Model snapshot found: $SNAPSHOT_DIR"
    
    # List model files
    echo
    echo -e "${CYAN}Model files:${NC}"
    ls -lh "$SNAPSHOT_DIR" | grep -E '\.(bin|safetensors|json)$' | while read line; do
        echo -e "  ${YELLOW}•${NC} $line"
    done
}

print_summary() {
    echo
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           Model Download Complete!                      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${CYAN}Next Steps:${NC}"
    echo -e "  ${YELLOW}1.${NC} Build executables: ${BLUE}./build_all.sh${NC}"
    echo -e "  ${YELLOW}2.${NC} Create deployment package: ${BLUE}./create_deployment_package.sh${NC}"
    echo
    echo -e "${CYAN}Model Information:${NC}"
    echo -e "  ${YELLOW}•${NC} Repository: ${GREEN}$MODEL_REPO${NC}"
    echo -e "  ${YELLOW}•${NC} Cache Location: ${GREEN}$CACHE_DIR${NC}"
    echo
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    print_header
    
    check_internet
    check_venv
    install_dependencies
    
    echo
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}                    Downloading Model                     ${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    download_model
    
    echo
    verify_model
    
    print_summary
}

main