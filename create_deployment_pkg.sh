#!/bin/bash
# =============================================================================
# POD OCR - Deployment Package Creator
# =============================================================================
# This script creates a portable deployment package for isolated systems
# Run this on a system WITH internet access
# =============================================================================

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/workspace/POD_OCR"
DEPLOY_DIR="POD_OCR_DEPLOY"
MODEL_REPO="OpenGVLab/InternVL2-8B"
HF_CACHE="/workspace/.cache/huggingface"

print_header() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         POD OCR Deployment Package Creator              ║${NC}"
    echo -e "${CYAN}║                                                          ║${NC}"
    echo -e "${CYAN}║     Creating portable package for isolated systems      ║${NC}"
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

check_prerequisites() {
    print_info "Checking prerequisites..."
    
    if [ ! -d "$PROJECT_DIR" ]; then
        print_error "Project directory not found: $PROJECT_DIR"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    print_success "Working directory: $PROJECT_DIR"
    
    # Check for executables
    if [ ! -f "./dist_app/my_app/my_app" ]; then
        print_error "dist_app/my_app/my_app not found. Run build_all.sh first."
        exit 1
    fi
    
    if [ ! -f "./dist_pixtral/my_app/my_app" ]; then
        print_error "dist_pixtral/my_app/my_app not found. Run build_all.sh first."
        exit 1
    fi
    
    if [ ! -f "./dist_lmdeploy/my_app/my_app" ]; then
        print_error "dist_lmdeploy/my_app/my_app not found. Run build_all.sh first."
        exit 1
    fi
    
    print_success "All executables found"
    
    # Check for venv
    if [ ! -d "./venv" ]; then
        print_error "Virtual environment not found. Create it first."
        exit 1
    fi
    print_success "Virtual environment found"
    
    # Check for required commands
    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found"
        exit 1
    fi
    
    print_success "All prerequisites met"
}

download_model() {
    print_info "Checking/Downloading model..."
    
    # Activate venv
    source venv/bin/activate
    
    # Install huggingface-hub if not present
    pip install huggingface-hub -q
    
    # Download model
    python3 << 'PYTHON_SCRIPT'
import os
from huggingface_hub import snapshot_download

model_repo = "OpenGVLab/InternVL2-8B"
cache_dir = "/workspace/.cache/huggingface"

print(f"Downloading/Verifying model: {model_repo}")
print(f"Cache directory: {cache_dir}")

try:
    model_path = snapshot_download(
        repo_id=model_repo,
        cache_dir=cache_dir,
        resume_download=True
    )
    print(f"✓ Model ready at: {model_path}")
except Exception as e:
    print(f"✗ Error downloading model: {e}")
    exit(1)
PYTHON_SCRIPT
    
    if [ $? -eq 0 ]; then
        print_success "Model downloaded/verified"
    else
        print_error "Model download failed"
        exit 1
    fi
}

create_deployment_structure() {
    print_info "Creating deployment structure..."
    
    # Remove old deployment if exists
    if [ -d "$DEPLOY_DIR" ]; then
        rm -rf "$DEPLOY_DIR"
    fi
    
    mkdir -p "$DEPLOY_DIR"
    mkdir -p "$DEPLOY_DIR/logs"
    
    print_success "Deployment directory created: $DEPLOY_DIR"
}

copy_executables() {
    print_progress "Copying executables..."
    
    cp -r dist_app "$DEPLOY_DIR/"
    echo -e "    ${CYAN}Copied: dist_app${NC}"
    
    cp -r dist_pixtral "$DEPLOY_DIR/"
    echo -e "    ${CYAN}Copied: dist_pixtral${NC}"
    
    cp -r dist_lmdeploy "$DEPLOY_DIR/"
    echo -e "    ${CYAN}Copied: dist_lmdeploy${NC}"
    
    print_success "Executables copied"
}

copy_venv() {
    print_progress "Copying virtual environment (this may take a while)..."
    
    cp -r venv "$DEPLOY_DIR/"
    
    # Get size
    VENV_SIZE=$(du -sh "$DEPLOY_DIR/venv" | cut -f1)
    print_success "Virtual environment copied (Size: $VENV_SIZE)"
}

copy_model() {
    print_progress "Copying model files (this may take a while)..."
    
    mkdir -p "$DEPLOY_DIR/model_cache"
    
    # Find the actual model directory
    MODEL_DIR=$(find "$HF_CACHE" -type d -name "models--OpenGVLab--InternVL2-8B" 2>/dev/null | head -1)
    
    if [ -z "$MODEL_DIR" ]; then
        print_error "Model directory not found in cache"
        exit 1
    fi
    
    cp -r "$MODEL_DIR" "$DEPLOY_DIR/model_cache/"
    
    # Get size
    MODEL_SIZE=$(du -sh "$DEPLOY_DIR/model_cache" | cut -f1)
    print_success "Model files copied (Size: $MODEL_SIZE)"
}

create_setup_script() {
    print_progress "Creating setup script for isolated system..."
    
    cat > "$DEPLOY_DIR/setup_isolated.sh" << 'EOF'
#!/bin/bash
# =============================================================================
# POD OCR - Isolated System Setup
# =============================================================================
# Run this script on the isolated system after extracting the package
# =============================================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         POD OCR - Isolated System Setup                 ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════╝${NC}"
echo

CURRENT_DIR=$(pwd)

echo -e "${YELLOW}[ℹ]${NC} Current directory: $CURRENT_DIR"
echo

# Update model path in run script
echo -e "${BLUE}[→]${NC} Updating configuration..."

# Find the model snapshot directory
MODEL_SNAPSHOT=$(find "$CURRENT_DIR/model_cache" -type d -name "snapshots" | head -1)
if [ -z "$MODEL_SNAPSHOT" ]; then
    echo -e "${RED}[✗]${NC} Model snapshot directory not found"
    exit 1
fi

ACTUAL_MODEL_PATH=$(find "$MODEL_SNAPSHOT" -mindepth 1 -maxdepth 1 -type d | head -1)
if [ -z "$ACTUAL_MODEL_PATH" ]; then
    echo -e "${RED}[✗]${NC} Model path not found"
    exit 1
fi

echo -e "${GREEN}[✓]${NC} Found model at: $ACTUAL_MODEL_PATH"

# Update run_all.sh with actual paths
sed -i "s|PROJECT_DIR=\"/workspace/POD_OCR\"|PROJECT_DIR=\"$CURRENT_DIR\"|g" run_all.sh
sed -i "s|VENV_PATH=\"\$PROJECT_DIR/venv\"|VENV_PATH=\"$CURRENT_DIR/venv\"|g" run_all.sh
sed -i "s|LOG_DIR=\"\$PROJECT_DIR/logs\"|LOG_DIR=\"$CURRENT_DIR/logs\"|g" run_all.sh
sed -i "s|MODEL_PATH=\".*\"|MODEL_PATH=\"$ACTUAL_MODEL_PATH\"|g" run_all.sh

# Update cache path
sed -i "s|export HF_HOME=/workspace/.cache|export HF_HOME=$CURRENT_DIR/model_cache|g" run_all.sh

echo -e "${GREEN}[✓]${NC} Configuration updated"

# Make scripts executable
chmod +x run_all.sh
chmod +x dist_app/my_app/my_app
chmod +x dist_pixtral/my_app/my_app
chmod +x dist_lmdeploy/my_app/my_app

echo -e "${GREEN}[✓]${NC} Executables set"
echo
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  Setup Complete!                         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "${CYAN}To start the services:${NC}"
echo -e "  ${YELLOW}./run_all.sh${NC}"
echo
echo -e "${CYAN}Deployment Details:${NC}"
echo -e "  • Working Directory: ${YELLOW}$CURRENT_DIR${NC}"
echo -e "  • Model Path: ${YELLOW}$ACTUAL_MODEL_PATH${NC}"
echo -e "  • Log Directory: ${YELLOW}$CURRENT_DIR/logs${NC}"
echo
EOF

    chmod +x "$DEPLOY_DIR/setup_isolated.sh"
    print_success "Setup script created"
}

copy_run_script() {
    print_progress "Copying run script..."
    
    if [ ! -f "run_all_new.sh" ]; then
        print_error "run_all_new.sh not found"
        exit 1
    fi
    
    cp run_all_new.sh "$DEPLOY_DIR/run_all.sh"
    chmod +x "$DEPLOY_DIR/run_all.sh"
    
    print_success "Run script copied"
}

create_readme() {
    print_progress "Creating README..."
    
    cat > "$DEPLOY_DIR/README.md" << 'EOF'
# POD OCR - Deployment Package

This package contains everything needed to run POD OCR on an isolated system without internet access.

## Package Contents

```
POD_OCR_DEPLOY/
├── dist_app/           - Main application executable
├── dist_pixtral/       - Pixtral service executable
├── dist_lmdeploy/      - LMDeploy service executable
├── venv/               - Python virtual environment with all dependencies
├── model_cache/        - Pre-downloaded AI models
├── logs/               - Log files directory
├── setup_isolated.sh   - Setup script for isolated system
├── run_all.sh          - Main launcher script
└── README.md           - This file
```

## System Requirements

- **OS**: Ubuntu 20.04+ (or compatible Linux)
- **RAM**: 32GB minimum, 64GB recommended
- **GPU**: NVIDIA GPU with 24GB+ VRAM (for InternVL model)
- **CUDA**: 11.8 or 12.1
- **Disk Space**: 50GB+ free space
- **Python**: 3.8+ (should be pre-installed on Ubuntu)

## Installation on Isolated System

### Step 1: Transfer Package

Transfer the `POD_OCR_DEPLOY.tar.gz` file to your isolated system using:
- USB drive
- Secure file transfer
- Network transfer (before disconnecting internet)

### Step 2: Extract Package

```bash
tar -xzf POD_OCR_DEPLOY.tar.gz
cd POD_OCR_DEPLOY
```

### Step 3: Run Setup

```bash
chmod +x setup_isolated.sh
./setup_isolated.sh
```

This will:
- Configure all paths for your system
- Set executable permissions
- Verify model files

### Step 4: Start Services

```bash
./run_all.sh
```

**Note**: First startup may take 2-3 minutes while models load into GPU memory.

## Service Endpoints

Once started, the following services will be available:

- **Main App**: http://localhost:8080
- **Pixtral Service**: http://localhost:3203
- **LMDeploy API**: http://localhost:23333

## Stopping Services

Press `CTRL+C` in the terminal where `run_all.sh` is running.

Or kill processes manually:
```bash
pkill -f "dist_app/my_app/my_app"
pkill -f "dist_pixtral/my_app/my_app"
pkill -f "dist_lmdeploy/my_app/my_app"
pkill -f "lmdeploy serve"
```

## Logs

Log files are stored in the `logs/` directory:

```bash
# View logs in real-time
tail -f logs/app.log
tail -f logs/pixtral.log
tail -f logs/lmdeploy_exec.log
tail -f logs/lmdeploy_serve.log
```

## Troubleshooting

### Services won't start

1. Check if ports are already in use:
   ```bash
   netstat -tulpn | grep -E '8080|3203|23333'
   ```

2. Check logs for errors:
   ```bash
   cat logs/lmdeploy_serve.log
   ```

3. Verify GPU is available:
   ```bash
   nvidia-smi
   ```

### GPU Out of Memory

If you get OOM errors, you may need to:
- Close other GPU applications
- Use a GPU with more VRAM
- Reduce batch size in application settings

### Permission Denied

Make sure all executables have execute permissions:
```bash
chmod +x run_all.sh
chmod +x dist_*/my_app/my_app
```

## Running Without Internet

This package is designed to work completely offline. All dependencies are included:
- ✅ Python packages (in venv/)
- ✅ AI models (in model_cache/)
- ✅ Application binaries (dist_*/)

You can start/stop services unlimited times without internet.

## Support

For issues or questions, check the logs directory for detailed error messages.

## Version Info

- **Build Date**: $(date)
- **Model**: InternVL2-8B
- **Python Version**: $(python3 --version 2>/dev/null || echo "N/A")
EOF

    print_success "README created"
}

create_archive() {
    print_progress "Creating deployment archive..."
    
    ARCHIVE_NAME="POD_OCR_DEPLOY_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    tar -czf "$ARCHIVE_NAME" "$DEPLOY_DIR/" 2>&1 | while read line; do
        echo -ne "\r    ${YELLOW}Compressing...${NC}"
    done
    echo
    
    ARCHIVE_SIZE=$(du -sh "$ARCHIVE_NAME" | cut -f1)
    print_success "Archive created: $ARCHIVE_NAME (Size: $ARCHIVE_SIZE)"
    
    # Calculate checksums
    print_info "Calculating checksums..."
    md5sum "$ARCHIVE_NAME" > "${ARCHIVE_NAME}.md5"
    sha256sum "$ARCHIVE_NAME" > "${ARCHIVE_NAME}.sha256"
    
    print_success "Checksums created"
}

print_summary() {
    echo
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           Deployment Package Created!                   ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${CYAN}Package Details:${NC}"
    echo -e "  ${YELLOW}•${NC} Archive: ${GREEN}POD_OCR_DEPLOY_*.tar.gz${NC}"
    echo -e "  ${YELLOW}•${NC} Location: ${CYAN}$PROJECT_DIR${NC}"
    echo
    
    TOTAL_SIZE=$(du -sh "$DEPLOY_DIR" | cut -f1)
    echo -e "${CYAN}Package Contents Size:${NC}"
    echo -e "  ${YELLOW}•${NC} Total: ${MAGENTA}$TOTAL_SIZE${NC}"
    
    if [ -d "$DEPLOY_DIR/venv" ]; then
        VENV_SIZE=$(du -sh "$DEPLOY_DIR/venv" | cut -f1)
        echo -e "  ${YELLOW}•${NC} Virtual Env: ${MAGENTA}$VENV_SIZE${NC}"
    fi
    
    if [ -d "$DEPLOY_DIR/model_cache" ]; then
        MODEL_SIZE=$(du -sh "$DEPLOY_DIR/model_cache" | cut -f1)
        echo -e "  ${YELLOW}•${NC} Model Cache: ${MAGENTA}$MODEL_SIZE${NC}"
    fi
    
    echo
    echo -e "${CYAN}Next Steps:${NC}"
    echo -e "  ${YELLOW}1.${NC} Transfer ${GREEN}POD_OCR_DEPLOY_*.tar.gz${NC} to isolated system"
    echo -e "  ${YELLOW}2.${NC} Extract: ${BLUE}tar -xzf POD_OCR_DEPLOY_*.tar.gz${NC}"
    echo -e "  ${YELLOW}3.${NC} Setup: ${BLUE}cd POD_OCR_DEPLOY && ./setup_isolated.sh${NC}"
    echo -e "  ${YELLOW}4.${NC} Run: ${BLUE}./run_all.sh${NC}"
    echo
    echo -e "${GREEN}✓ Package ready for deployment!${NC}"
    echo
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    print_header
    
    check_prerequisites
    echo
    
    download_model
    echo
    
    create_deployment_structure
    echo
    
    copy_executables
    echo
    
    copy_venv
    echo
    
    copy_model
    echo
    
    copy_run_script
    echo
    
    create_setup_script
    echo
    
    create_readme
    echo
    
    create_archive
    echo
    
    print_summary
}

main