#!/usr/bin/env bash
set -euo pipefail
# PM2 launcher for LMDeploy (23333) -> Pixtral (3203) -> API (8080)
# Usage: bash run_all_pm2.sh [--tp N] [--venv /path/to/venv]

TP=${1:-}
if [[ "$TP" == "--tp" ]]; then
  TP_VAL=${2:-1}
  shift 2
else
  TP_VAL=${TP:-1}
fi

VENV_PATH=${1:-}
if [[ "$VENV_PATH" == "--venv" ]]; then
  VENV=${2:-"/workspace/POD_OCR/venv"}
  shift 2
else
  VENV="/workspace/POD_OCR/venv"
fi

ROOT_DIR="/workspace/POD_OCR"
cd "$ROOT_DIR"

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Installing Python3..."
    if command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y python3 python3-pip python3-venv
    elif command -v yum &> /dev/null; then
        yum install -y python3 python3-pip
    else
        echo "ERROR: Cannot install Python3. Please install it manually."
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [[ ! -d "$VENV" ]]; then
    echo "Creating virtual environment at $VENV..."
    python3 -m venv "$VENV"
    echo "Virtual environment created successfully."
fi

# Activate virtual environment
if [[ -f "$VENV/bin/activate" ]]; then
  source "$VENV/bin/activate"
  echo "Virtual environment activated."
else
  echo "ERROR: Virtual environment activation script not found at $VENV/bin/activate"
  exit 1
fi

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "PM2 not found. Installing PM2..."
    if command -v npm &> /dev/null; then
        npm install -g pm2
    elif command -v apt-get &> /dev/null; then
        echo "Installing Node.js and npm first..."
        apt-get update && apt-get install -y nodejs npm
        npm install -g pm2
    elif command -v yum &> /dev/null; then
        echo "Installing Node.js and npm first..."
        yum install -y nodejs npm
        npm install -g pm2
    else
        echo "ERROR: Cannot install PM2. Please install Node.js and npm first."
        echo "Run: apt-get update && apt-get install -y nodejs npm"
        echo "Then: npm install -g pm2"
        exit 1
    fi
fi

echo "Installing dependencies..."
if [[ ! -f "requirements.txt" ]]; then
    echo "ERROR: requirements.txt not found in $ROOT_DIR"
    exit 1
fi
pip install -r requirements.txt

pip install hf_transfer nvidia-ml-py

# Adjust tp by editing lmdeploy_app.py dynamically
echo "Configuring LMDeploy with tp=$TP_VAL..."
if [[ -f "lmdeploy_app.py" ]]; then
    sed -i -E "s/tp=\s*[0-9]+/tp=${TP_VAL}/" lmdeploy_app.py || true
else
    echo "WARNING: lmdeploy_app.py not found, skipping tp configuration"
fi

# Stop existing PM2 processes if running
echo "Stopping any existing PM2 processes..."
pm2 delete pod-ocr-lmdeploy 2>/dev/null || true
pm2 delete pod-ocr-pixtral 2>/dev/null || true
pm2 delete pod-ocr-api 2>/dev/null || true

# Create PM2 ecosystem file
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: python not found or not executable at ${VENV}/bin/python"
  echo "Contents of ${VENV}/bin/:"
  ls -la "${VENV}/bin" || true
  exit 1
fi

# Create PM2 ecosystem file (use a shell wrapper that activates venv before running python)
cat > ecosystem.config.js <<EOF
module.exports = {
  apps: [
    {
      name: 'pod-ocr-lmdeploy',
      // use /bin/bash to activate venv then exec python so environment matches interactive run
      script: '/bin/bash',
      args: '-lc "source ${VENV}/bin/activate && exec python lmdeploy_app.py"',
      cwd: '${ROOT_DIR}',
      interpreter: 'none',
      out_file: '${ROOT_DIR}/logs_lmdeploy.txt',
      error_file: '${ROOT_DIR}/logs_lmdeploy.txt',
      combine_logs: true,
      env: {
        VIRTUAL_ENV: '${VENV}',
        PATH: '${VENV}/bin:' + process.env.PATH
      }
    },
    {
      name: 'pod-ocr-pixtral',
      script: '/bin/bash',
      args: '-lc "source ${VENV}/bin/activate && exec python pixtral.py"',
      cwd: '${ROOT_DIR}',
      interpreter: 'none',
      out_file: '${ROOT_DIR}/logs_pixtral.txt',
      error_file: '${ROOT_DIR}/logs_pixtral.txt',
      combine_logs: true,
      env: {
        VIRTUAL_ENV: '${VENV}',
        PATH: '${VENV}/bin:' + process.env.PATH
      }
    },
    {
      name: 'pod-ocr-api',
      script: '/bin/bash',
      args: '-lc "source ${VENV}/bin/activate && exec python app.py"',
      cwd: '${ROOT_DIR}',
      interpreter: 'none',
      out_file: '${ROOT_DIR}/logs_api.txt',
      error_file: '${ROOT_DIR}/logs_api.txt',
      combine_logs: true,
      env: {
        VIRTUAL_ENV: '${VENV}',
        PATH: '${VENV}/bin:' + process.env.PATH
      }
    }
  ]
};
EOF

echo "Starting LMDeploy with PM2..."
pm2 start ecosystem.config.js --only pod-ocr-lmdeploy

echo -n "Waiting for LMDeploy to be ready"
for i in {1..120}; do
  if curl -s http://localhost:23333/v1/models >/dev/null 2>&1; then
    echo " - OK"
    break
  fi
  echo -n "."
  sleep 2
done

if ! curl -s http://localhost:23333/v1/models >/dev/null 2>&1; then
  echo -e "\nERROR: LMDeploy not responding on :23333. Check logs with: pm2 logs pod-ocr-lmdeploy"
  pm2 delete pod-ocr-lmdeploy
  exit 1
fi

echo "Starting Pixtral with PM2..."
pm2 start ecosystem.config.js --only pod-ocr-pixtral

echo -n "Waiting for Pixtral to be ready"
for i in {1..60}; do
  if curl -s http://localhost:3203/docs >/dev/null 2>&1; then
    echo " - OK"
    break
  fi
  echo -n "."
  sleep 1
done

if ! curl -s http://localhost:3203/docs >/dev/null 2>&1; then
  echo -e "\nERROR: Pixtral not responding on :3203. Check logs with: pm2 logs pod-ocr-pixtral"
  pm2 delete pod-ocr-pixtral pod-ocr-lmdeploy
  exit 1
fi

echo "Starting API with PM2..."
pm2 start ecosystem.config.js --only pod-ocr-api

echo -n "Waiting for API to be ready"
for i in {1..60}; do
  if curl -s http://localhost:8080/docs >/dev/null 2>&1; then
    echo " - OK"
    break
  fi
  echo -n "."
  sleep 1
done

if ! curl -s http://localhost:8080/docs >/dev/null 2>&1; then
  echo -e "\nERROR: API not responding on :8080. Check logs with: pm2 logs pod-ocr-api"
  pm2 delete pod-ocr-api pod-ocr-pixtral pod-ocr-lmdeploy
  exit 1
fi

echo ""
echo "✓ All services are up and running!"
echo ""
echo "PM2 Process Status:"
pm2 list
echo ""
echo "Useful PM2 commands:"
echo "  pm2 logs                    - View all logs"
echo "  pm2 logs pod-ocr-lmdeploy   - View LMDeploy logs"
echo "  pm2 logs pod-ocr-pixtral    - View Pixtral logs"
echo "  pm2 logs pod-ocr-api        - View API logs"
echo "  pm2 monit                   - Monitor resources"
echo "  pm2 restart all             - Restart all services"
echo "  pm2 stop all                - Stop all services"
echo "  pm2 delete all              - Remove all services"
echo ""
echo "Log files: logs_lmdeploy.txt, logs_pixtral.txt, logs_api.txt"

# Save PM2 process list to resurrect on reboot
pm2 save