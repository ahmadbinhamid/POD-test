#!/bin/bash

# Set environment variables
export HF_HOME=/workspace/.cache

echo "==================================="
echo "Starting All Applications"
echo "==================================="

# Check if executables exist
if [ ! -f "./dist_app/my_app/my_app" ]; then
    echo "Error: dist/my_app/my_app not found. Run build_all.sh first."
    exit 1
fi

if [ ! -f "./dist_pixtral/my_app/my_app" ]; then
    echo "Error: dist_pixtral/my_app/my_app not found. Run build_all.sh first."
    exit 1
fi

if [ ! -f "./dist_lmdeploy/my_app/my_app" ]; then
    echo "Error: dist_lmdeploy/my_app/my_app not found. Run build_all.sh first."
    exit 1
fi

# Start lmdeploy first (since pixtral depends on it)
echo "Starting lmdeploy_app on port 23333..."
./dist_lmdeploy/my_app/my_app &
LMDEPLOY_PID=$!

# Wait a bit for lmdeploy to start
echo "Waiting for lmdeploy to initialize..."
sleep 5

# Start pixtral
echo "Starting pixtral_app on port 3203..."
./dist_pixtral/my_app/my_app &
PIXTRAL_PID=$!

# Wait a bit for pixtral to start
sleep 2

# Start main app
echo "Starting my_app on port 8080..."
./dist_app/my_app/my_app &
APP_PID=$!

echo ""
echo "==================================="
echo "All applications started!"
echo "==================================="
echo "PIDs:"
echo "  - lmdeploy_app: $LMDEPLOY_PID"
echo "  - pixtral_app: $PIXTRAL_PID"
echo "  - my_app: $APP_PID"
echo ""
echo "Press CTRL+C to stop all applications"
echo "==================================="

# Function to handle cleanup
cleanup() {
    echo ""
    echo "Stopping all applications..."
    kill $APP_PID 2>/dev/null
    kill $PIXTRAL_PID 2>/dev/null
    kill $LMDEPLOY_PID 2>/dev/null
    echo "All applications stopped."
    exit 0
}

# Set up trap for CTRL+C
trap cleanup SIGINT SIGTERM

# Wait for all background processes
wait $APP_PID $PIXTRAL_PID $LMDEPLOY_PID