#!/usr/bin/env python3
import subprocess
import time
import requests
import sys
import os
import signal

# Configurable paths
ROOT_DIR = "/workspace/POD_OCR"
VENV_PATH = os.path.join(ROOT_DIR, ".venv", "bin", "python")

# Entry points for each service
LMDEPLOY_CMD = [VENV_PATH, "lmdeploy_app.py"]
PIXTRAL_CMD  = [VENV_PATH, "pixtral.py"]
API_CMD      = [VENV_PATH, "app.py"]

# Health check URLs
CHECKS = {
    "LMDeploy": ("http://localhost:23333/v1/models", LMDEPLOY_CMD),
    "Pixtral":  ("http://localhost:3203/docs", PIXTRAL_CMD),
    "API":      ("http://localhost:8080/docs", API_CMD),
}

processes = {}

def wait_for(name, url, timeout=120):
    """Wait until a service responds to HTTP GET."""
    print(f"Waiting for {name} at {url} ...")
    for i in range(timeout):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code < 500:
                print(f"✅ {name} is up!")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"❌ {name} not responding after {timeout} sec")

def start_service(name, cmd, url):
    """Start a service in background and wait for it."""
    print(f"▶️ Starting {name} with command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=ROOT_DIR)
    processes[name] = proc
    wait_for(name, url)

def stop_all():
    """Terminate all running processes."""
    print("\n🛑 Stopping all services...")
    for name, proc in processes.items():
        try:
            proc.terminate()
            proc.wait(timeout=5)
            print(f"✅ {name} stopped.")
        except Exception:
            print(f"⚠️ Force killing {name}...")
            proc.kill()

def main():
    try:
        # Ensure deps are installed (optional)
        print("📦 Installing dependencies...")
        subprocess.run([VENV_PATH, "-m", "pip", "install", "-r", "requirements.txt"], cwd=ROOT_DIR)

        # Start services in order
        for name, (url, cmd) in CHECKS.items():
            start_service(name, cmd, url)

        print("\n🚀 All services are up and running!")
        print("   LMDeploy -> :23333")
        print("   Pixtral  -> :3203")
        print("   API      -> :8080")
        print("\nPress Ctrl+C to stop everything...")

        # Keep alive
        while True:
            time.sleep(5)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        stop_all()

if __name__ == "__main__":
    main()
