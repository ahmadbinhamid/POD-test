import multiprocessing
import sys
import os
 
# CRITICAL: Must be at the very top, before other imports
if getattr(sys, 'frozen', False):
    # Running in PyInstaller bundle
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
 
# ============================================================
# PATCH: Allow daemon processes to have children
# This must be AFTER freeze_support but BEFORE lmdeploy import
# ============================================================
import multiprocessing.process
 
class NonDaemonicProcess(multiprocessing.Process):
    """Process that allows daemon processes to spawn children"""
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, value):
        pass
 
class NonDaemonicContext(type(multiprocessing.get_context())):
    Process = NonDaemonicProcess
 
# Monkey-patch the multiprocessing module
multiprocessing.process.BaseProcess._Popen = NonDaemonicContext().Process._Popen
 
print("✓ Multiprocessing patch applied successfully")
# ============================================================
 
# Now import lmdeploy and other modules
import PIL
from lmdeploy import api
from lmdeploy import PytorchEngineConfig
import torch
import glob
 
def find_local_model(cache_dir="/workspace/.cache", model_name="InternVL2-8B"):
    """Find the model in HuggingFace cache - DO NOT DOWNLOAD"""
    search_path = f"{cache_dir}/huggingface/models--OpenGVLab--{model_name}/snapshots"
    print(f"Searching for model in: {search_path}")
    if os.path.exists(search_path):
        snapshots = [d for d in glob.glob(f"{search_path}/*") if os.path.isdir(d)]
        if snapshots:
            latest = max(snapshots, key=os.path.getmtime)
            print(f"✓ Found cached model at: {latest}")
            return latest
    print(f"✗ ERROR: Model not found in cache!")
    return None
 
def main():
    print("=================================")
    print("LMDeploy Application Starting")
    print("=================================")
    # Check available GPUs
    try:
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
    except Exception as e:
        print(f"Warning: Could not detect GPUs: {e}")
        gpu_count = 0
    # Configure for available GPUs
    if gpu_count == 0:
        print("WARNING: No GPUs detected. Running on CPU (will be slow).")
        backend_config = PytorchEngineConfig(
            tp=1,
            max_batch_size=1,
            cache_max_entry_count=0.5,
        )
    elif gpu_count == 1:
        print("Using 1 GPU")
        backend_config = PytorchEngineConfig(
            tp=1,
            max_batch_size=4,
            cache_max_entry_count=0.8,
        )
    else:
        print(f"Using {min(gpu_count, 2)} GPUs (tensor parallel)")
        backend_config = PytorchEngineConfig(
            tp=min(gpu_count, 2),
            max_batch_size=4,
            cache_max_entry_count=0.8,
        )
    # Get cache directory from environment
    cache_dir = os.environ.get('HF_HOME', '/workspace/.cache')
    print(f"Using cache directory: {cache_dir}")
    # Find local model
    local_model = find_local_model(cache_dir, "InternVL2-8B")
    if not local_model:
        raise RuntimeError("Model not found in cache! Please ensure the model is downloaded.")
    model_path = local_model
    print(f"✓ Using LOCAL cached model (no download)")
    print(f"  Path: {model_path}")
    print(f"Backend config: tp={backend_config.tp}, max_batch={backend_config.max_batch_size}")
    print("\n=================================")
    print("Starting LMDeploy server on 0.0.0.0:23333...")
    print("=================================\n")
    try:
        # Start the server
        api.serve(
            model_name="InternVL2-8B",
            model_path=model_path,
            server_name="0.0.0.0",
            server_port=23333,
            backend_config=backend_config
        )
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped by user")
    except Exception as e:
        print(f"\n✗ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
 
# CRITICAL: Must wrap main code in this guard
if __name__ == '__main__':
    multiprocessing.freeze_support()  # Required for Windows/PyInstaller
    main()