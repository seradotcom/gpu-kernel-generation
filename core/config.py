import os
import sys
import tarfile
from dotenv import load_dotenv

# Load variables from .env file if it exists
load_dotenv()

# --- NATIVE MLIR ENVIRONMENT SETUP ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. Try tarball first (GitHub Actions artifact)
TARBALL_PATH = os.path.join(PROJECT_ROOT, "mlir-pybindings.tar.gz")
EXTRACT_DIR = os.path.join(PROJECT_ROOT, ".mlir_bindings")
TARBALL_MLIR_PATH = os.path.join(EXTRACT_DIR, "llvm-install", "python_packages", "mlir_core")

MLIR_BINDINGS_PATH = None

if os.path.exists(TARBALL_MLIR_PATH):
    sys.path.append(TARBALL_MLIR_PATH)
    MLIR_BINDINGS_PATH = TARBALL_MLIR_PATH
    print(f"[Info] Using extracted MLIR bindings from tarball: {TARBALL_MLIR_PATH}")
elif os.path.exists(TARBALL_PATH):
    print(f"[Info] Extracting MLIR bindings tarball to {EXTRACT_DIR}...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with tarfile.open(TARBALL_PATH, "r:gz") as tar:
        tar.extractall(path=EXTRACT_DIR)
    if os.path.exists(TARBALL_MLIR_PATH):
        sys.path.append(TARBALL_MLIR_PATH)
        MLIR_BINDINGS_PATH = TARBALL_MLIR_PATH
        print(f"[Info] MLIR bindings extracted and ready: {TARBALL_MLIR_PATH}")
    else:
        print(f"[Warning] Tarball extraction did not produce expected path: {TARBALL_MLIR_PATH}")

# 2. Fallback to local llvm-project build
if MLIR_BINDINGS_PATH is None:
    LLVM_PROJECT_DIR = os.path.join(PROJECT_ROOT, "llvm-project")
    LOCAL_MLIR_PATH = os.path.join(LLVM_PROJECT_DIR, "build", "tools", "mlir", "python_packages", "mlir_core")
    if os.path.exists(LOCAL_MLIR_PATH):
        sys.path.append(LOCAL_MLIR_PATH)
        MLIR_BINDINGS_PATH = LOCAL_MLIR_PATH
        print(f"[Info] Using local MLIR build: {LOCAL_MLIR_PATH}")
    else:
        print(f"[Warning] No MLIR bindings found. Tried tarball ({TARBALL_PATH}) and local build ({LOCAL_MLIR_PATH}).")

# --- API KEYS & ADC SETUP ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")

# Google Cloud Vertex AI Settings
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "TU_ID_DE_PROYECTO")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# --- MODEL CONFIGURATION ---
MODEL_KIMI = "meta/llama-3.1-405b-instruct" # Proxy model in Nvidia API
MODEL_GEMINI = "gemini-2.5-flash"      # Gemini model
MODEL_OLLAMA = "gemma4:e2b"                 # Local Ollama model

OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"

USE_REMOTE_MODEL = os.getenv("USE_REMOTE_MODEL", "0") == "1"
REMOTE_MODEL_URL = os.getenv("GEMMA_API_URL", "https://unsubtly-dash-economy.ngrok-free.dev")

# Strict parameters for code generation
GENERATION_PARAMS = {
    "max_tokens": 8192,
    "temperature": 0.1, 
    "top_p": 0.9,
    "repetition_penalty": 1.0,
}

# --- MLOPS CONFIGURATION ---
WANDB_PROJECT_NAME = "llm-mlir-compiler"
WANDB_ENTITY = None  # Automatic if configured in the environment

# --- REMOTE MODEL CONFIGURATION ---
REMOTE_PROMPT_TEMPLATE = "chatml" # Use "chatml" for Qwen, "gemma" for Gemma

