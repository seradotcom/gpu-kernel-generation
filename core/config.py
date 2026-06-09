import os
import sys
import tarfile
import glob
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

# 3. Check sys.path / PYTHONPATH for pre-installed bindings (e.g. Google Drive, custom envs)
if MLIR_BINDINGS_PATH is None:
    for p in sys.path:
        if p.endswith("mlir_core") and os.path.exists(p):
            MLIR_BINDINGS_PATH = p
            print(f"[Info] Using MLIR bindings from sys.path: {p}")
            break

# 4. Common cloud / mounted-drive fallbacks (dynamic search using glob)
if MLIR_BINDINGS_PATH is None:
    # Google Colab — search under mounted Drive for any mlir_core directory
    DRIVE_BASE = "/content/drive/MyDrive/llvm-install"
    if os.path.exists(DRIVE_BASE):
        mlir_core_paths = glob.glob(f"{DRIVE_BASE}/**/mlir_core", recursive=True)
        if mlir_core_paths:
            p = mlir_core_paths[0]
            sys.path.append(p)
            MLIR_BINDINGS_PATH = p
            print(f"[Info] Using MLIR bindings from Colab Drive (dynamic search): {p}")

    # Kaggle — static fallback path
    if MLIR_BINDINGS_PATH is None:
        KAGGLE_PATH = "/kaggle/input/llvm-install/python_packages/mlir_core"
        if os.path.exists(KAGGLE_PATH):
            sys.path.append(KAGGLE_PATH)
            MLIR_BINDINGS_PATH = KAGGLE_PATH
            print(f"[Info] Using MLIR bindings from Kaggle fallback: {KAGGLE_PATH}")

if MLIR_BINDINGS_PATH is None:
    print(f"[Warning] No MLIR bindings found. Tried tarball ({TARBALL_PATH}), local build ({LOCAL_MLIR_PATH}), sys.path, and cloud fallbacks.")

# --- API KEYS & ADC SETUP ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")

# Google Cloud Vertex AI Settings
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "TU_ID_DE_PROYECTO")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# --- MODEL CONFIGURATION ---
MODEL_KIMI = "meta/llama-3.1-405b-instruct" # Proxy model in Nvidia API
MODEL_GEMINI = "gemini-2.0-flash-lite-001"  # Cheapest Gemini model, highest quota
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

# --- GROQ CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_GROQ = "llama-3.1-8b-instant"      # Cheaper/faster, good for code
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# --- VLLM (Remote / Ngrok) CONFIGURATION ---
# URL points to the custom FastAPI vLLM server (AsyncLLMEngine + /generate endpoint)
# Not the OpenAI-compatible /v1/chat/completions server.
VLLM_URL = os.getenv("VLLM_URL", "https://efficient-lethargy-haggler.ngrok-free.dev")
VLLM_MODEL = os.getenv("VLLM_MODEL", "QuantTrio/Qwen3.5-9B-AWQ")

# --- REMOTE MODEL CONFIGURATION ---
REMOTE_PROMPT_TEMPLATE = "chatml" # Use "chatml" for Qwen, "gemma" for Gemma

