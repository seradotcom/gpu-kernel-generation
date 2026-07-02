# Semantic GPU Kernel Generator — LLM-MLIR Compiler

> A hybrid pipeline for generating semantically correct Triton/MLIR GPU kernels using Large Language Models and Constrained Decoding.

Traditional LLM pipelines for GPU kernel generation tend to fail silently — producing code that looks valid but contains syntax errors or hallucinated operations. This project takes a different approach through a **Semantically Constrained Generation** architecture. Rather than generating raw Triton or Python code directly, the pipeline drives an LLM (Qwen3.5-9B-AWQ) to output a strict JSON Abstract Syntax Tree (AST). That AST is then translated locally into MLIR (Multi-Level Intermediate Representation) dialects, where static mathematical and semantic validation occurs before any compilation step.

---

## System Architecture

The pipeline runs across a decoupled local/cloud environment with three main components:

1. **Local Orchestrator (Python)** — Handles prompt formulation, JSON schema enforcement via Pydantic, AST-to-MLIR translation, and MLOps metric tracking.
2. **Remote Inference Engine (L4 GPU)** — Serves the `QuantTrio/Qwen3.5-9B-AWQ` model (4-bit AWQ) with **vLLM** on a single NVIDIA L4, behind a FastAPI endpoint exposed through an Ngrok tunnel.
3. **Constrained Decoding (XGrammar)** — The orchestrator sends a Pydantic JSON Schema to the remote endpoint, which compiles it with `xgrammar` to constrain the model's output logits, preventing malformed JSON or illegal MLIR operations at the generation level.

---

## 1. Initial Setup & Installation

### 1.1 — Clone the Repository and Create the Environment

Using a virtual environment is required to isolate project dependencies and avoid conflicts with system packages.

```bash
# Clone the repository
git clone https://github.com/seradotcom/gpu-kernel-generation.git
cd llm-mlir-compiler

# Create a Python 3.10 virtual environment
python3.10 -m venv .venv

# Activate the virtual environment

# Linux / macOS:
source .venv/bin/activate

# Windows:
# .venv\Scripts\activate
```

### 1.2 — Install Python Dependencies

With the virtual environment active, install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.3 — Configure MLIR Python Bindings *(Critical)*

The local orchestrator depends on native LLVM/MLIR bindings for semantic validation.

👉 Refer to [INSTALL_MLIR.md](./INSTALL_MLIR.md) for the complete guide on building LLVM/MLIR from source or using the provided Docker containers. This step is required before running the pipeline.

---

## ☁️ 2. Remote Model Configuration (vLLM on L4)

The LLM runs remotely (single NVIDIA L4 GPU) and connects back to the local orchestrator via API.

1. Open the official serving notebook `resources/tests_ptx.ipynb` (end-to-end pipeline) or `resources/tests_raw_llm.ipynb` (zero-shot baseline).
2. Provision a single **NVIDIA L4** GPU and enable internet access.
3. Run all cells to load `QuantTrio/Qwen3.5-9B-AWQ` with vLLM, initialize FastAPI, and start the Ngrok tunnel.
4. Copy the generated **Ngrok Public URL** from the cell output — it will follow the pattern `https://<random-id>.ngrok-free.dev`.

---

## 🔐 3. Environment Variables

Create a `.env` file in the project root and fill in the values below:

```env
# Remote model endpoint (paste the Ngrok URL from the L4 serving notebook here)
USE_REMOTE_MODEL=1        # Set to 1 for remote inference, 0 for local Ollama
LLM_API_URL="https://your-ngrok-url.ngrok-free.dev"

# MLOps tracking (Weights & Biases)
WANDB_API_KEY=your_wandb_api_key_here

# External fallback APIs (optional)
NVIDIA_API_KEY=your_nvapi_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 🚀 4. Running the Pipeline

Once MLIR is configured, the virtual environment is active, and the remote L4 server is running:

```bash
# Optional: authenticate with Weights & Biases for live MLOps tracking
wandb login

# Start the orchestrator
python main.py
```

Each run logs metrics to your W&B dashboard, including prompt latency, schema validation success rates, full JSON outputs, and the resulting MLIR AST representations.

---

## 📊 5. Entry Points & Scripts Guide

This repository contains multiple execution paths depending on whether you are running our custom internal benchmarks, the official TritonBench evaluation, or ablation studies. 

### A. Core Pipeline & Manual Testing
* **`main.py`**
  The simplest entry point. It runs a single end-to-end prompt through the LLM -> AST -> MLIR translation cycle. Use this to verify that your environment (MLIR bindings, W&B, Remote Server) is working correctly.
* **`run_benchmarks.py`**
  The script used to run our **Custom Manual Benchmarks**. It processes our internal datasets (`benchmark_prompts_EASY/MED/HARD.json`), compiles the instructions through our full semantic pipeline into PTX, and evaluates the mathematical correctness against PyTorch references locally.

### B. Official TritonBench Evaluation
To evaluate our pipeline against the industry-standard TritonBench-T track, we decouple the generation from the evaluation.
* **`1_generate_ptx.py`**
  Consumes the official TritonBench prompts and runs our semantic compiler pipeline (LLM -> MLIR -> PTX) to generate the raw assembly, saving it to disk.
* **`2_build_eval_wrappers.py`**
  Packages the generated `.ptx` files into `CuPy` Python wrappers (`predictions.jsonl`) so they can be injected into the TritonBench official evaluator.
*(For detailed instructions on running these, see [`TRITONBENCH_EVALUATION_GUIDE.md`](./TRITONBENCH_EVALUATION_GUIDE.md))*

### C. Baselines & Ablation Studies (Proving the Architecture)
These files bypass our MLIR compiler and ask the LLM to generate raw code directly. They are used to measure the failure rate of standard LLM generation and prove the necessity of our AST/MLIR architecture.
* **`baseline_ablation.py`**
  Runs the LLM directly on our custom manual benchmarks (EASY/MED/HARD) to generate raw Triton code, evaluating how often it hallucinates or produces syntax errors.
* **`resources/tests_raw_llm.ipynb`**
  A Jupyter Notebook that runs a zero-shot raw LLM baseline directly on the official TritonBench dataset.
* **`resources/tests_ptx.ipynb`**
  An interactive notebook for debugging and inspecting the PTX compilation process with TritonBench.

---

## 📂 6. Project Structure

```
llm-mlir-compiler/
├── core/
│   ├── config.py           # Global environment variables and thresholds
│   ├── llm_client.py       # API router (Remote, Gemini, Kimi, Ollama)
│   ├── mlir_translator.py  # Converts JSON AST to MLIR dialects
│   ├── mlops_tracker.py    # Weights & Biases integration
│   ├── schemas.py          # Pydantic schemas that drive XGrammar constraints
│   └── triton_backend.py   # Lowering logic: TTIR → TTGIR → PTX (NVIDIA)
├── main.py                 # Entry point and orchestration loop
├── requirements.txt        # Python dependencies
├── INSTALL_MLIR.md         # MLIR/LLVM setup guide
├── TRITONBENCH_EVALUATION_GUIDE.md  # Official TritonBench-T execution guide
└── .env                    # Environment variables (git-ignored)
```

---

## 📚 7. References

- [TritonBench: Evaluating Large Language Models for Generating Triton Operators](https://arxiv.org/abs/2502.14752) — Official Paper
- [TritonBench Official Repository](https://github.com/thunlp/TritonBench/) — thunlp/TritonBench
- [MLIR Documentation](https://mlir.llvm.org/docs/) — Multi-Level Intermediate Representation
- [Triton MLIR Dialects](https://triton-lang.org/main/dialects/dialects.html)
- [XGrammar — HuggingFace](https://huggingface.co/docs/text-generation-inference/conceptual/guidance)
- [Weights & Biases MLOps](https://docs.wandb.ai/)