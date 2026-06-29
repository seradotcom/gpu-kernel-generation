# Official TritonBench-T Evaluation Guide

This document outlines the architecture, rationale, and execution process for evaluating the LLM-to-MLIR kernel generation model using the official **TritonBench-T** industry standard.

To accommodate different research needs, the evaluation can be performed using two different strategies:
- **Strategy A: Native Triton Python Evaluation** (Uses a second LLM pass to build Python host wrappers)
- **Strategy B: Direct PTX Assembly Injection** (Bypasses Python compilation, evaluating pure generated assembly via CuPy)

---

## Prerequisites: Setting up TritonBench

To evaluate the generated code, the official TritonBench repository must be cloned and placed in the `resources/` directory.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TritonBench-authors/TritonBench.git resources/tritonbench_repo
   ```

2. **Rebuild the Missing Statistics File:**
   The official TritonBench repo forgot to ship a required JSON mapping file for the `eval_T` scripts. We provide a custom script `build_t_stats.py` in the `resources/` folder to reconstruct it.
   ```bash
   cp resources/build_t_stats.py resources/tritonbench_repo/
   cd resources/tritonbench_repo
   python build_t_stats.py
   cd ../..
   ```

3. **Fix the Hardcoded Bugs in the Official Eval Scripts:**
   The TritonBench authors left bugs in their evaluation scripts that will cause crashes. Open `resources/tritonbench_repo/EVAL/eval_T/0_call_acc.py` and modify it:
   - **Line 9:** Change `py_folder = "data/TritonBench_G_v1/"` to `"data/TritonBench_T_v1/"` (It points to the G folder instead of the T folder by mistake).
   - **Line 10:** Change `py_interpreter = "/home/lijianling/.../python"` to your virtual environment's python path (e.g. `python3`).
   Do the same for `EVAL/eval_T/1_exe_acc.py` (Line 8 `py_interpreter`).

---

## Strategy A: Native Triton Python Evaluation (`run_tritonbench_t.py`)

In this strategy, we generate the `@triton.jit` kernel, and then use a *second LLM pass* to build a native Python host wrapper that matches the signature TritonBench expects. 

1. **Run the Generation Pipeline:**
   We use the `run_tritonbench_t.py` script provided in `resources/`. You can copy it to the root directory to run it, or run it pointing to the right paths:
   ```bash
   cp resources/run_tritonbench_t.py .
   python run_tritonbench_t.py \
       --alpaca resources/tritonbench_repo/data/TritonBench_T_simp_alpac_v1.json \
       --out predictions.jsonl \
       --model remote \
       --max_retries 3
   ```
   *(Use `--limit 5` to smoke-test before running the full 166 batch).*

2. **Evaluate the Results:**
   Once `predictions.jsonl` is generated, use the official benchmark tools.
   ```bash
   cd resources/tritonbench_repo
   
   # 1. Call Accuracy (Does it run without crashing?)
   python EVAL/eval_T/0_call_acc.py --source ../../predictions.jsonl --target temp_T --GPUs [0]
   
   # 2. Execution Accuracy (Does the math match the gold reference?)
   python EVAL/eval_T/1_exe_acc.py --folder temp_T --GPUs [0]
   ```

---

## Strategy B: Direct PTX Assembly Injection (Decoupled)

This strategy evaluates the **raw PTX assembly** generated directly by our MLIR compiler, completely bypassing the Python Triton compiler to preserve the mathematical purity of our generation. 

1. **Install CuPy Driver Dependency:**
   Since we will launch direct assembly from Python, install:
   ```bash
   pip install cupy-cuda12x
   # (Replace "12x" with your target environment's CUDA version)
   ```

2. **Phase 1: Model Compilation (`1_generate_ptx.py`)**
   Generates raw `.ptx` assembly files for each successful instruction without building host wrappers during the LLM cycle.
   ```bash
   python 1_generate_ptx.py \
       --alpaca resources/tritonbench_repo/data/TritonBench_T_simp_alpac_v1.json \
       --out_dir benchmark_ptx_output \
       --model remote
   ```

3. **Phase 2: Evaluation Injection (`2_build_eval_wrappers.py`)**
   Scans the successfully compiled `.ptx` files and structurally maps the AST pointers (`%arg0`, etc.) to PyTorch tensor signatures offline. Wraps the PTX in `CuPy` (`cp.RawModule`).
   ```bash
   python 2_build_eval_wrappers.py \
       --ptx_dir benchmark_ptx_output \
       --out predictions.jsonl
   ```

4. **Phase 3: Execution on the Official Benchmark**
   Just like in Strategy A, run the evaluation using `predictions.jsonl`:
   ```bash
   cd resources/tritonbench_repo
   python3 EVAL/eval_T/0_call_acc.py --source ../../predictions.jsonl --target temp_T --GPUs [0]
   python3 EVAL/eval_T/1_exe_acc.py --folder temp_T --GPUs [0]
   ```

---

## Understanding Execution Accuracy (Important Caveat)
The script `1_exe_acc.py` decides correctness by **string-comparing stdout** of the two scripts. Several gold harnesses end without printing anything, so their stdout is empty and the comparison passes trivially. For a true numerical signal, append a deterministic `print(test_results)` to the gold test blocks, or add your own `torch.allclose` check. Call accuracy (`0_call_acc.py`) remains the most reliable "it ran" signal out of the box.
