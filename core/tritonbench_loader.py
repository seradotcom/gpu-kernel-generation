"""
THUNLP TritonBench Loader and Prompt Abstractor.

Downloads a small subset of the THUNLP TritonBench-G dataset,
abstracts the verbose TritonBench instructions into MLIR-friendly prompts,
and maps them to the corresponding test files.
"""

import json
import os
import re
import subprocess
import textwrap

TRITONBENCH_REPO = "https://github.com/thunlp/TritonBench.git"
TRITONBENCH_DIR = "thunlp_tritonbench"
DATA_JSON = "data/TritonBench_G_simp_alpac_v1.json"
TEST_DIR = "data/TritonBench_G_v1"


def ensure_tritonbench_cloned():
    """Clone THUNLP TritonBench if not already present."""
    if os.path.exists(TRITONBENCH_DIR):
        return
    print(f"[TritonBench] Cloning {TRITONBENCH_REPO}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", TRITONBENCH_REPO, TRITONBENCH_DIR],
        check=True,
        capture_output=True,
    )
    print(f"[TritonBench] Cloned to ./{TRITONBENCH_DIR}")


def load_json_dataset() -> list:
    """Load the TritonBench-G simple instruction dataset."""
    path = os.path.join(TRITONBENCH_DIR, DATA_JSON)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_filename_from_output(output_code: str) -> str:
    """Heuristic: find the filename from the output code's structure or class names."""
    # Try to find class/function names that might map to files
    # This is a best-effort heuristic
    return ""


def _abstract_prompt(instruction: str, filename_hint: str = "") -> str:
    """
    Convert a verbose TritonBench instruction into a concise,
    math-focused prompt suitable for MLIR generation.
    """
    # Remove the boilerplate "You are a expert in writing Triton operators..."
    instruction = re.sub(
        r"You are a expert in writing Triton operators.*?according following instruction\.\s*",
        "",
        instruction,
        flags=re.DOTALL | re.IGNORECASE,
    )
    instruction = instruction.strip()

    # Remove specific Triton jargon that would confuse the MLIR phase
    replacements = [
        (r"@triton\.jit", ""),
        (r"tl\.program_id", ""),
        (r"tl\.arange", ""),
        (r"tl\.load", ""),
        (r"tl\.store", ""),
        (r"tl\.dot", ""),
        (r"tl\.zeros", ""),
        (r"tl\.where", ""),
        (r"tl\.sigmoid", ""),
        (r"tl\.exp", ""),
        (r"tl\.log", ""),
        (r"tl\.maximum", ""),
        (r"tl\.sum", ""),
        (r"tl\.full", ""),
        (r"BLOCK_SIZE", ""),
        (r"num_warps", ""),
        (r"num_stages", ""),
        (r"`[^`]+`", lambda m: m.group(0).replace("`", "")),  # Remove backticks but keep content
    ]
    for pattern, repl in replacements:
        instruction = re.sub(pattern, repl, instruction, flags=re.IGNORECASE)

    # Collapse multiple newlines
    instruction = re.sub(r"\n{2,}", "\n", instruction)
    instruction = instruction.strip()

    # Truncate if too long for the MLIR phase (keep first ~600 chars of the core description)
    if len(instruction) > 800:
        instruction = instruction[:800] + "..."

    # Add a concise header
    abstracted = (
        "Generate a GPU kernel (in MLIR JSON form) that implements the following operation.\n\n"
        f"{instruction}\n\n"
        "Focus on the mathematical semantics, pointer arithmetic, and control flow. "
        "Use standard MLIR operations (arith, scf, tt dialects)."
    )
    return abstracted


# Small curated subset for quick evaluation.
# Maps: friendly_name -> (json_entry_matcher_hint, test_filename)
# The matcher_hint is a substring we look for in the instruction to identify the entry.
BENCHMARK_SUBSET = {
    "dequantize_rowwise": {
        "matcher": "row-wise dequantization",
        "test_file": "dequantize_rowwise.py",
    },
    "cosine_similarity": {
        "matcher": "cosine similarity",
        "test_file": "cosine_compute.py",
    },
    "cross_entropy": {
        "matcher": "cross entropy",
        "test_file": "cross_entropy1.py",
    },
    "batched_vecmat": {
        "matcher": "batched vector-matrix",
        "test_file": "batched_vecmat_mult.py",
    },
    "add_example": {
        "matcher": "element-wise addition",
        "test_file": "add_example.py",
    },
}


def load_benchmark_subset() -> dict:
    """
    Load the curated subset of TritonBench-G benchmarks.
    Returns: dict mapping name -> {instruction, abstracted_prompt, test_path, output_code}
    """
    ensure_tritonbench_cloned()
    dataset = load_json_dataset()

    results = {}
    for name, cfg in BENCHMARK_SUBSET.items():
        matcher = cfg["matcher"].lower()
        entry = None
        for item in dataset:
            instr_lower = item["instruction"].lower()
            if matcher in instr_lower:
                entry = item
                break

        if entry is None:
            print(f"[Warning] Could not find TritonBench entry for '{name}' with matcher '{matcher}'")
            continue

        test_path = os.path.join(TRITONBENCH_DIR, TEST_DIR, cfg["test_file"])
        if not os.path.exists(test_path):
            print(f"[Warning] Test file not found: {test_path}")
            continue

        # Use the raw TritonBench instruction for Phase 1 (it already contains
        # explicit operation hints like tl.load, tl.store, BLOCK_SIZE, etc.).
        # Keep the old abstracted version as fallback for compatibility.
        raw_instruction = entry["instruction"]
        abstracted = _abstract_prompt(raw_instruction)

        results[name] = {
            "name": name,
            "instruction": raw_instruction,       # Primary: explicit TritonBench prompt
            "abstracted_prompt": abstracted,      # Fallback: jargon-stripped version
            "test_path": test_path,
            "reference_code": entry.get("output", ""),
        }

    return results


def get_test_function(test_path: str) -> str:
    """
    Extract the test harness from the reference Python file.
    THUNLP format: reference code, then a long separator line, then the test code.
    We return everything after the separator (includes def test_... and the call line).
    """
    with open(test_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try the standard THUNLP separator (146 '#' chars is common, but any long run works)
    for sep_len in [146, 100, 80, 50]:
        sep = "#" * sep_len
        if sep in content:
            parts = content.split(sep)
            if len(parts) > 1:
                test_part = parts[-1].strip()
                if "def test_" in test_part:
                    return test_part

    # Fallback: capture from first def test_ to end of file
    match = re.search(r"def test_.*", content, re.DOTALL)
    if match:
        return match.group(0).strip()

    return ""


def evaluate_generated_kernel(generated_code: str, test_path: str, tmp_dir: str = "output/tritonbench_eval") -> dict:
    """
    Evaluate a generated Triton kernel by concatenating it with the test function
    and running it via Python subprocess (mimicking THUNLP's 0_call_acc.py).
    """
    os.makedirs(tmp_dir, exist_ok=True)
    test_code = get_test_function(test_path)
    if not test_code:
        return {"success": False, "error": "Could not extract test function from reference file."}

    # Write combined file
    combined = generated_code + "\n" + "#" * 80 + "\n" + test_code + "\n"
    basename = os.path.basename(test_path)
    eval_path = os.path.join(tmp_dir, f"eval_{basename}")
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(combined)

    # Run it
    try:
        result = subprocess.run(
            ["python", eval_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        success = result.returncode == 0
        return {
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "eval_path": eval_path,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Evaluation timed out after 120s.", "eval_path": eval_path}
    except Exception as e:
        return {"success": False, "error": str(e), "eval_path": eval_path}


if __name__ == "__main__":
    # Quick smoke test
    benchmarks = load_benchmark_subset()
    for name, data in benchmarks.items():
        print(f"\n{name}")
        print(f"  Abstracted (first 200 chars): {data['abstracted_prompt'][:200]}...")
        print(f"  Test file exists: {os.path.exists(data['test_path'])}")
        test_fn = get_test_function(data['test_path'])
        print(f"  Test function length: {len(test_fn)} chars")
