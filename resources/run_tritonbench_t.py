"""
run_tritonbench_t.py  —  Drive the gpu-kernel-generation pipeline over TritonBench-T
and emit a predictions.jsonl in the exact format the TritonBench eval scripts expect.

Place this file at the ROOT of the gpu-kernel-generation repo (next to run_benchmarks.py)
so the `from core...` imports resolve.

What the eval expects (see EVAL/eval_T/0_call_acc.py :: get_codes_for_test):
  - A .jsonl file, ONE json object per line.
  - The FIRST key of each object holds the full instruction string (the eval does
    key = list(data[0].keys())[0]).
  - A "predict" key holds the generated code (```python fences are tolerated; the eval
    strips them via clear_code()).
  - The eval appends the GOLD operator's test_*() block to your predict and runs it, so
    your predict MUST define a host-callable function with the EXACT name + signature
    from the instruction's "Wrapper Entry Information:" line — not just a @triton.jit kernel.

Pipeline per instruction:
  1. Build the system prompt and call the LLM to get the JSON MLIR-AST (MlirResponse).
  2. Validate + lower the AST to a bare @triton.jit kernel (TritonPythonGenerator).
  3. Generate a host wrapper whose signature matches "Wrapper Entry Information:" and
     which allocates output, computes the grid, launches the kernel, and returns a tensor.
  4. Concatenate kernel + wrapper as the `predict` payload.

Usage:
    python run_tritonbench_t.py \
        --alpaca /path/to/TritonBench/data/TritonBench_T_simp_alpac_v1.json \
        --out predictions.jsonl \
        --model gemini \
        --max_retries 3
"""

import os
import re
import sys
import json
import argparse
import traceback

from core.llm_client import generate_llm_response
from core.schemas import MlirResponse
from core.semantic_validator import SemanticValidator
from core.triton_python_generator import TritonPythonGenerator
from core.prompt_builder import PromptBuilder


# ----------------------------------------------------------------------------- helpers

def extract_json(raw: str) -> str:
    """Mirror run_benchmarks.py's JSON extraction from a raw LLM response."""
    s = raw.strip()
    m = re.search(r"```json\s*(.*?)\s*```", s, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    s = re.sub(r"```.*?```", "", s, flags=re.DOTALL).strip()
    m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    return m.group(1).strip() if m else s


def parse_wrapper_info(instruction: str) -> str:
    """Return the raw 'Wrapper Entry Information:' text (the target signature)."""
    return instruction.split("Wrapper Entry Information:")[-1].strip()


WRAPPER_SYS = (
    "You are an expert Triton + PyTorch engineer. You will be given a working "
    "@triton.jit kernel and a target wrapper signature. Write ONLY the Python host "
    "wrapper function that:\n"
    "  1. Has EXACTLY the given name and signature (defaults included).\n"
    "  2. Accepts torch tensors as described, moves/allocates an output tensor with "
    "torch.empty_like or the correct shape/dtype on the inputs' device.\n"
    "  3. Computes a 1-D grid with triton.cdiv over the number of elements and launches "
    "the provided kernel, passing arguments in the kernel's declared order.\n"
    "  4. Returns the output tensor.\n"
    "Do NOT redefine the kernel. Do NOT include a test. Output ONLY the wrapper function "
    "code, no markdown fences, no commentary."
)


def generate_host_wrapper(model: str, kernel_src: str, wrapper_info: str) -> str:
    """Second LLM pass: produce a host wrapper matching the TritonBench signature.

    This is the integration-critical stage: TritonBench calls a host function, not a
    bare kernel. If you prefer a deterministic path for simple elementwise ops, you can
    replace this call with a template that maps (in0, in1, ..., out) positionally.
    """
    user = (
        f"KERNEL (already defined, do not repeat it):\n```python\n{kernel_src}\n```\n\n"
        f"TARGET WRAPPER SIGNATURE (from TritonBench):\n{wrapper_info}\n\n"
        "Write the host wrapper now."
    )
    raw = generate_llm_response(model, WRAPPER_SYS, user, schema=None)
    # Strip any stray fences the model adds despite instructions.
    raw = raw.strip()
    m = re.search(r"```python\s*(.*?)\s*```", raw, flags=re.DOTALL)
    if m:
        raw = m.group(1).strip()
    return raw


# ----------------------------------------------------------------------------- per-item

def generate_one(model: str, instruction: str, max_retries: int,
                 validator: SemanticValidator, generator: TritonPythonGenerator,
                 builder: PromptBuilder) -> str:
    """Run the full pipeline for one instruction, returning the `predict` code string.

    On failure, returns a short comment string so the run still produces a line (the
    eval will simply count it as a non-passing case)."""
    base_user = instruction  # the TritonBench instruction IS the task description
    system_prompt = builder.build_prompt(base_user, MlirResponse.model_json_schema())
    user_prompt = base_user
    last_err = ""

    for attempt in range(max_retries):
        try:
            raw = generate_llm_response(model, system_prompt, user_prompt, schema=MlirResponse)
            obj = MlirResponse(**json.loads(extract_json(raw)))

            errs = validator.validate(obj)
            if errs:
                last_err = "\n".join(errs)
                user_prompt = base_user + f"\n\n--- Fix these MLIR errors ---\n{last_err}\n"
                continue

            kernel_src = generator.generate(obj.code)
            wrapper_src = generate_host_wrapper(model, kernel_src, parse_wrapper_info(instruction))

            # Order matters: kernel first (the wrapper references it by name).
            return kernel_src.rstrip() + "\n\n" + wrapper_src.strip() + "\n"

        except Exception as e:  # noqa: BLE001  — keep the batch going
            last_err = f"{e}\n{traceback.format_exc()}"
            user_prompt = base_user + f"\n\n--- Previous attempt failed ---\n{str(e)}\n"

    return f"# generation failed after {max_retries} attempts\n# last error: {last_err.splitlines()[-1] if last_err else 'unknown'}\n"


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpaca", required=True,
                    help="Path to TritonBench_T_simp_alpac_v1.json (or the _comp_ variant).")
    ap.add_argument("--out", default="predictions.jsonl")
    ap.add_argument("--model", default="gemini",
                    help="Model name passed to generate_llm_response (gemini, remote, ...).")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="Only run the first N (0 = all).")
    args = ap.parse_args()

    alpaca = json.load(open(args.alpaca, "r", encoding="utf-8"))
    if args.limit:
        alpaca = alpaca[:args.limit]

    validator = SemanticValidator()
    generator = TritonPythonGenerator()
    builder = PromptBuilder()

    n_ok = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, entry in enumerate(alpaca):
            instruction = entry["instruction"]
            print(f"[{i+1}/{len(alpaca)}] generating...", flush=True)
            predict = generate_one(args.model, instruction, args.max_retries,
                                   validator, generator, builder)
            if not predict.lstrip().startswith("#"):
                n_ok += 1
            # First key MUST be the instruction (eval reads list(obj.keys())[0]).
            fout.write(json.dumps({"instruction": instruction, "predict": predict},
                                  ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\nDone. Wrote {len(alpaca)} lines to {args.out} ({n_ok} produced code).")
    print("Next: build the stats file, fix eval paths, then run 0_call_acc.py / 1_exe_acc.py.")


if __name__ == "__main__":
    main()
