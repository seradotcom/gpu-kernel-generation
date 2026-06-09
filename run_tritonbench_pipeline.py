#!/usr/bin/env python3
"""
Simplified TritonBench Pipeline: Direct Triton Python Generation + Retry Loop

Architecture:
1. For each benchmark, use the raw TritonBench instruction as the user prompt.
2. Generate Triton Python directly via LLM (vLLM backend).
3. Evaluate against THUNLP's native test harness.
4. If it fails, feed the error back to the LLM and retry (max 3 attempts).

To run:
    python run_tritonbench_pipeline.py

Or in a notebook:
    %run run_tritonbench_pipeline.py
"""

import json
import os
import re
import traceback

from core.llm_client import generate_llm_response
from core.prompt_builder import PromptBuilder
from core.tritonbench_loader import (
    load_benchmark_subset,
    evaluate_generated_kernel,
)


def _extract_python(raw: str) -> str:
    """Extract Python code from markdown or raw string."""
    raw = raw.strip()
    match = re.search(r"```python\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


def run_tritonbench_pipeline():
    print("=" * 70)
    print("THUNLP TritonBench Pipeline — Direct Triton Python")
    print("=" * 70)

    os.makedirs("output", exist_ok=True)

    prompt_builder = PromptBuilder()
    system_prompt = prompt_builder.build_triton_python_prompt()

    print("Loading THUNLP TritonBench subset...")
    benchmarks = load_benchmark_subset()
    if not benchmarks:
        print("ERROR: No benchmarks loaded. Make sure thunlp_tritonbench/ is cloned.")
        return {}

    print(f"Loaded {len(benchmarks)} benchmarks: {', '.join(benchmarks.keys())}\n")

    results = {}
    MAX_ATTEMPTS = 3

    for name, bench_data in benchmarks.items():
        print(f"\n{'=' * 70}")
        print(f"[{name.upper()}]")
        print(f"{'=' * 70}")

        raw_instruction = bench_data["instruction"]
        test_path = bench_data["test_path"]
        success = False
        error_history = ""
        generated_kernel = ""

        for attempt in range(MAX_ATTEMPTS):
            print(f"\n  [Attempt {attempt + 1}/{MAX_ATTEMPTS}]")

            # Build user prompt: raw instruction + previous error feedback
            user_prompt = raw_instruction
            if error_history:
                user_prompt += (
                    f"\n\nYour previous attempt FAILED with this error:\n"
                    f"{'=' * 60}\n{error_history}\n{'=' * 60}\n"
                    f"Please fix the code and regenerate the complete Triton Python kernel."
                )

            try:
                print("    -> Calling LLM for Triton Python generation...")
                raw_response = generate_llm_response(
                    "vllm", system_prompt, user_prompt, schema=None
                )
                print(f"    -> LLM responded ({len(raw_response)} chars)")

                # Extract Python code
                generated_kernel = _extract_python(raw_response)
                artifact_path = f"output/{name}_attempt{attempt + 1}.py"
                with open(artifact_path, "w") as f:
                    f.write(generated_kernel)
                print(f"    -> Saved to {artifact_path}")

                # Evaluate using THUNLP native test harness
                print("    -> Running THUNLP native evaluation...")
                eval_result = evaluate_generated_kernel(generated_kernel, test_path)

                if eval_result["success"]:
                    success = True
                    results[name] = {
                        "status": "success",
                        "attempts": attempt + 1,
                        "eval_path": eval_result.get("eval_path"),
                    }
                    print(f"\n  [SUCCESS] THUNLP test passed after {attempt + 1} attempt(s)")
                    break
                else:
                    error_msg = eval_result.get("error", eval_result.get("stderr", "Unknown error"))
                    print(f"    -> FAILED: {error_msg[:500]}")
                    error_history = error_msg

            except Exception as e:
                print(f"    -> Exception: {str(e)[:400]}")
                traceback.print_exc()
                error_history = traceback.format_exc()

        if not success:
            print(f"\n  [FAILED] All {MAX_ATTEMPTS} attempts exhausted")
            results[name] = {
                "status": "failed",
                "attempts": MAX_ATTEMPTS,
                "error": error_history[:1000] if error_history else "",
            }

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    success_count = sum(1 for v in results.values() if v["status"] == "success")
    total = len(results)
    for k, v in results.items():
        status = "PASS" if v["status"] == "success" else "FAIL"
        print(f"{k:25s} | {status:4s} | Attempts: {v['attempts']}")

    if total > 0:
        print(f"\nSuccess Rate: {success_count}/{total} ({100 * success_count / total:.1f}%)")

    with open("output/tritonbench_pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to output/tritonbench_pipeline_results.json")
    print("\n=== Pipeline Finished ===")
    return results


if __name__ == "__main__":
    run_tritonbench_pipeline()
