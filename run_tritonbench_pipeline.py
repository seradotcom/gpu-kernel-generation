#!/usr/bin/env python3
"""
THUNLP TritonBench Pipeline with MLIR Verify Feedback

Architecture:
1. Phase 1 (UNCHANGED existing logic): Abstracted prompt -> JSON MLIR -> Semantic Validator -> MLIRTranslator.verify()
2. Phase 2 (NEW): Actual TritonBench prompt + MLIR feedback -> unconstrained Triton Python
3. Phase 3 (THUNLP native): Concatenate generated kernel with reference test function -> run -> report pass/fail

To run:
    python run_tritonbench_pipeline.py

Or in a notebook:
    %run run_tritonbench_pipeline.py
"""

import json
import os
import re
import sys
import traceback

from core.llm_client import generate_llm_response
from core.mlir_translator import MLIRTranslator
from core.prompt_builder import PromptBuilder
from core.schemas import MlirResponse
from core.semantic_validator import SemanticValidator
from core.tritonbench_loader import (
    load_benchmark_subset,
    get_test_function,
    evaluate_generated_kernel,
)


def _extract_json(raw: str) -> str:
    """Extract clean JSON from markdown or raw string."""
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


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


def _build_mlir_feedback(success: bool, mlir_code: str = "", error_msg: str = "", mlir_obj=None) -> str:
    """Build structured feedback from the MLIR verify step."""
    if success and mlir_obj:
        args_info = "\n".join([f"  - {arg.name}: {arg.type}" for arg in mlir_obj.code.arguments])
        ops_summary = []
        for op in mlir_obj.code.operations:
            opcode = getattr(op, "opcode", "unknown")
            if hasattr(opcode, "value"):
                opcode = opcode.value
            result = getattr(op, "result", "none")
            ops_summary.append(f"  - {opcode} -> {result}")
        ops_str = "\n".join(ops_summary[:20])
        if len(ops_summary) > 20:
            ops_str += f"\n  ... and {len(ops_summary) - 20} more operations"

        feedback = (
            f"MLIR VERIFICATION: SUCCESSFUL\n"
            f"The MLIR compiler verified the kernel structure is semantically correct.\n\n"
            f"Verified Function: {mlir_obj.code.function_name}\n"
            f"Arguments:\n{args_info}\n\n"
            f"Operation Sequence (first 20):\n{ops_str}\n\n"
            f"Verified MLIR:\n{mlir_code[:2000]}\n\n"
            f"INSTRUCTION: Use this verified structure as a blueprint. "
            f"Mirror the exact types, shapes, and operation ordering in your Triton Python code."
        )
        return feedback
    else:
        feedback = (
            f"MLIR VERIFICATION: FAILED\n"
            f"The MLIR compiler rejected the kernel with this error:\n"
            f"```\n{error_msg}\n```\n\n"
            f"INSTRUCTION: Avoid these semantic mistakes in your Triton Python:\n"
        )
        if "not found in environment" in error_msg or "never defined" in error_msg:
            feedback += "- SSA ERROR: Define all variables before using them.\n"
        if "operand #1 must be ptr" in error_msg or "requires a pointer" in error_msg:
            feedback += "- POINTER ERROR: tl.load/tl.store need proper pointer arithmetic (ptr + offsets).\n"
        if "iter_args" in error_msg and "results" in error_msg:
            feedback += "- LOOP ERROR: Ensure all loop variables are initialized and updated correctly.\n"
        if "type mismatch" in error_msg.lower():
            feedback += "- TYPE ERROR: All arithmetic operands must have compatible types.\n"
        feedback += (
            "\nGenerate Triton Python that avoids ALL these semantic pitfalls."
        )
        return feedback


def run_tritonbench_pipeline():
    print("=" * 70)
    print("THUNLP TritonBench Pipeline with MLIR Verify Feedback")
    print("=" * 70)
    print("[Phase 1] JSON + MLIR verify (existing pipeline)")
    print("[Phase 2] Triton Python generation (unconstrained)")
    print("[Phase 3] THUNLP native evaluation (test harness)\n")

    os.makedirs("output", exist_ok=True)

    # Initialize existing components (ALL untouched logic)
    prompt_builder = PromptBuilder()
    validator = SemanticValidator()
    translator = MLIRTranslator()

    # Load curated benchmark subset from THUNLP TritonBench
    print("Loading THUNLP TritonBench subset...")
    benchmarks = load_benchmark_subset()
    if not benchmarks:
        print("ERROR: No benchmarks loaded. Make sure thunlp_tritonbench/ is cloned.")
        return {}

    print(f"Loaded {len(benchmarks)} benchmarks: {', '.join(benchmarks.keys())}\n")

    results = {}

    for name, bench_data in benchmarks.items():
        print(f"\n{'=' * 70}")
        print(f"[{name.upper()}]")
        print(f"{'=' * 70}")

        abstracted_prompt = bench_data["abstracted_prompt"]
        actual_prompt = bench_data["instruction"]
        test_path = bench_data["test_path"]

        # =====================================================================
        # PHASE 1: Minimal JSON prompt for Groq (schema stripped to avoid 413)
        # =====================================================================
        print("\n[Phase 1/3] MLIR Generation and Verify")
        system_prompt_json = (
            "You are a GPU compiler expert. Generate a kernel as a JSON object with this exact top-level structure:\n"
            '{"reasoning": "...", "code": {"function_name": "...", "arguments": [...], "operations": [...], "returns": []}}\n\n'
            "Rules:\n"
            "- arguments: list of {name: string, type: string} (e.g., '!tt.ptr<f32>', 'tensor<256xf32>')\n"
            "- operations: list of {opcode: string, operands: [...], result: string, out_type: string (optional), attributes: dict (optional)}\n"
            "- Supported opcodes include: arith.addf, arith.cmpf, arith.constant, tt.load, tt.store, tt.splat, tt.make_range, tt.addptr, scf.for, scf.yield, scf.if, math.exp, tt.reduce\n"
            "- Use 'result': 'none' for ops with no output (e.g., tt.store, scf.yield).\n"
            "- Define every register before use. Use exact same type strings for operands.\n"
            "- The kernel must return an empty list: \"returns\": []\n"
            "- Output ONLY the JSON object. No markdown, no extra text.\n\n"
            "Task:\n" + abstracted_prompt
        )

        mlir_feedback = None
        json_raw = ""
        mlir_code = ""
        mlir_obj = None
        mlir_success = False
        error_str = ""

        max_json_attempts = 2
        for json_attempt in range(max_json_attempts):
            print(f"  MLIR attempt {json_attempt + 1}/{max_json_attempts}")
            current_user_prompt = abstracted_prompt if json_attempt == 0 else (
                abstracted_prompt + f"\n\nPrevious attempt failed with: {error_str[:500]}. Fix and regenerate valid JSON."
            )

            try:
                print("    -> Calling LLM for JSON MLIR generation...")
                json_raw = generate_llm_response(
                    "groq", system_prompt_json, current_user_prompt, schema=MlirResponse
                )
                print(f"    -> LLM responded ({len(json_raw)} chars)")

                # Save raw JSON
                with open(f"output/{name}_mlir_attempt{json_attempt + 1}_raw.json", "w") as f:
                    f.write(json_raw)

                # Extract and parse JSON (same logic as existing code)
                clean_json = _extract_json(json_raw)
                try:
                    response_json = json.loads(clean_json)
                except json.JSONDecodeError:
                    brace_match = re.search(r'(\{.*\})', clean_json, re.DOTALL)
                    if brace_match:
                        response_json = json.loads(brace_match.group(1).strip())
                    else:
                        raise

                mlir_obj = MlirResponse(**response_json)
                print("    -> Pydantic validation PASSED")

                # Semantic validation
                print("    -> Running semantic validation...")
                semantic_errors = validator.validate(mlir_obj)
                if semantic_errors:
                    raise RuntimeError(f"Semantic validation failed:\n" + "\n".join(semantic_errors))
                print("    -> Semantic validation PASSED")

                # Translate to MLIR and verify
                print("    -> Translating to MLIR and running .verify()...")
                mlir_code = translator.translate_to_module(mlir_obj.code)
                print("    -> MLIR .verify() PASSED")

                # Save verified MLIR
                with open(f"output/{name}_mlir_attempt{json_attempt + 1}.mlir", "w") as f:
                    f.write(mlir_code)

                mlir_success = True
                print(f"\n  [Phase 1/3] MLIR SUCCESS after {json_attempt + 1} attempt(s)")
                break

            except Exception as e:
                error_str = str(e)
                print(f"    -> FAILED: {error_str[:400]}")
                traceback.print_exc()
                with open(f"output/{name}_mlir_attempt{json_attempt + 1}_error.txt", "w") as f:
                    f.write(traceback.format_exc())

                if json_attempt < max_json_attempts - 1:
                    print("    -> Retrying...")
                else:
                    print("    -> Max MLIR retries reached. Proceeding with failure feedback.")

        # Build MLIR feedback
        mlir_feedback = _build_mlir_feedback(
            success=mlir_success,
            mlir_code=mlir_code,
            error_msg=error_str if not mlir_success else "",
            mlir_obj=mlir_obj,
        )
        with open(f"output/{name}_mlir_feedback.txt", "w") as f:
            f.write(mlir_feedback)

        # =====================================================================
        # PHASE 2: Generate Triton Python using MLIR feedback + actual prompt
        # =====================================================================
        print("\n[Phase 2/3] Triton Python Generation")
        system_prompt_triton = prompt_builder.build_triton_python_prompt()

        triton_user_prompt = (
            f"Task Description (from TritonBench):\n{actual_prompt}\n\n"
            f"MLIR Structural Feedback:\n{'=' * 60}\n{mlir_feedback}\n{'=' * 60}\n\n"
            f"Based on the task and the MLIR structural feedback above, generate a complete "
            f"Triton Python kernel. {'The MLIR was verified, so follow its exact structure.' if mlir_success else 'The MLIR failed verification, so carefully avoid the semantic errors described above.'} "
            f"Output ONLY valid Python code with imports and a launcher function."
        )

        success = False
        triton_error_history = ""
        triton_code = ""
        generated_kernel = ""

        max_triton_attempts = 3
        for triton_attempt in range(max_triton_attempts):
            print(f"  Triton attempt {triton_attempt + 1}/{max_triton_attempts}")

            try:
                print("    -> Calling LLM for Triton Python generation...")
                triton_raw = generate_llm_response(
                    "groq", system_prompt_triton, triton_user_prompt, schema=None
                )
                print(f"    -> LLM responded ({len(triton_raw)} chars)")

                # Extract Python code
                triton_code = _extract_python(triton_raw)
                generated_kernel = triton_code

                # Save artifact
                artifact_path = f"output/{name}_triton_attempt{triton_attempt + 1}.py"
                with open(artifact_path, "w") as f:
                    f.write(triton_code)
                print(f"    -> Saved to {artifact_path}")

                # =====================================================================
                # PHASE 3: Evaluate using THUNLP's native test harness
                # =====================================================================
                print("\n[Phase 3/3] THUNLP Native Evaluation")
                eval_result = evaluate_generated_kernel(triton_code, test_path)

                if eval_result["success"]:
                    success = True
                    results[name] = {
                        "status": "success",
                        "mlir_verified": mlir_success,
                        "json_attempts": json_attempt + 1 if mlir_success else max_json_attempts,
                        "triton_attempts": triton_attempt + 1,
                        "eval_path": eval_result.get("eval_path"),
                    }
                    print(f"\n  [Phase 3/3] SUCCESS! THUNLP test passed.")
                    break
                else:
                    error_msg = eval_result.get("error", eval_result.get("stderr", "Unknown error"))
                    print(f"    -> THUNLP evaluation FAILED")
                    print(f"    -> Error: {error_msg[:500]}")
                    triton_error_history += f"\n- Attempt {triton_attempt + 1}:\n{error_msg}\n"

                    # Add failure feedback for retry
                    triton_user_prompt += (
                        f"\n\nYour previous Triton Python code FAILED the THUNLP test with:\n"
                        f"```\n{error_msg}\n```\n"
                        f"Fix the code and regenerate.\n"
                    )

            except Exception as e:
                print(f"    -> Exception: {str(e)[:400]}")
                traceback.print_exc()
                triton_error_history += f"\n- Attempt {triton_attempt + 1} exception:\n{traceback.format_exc()}\n"
                triton_user_prompt += (
                    f"\n\nYour previous attempt caused an exception:\n{str(e)}\n"
                    f"Fix the code and regenerate.\n"
                )

        if not success:
            print(f"\n  [Phase 2-3/3] FAILED after {max_triton_attempts} attempts")
            results[name] = {
                "status": "failed",
                "mlir_verified": mlir_success,
                "mlir_feedback": mlir_feedback[:500] if mlir_feedback else "",
                "triton_errors": triton_error_history[:1000] if triton_error_history else "",
            }

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    success_count = 0
    for k, v in results.items():
        status = v["status"]
        if status == "success":
            success_count += 1
            mlir_ok = "VERIFIED" if v.get("mlir_verified") else "FAILED"
            print(
                f"{k:25s} | SUCCESS | MLIR: {mlir_ok:7s} | "
                f"Attempts: J={v.get('json_attempts', 0)} T={v['triton_attempts']}"
            )
        else:
            mlir_ok = "VERIFIED" if v.get("mlir_verified") else "FAILED"
            print(f"{k:25s} | FAILED  | MLIR: {mlir_ok:7s}")

    total = len(results)
    if total > 0:
        print(f"\nSuccess Rate: {success_count}/{total} ({100 * success_count / total:.1f}%)")
    else:
        print("\nNo benchmarks were evaluated.")

    # Save full results
    with open("output/tritonbench_pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to output/tritonbench_pipeline_results.json")
    print("\n=== Pipeline Finished ===")
    return results


if __name__ == "__main__":
    run_tritonbench_pipeline()
