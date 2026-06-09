#!/usr/bin/env python3
"""
Triton Direct Pipeline with MLIR Verify Feedback

Architecture:
1. Phase 1 (UNCHANGED): Generate JSON MLIR using existing constrained pipeline,
   validate semantics, translate to MLIR, and run .verify()
2. Phase 2 (NEW): Use the MLIR verify result (success or failure) as rich
   semantic feedback for the LLM to generate unconstrained Triton Python
3. Phase 3 (UNCHANGED): Compile, execute, and benchmark via TritonExecutor

This leverages the existing MLIR infrastructure for semantic validation
while letting the LLM generate natural Triton Python that actually compiles.
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
from core.triton_executor import TritonExecutor
from core.mlops_tracker import MLOpsTracker


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

    # Try ```python first
    match = re.search(r"```python\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic ```
    match = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    return raw.strip()


def _build_mlir_feedback(
    success: bool,
    mlir_code: str = "",
    error_msg: str = "",
    mlir_obj: MlirResponse = None
) -> str:
    """Build structured feedback from the MLIR verify step."""
    if success and mlir_obj:
        # Build rich feedback from the verified MLIR structure
        args_info = []
        for arg in mlir_obj.code.arguments:
            args_info.append(f"  - {arg.name}: {arg.type}")
        args_str = "\n".join(args_info) if args_info else "  (none)"

        ops_summary = []
        for op in mlir_obj.code.operations:
            opcode = getattr(op, "opcode", "unknown")
            if hasattr(opcode, "value"):
                opcode = opcode.value
            result = getattr(op, "result", "none")
            ops_summary.append(f"  - {opcode} -> {result}")
        # Limit to first 20 ops to save tokens
        ops_str = "\n".join(ops_summary[:20])
        if len(ops_summary) > 20:
            ops_str += f"\n  ... and {len(ops_summary) - 20} more operations"

        feedback = (
            f"MLIR VERIFICATION: SUCCESSFUL\n"
            f"The MLIR compiler has verified that the kernel structure is semantically correct.\n\n"
            f"Verified Function: {mlir_obj.code.function_name}\n"
            f"Arguments:\n{args_str}\n\n"
            f"Operation Sequence (first 20):\n{ops_str}\n\n"
            f"Full Verified MLIR:\n{mlir_code}\n\n"
            f"INSTRUCTION: Use this verified structure as a blueprint for your Triton Python code. "
            f"The types, shapes, and operation ordering have been compiler-validated. "
            f"Mirror this exact structure in natural Triton Python syntax."
        )
        return feedback
    else:
        # Even failures provide valuable negative feedback
        feedback = (
            f"MLIR VERIFICATION: FAILED\n"
            f"The MLIR compiler rejected the kernel structure with this exact error:\n"
            f"```\n{error_msg}\n```\n\n"
            f"INSTRUCTION: When writing Triton Python, you MUST avoid the semantic mistakes above. "
            f"Key lessons from the MLIR error:\n"
        )

        # Add targeted guidance based on common MLIR error patterns
        if "not found in environment" in error_msg or "never defined in this scope" in error_msg:
            feedback += (
                "- SSA ERROR: You used a variable before defining it. In Triton Python, "
                "ensure EVERY variable is assigned before use. No forward references.\n"
            )
        if "operand #1 must be ptr" in error_msg or "requires a pointer" in error_msg:
            feedback += (
                "- POINTER ERROR: `tl.load` and `tl.store` need pointer expressions, not raw values. "
                "Always do pointer arithmetic: `ptr + offsets` where `offsets = tl.arange(0, BLOCK_SIZE)`.\n"
            )
        if "iter_args" in error_msg and "results" in error_msg:
            feedback += (
                "- LOOP ERROR: Loop-carried variables must match in count between initialization and yield. "
                "In Python, ensure you initialize all loop variables before the `for` loop and update all of them each iteration.\n"
            )
        if "scf.yield" in error_msg:
            feedback += (
                "- YIELD ERROR: The last operation in a loop body must yield all loop-carried variables. "
                "In Python `for` loops, just assign to the same variable names; no explicit yield needed.\n"
            )
        if "type mismatch" in error_msg.lower() or "same type" in error_msg.lower():
            feedback += (
                "- TYPE ERROR: All operands to arithmetic operations must have compatible types. "
                "In Triton Python, use `float(x)` or `tl.cast(x, tl.float32)` if needed.\n"
            )
        if "failed to verify that result type matches ptr type" in error_msg:
            feedback += (
                "- POINTER TYPE ERROR: Pointer arithmetic must preserve pointer types. "
                "In Triton Python, `ptr + offsets` naturally preserves pointer semantics.\n"
            )
        if "arith.constant" in error_msg and "value" in error_msg:
            feedback += (
                "- CONSTANT ERROR: All literal values must be explicitly typed. "
                "In Triton Python, use `0.0` for float constants and `0` for int constants.\n"
            )

        feedback += (
            "\nGenerate Triton Python that avoids ALL these semantic pitfalls. "
            "The MLIR compiler has taught us what NOT to do."
        )
        return feedback


def run_triton_direct_pipeline():
    print("=== Triton Direct Pipeline with MLIR Verify Feedback ===")
    print("[Info] Phase 1: JSON+MLIR verify (existing pipeline)")
    print("[Info] Phase 2: Triton Python generation (unconstrained)")
    print("[Info] Phase 3: Triton compilation + benchmarking\n")

    os.makedirs("output", exist_ok=True)
    print("[Info] Artifacts will be saved to ./output/")

    # Load benchmark prompts
    prompts_file = "benchmark_prompts.json"
    if not os.path.exists(prompts_file):
        print(f"Error: {prompts_file} not found. Creating a default one.")
        default_prompts = {
            "vector_add": {
                "difficulty": "easy",
                "prompt": "Generate a Triton kernel that performs element-wise addition of two float32 vectors (A + B = C) using pointer arithmetic with a block size of 256."
            }
        }
        with open(prompts_file, "w") as f:
            json.dump(default_prompts, f, indent=2)
        print(f"Created {prompts_file}. Please edit it to add more benchmarks.")
        return

    with open(prompts_file, "r") as f:
        benchmarks = json.load(f)

    # Initialize existing components (ALL untouched)
    prompt_builder = PromptBuilder()
    validator = SemanticValidator()
    translator = MLIRTranslator()
    executor = TritonExecutor()

    # Optional MLOps tracker
    try:
        tracker = MLOpsTracker(job_type="triton-direct-mlir-feedback")
    except Exception as e:
        tracker = None
        print(f"[Info] Running without MLOps tracking: {e}")

    results = {}

    for name, data in benchmarks.items():
        difficulty = data.get("difficulty", "unknown")
        base_user_prompt = data["prompt"]

        print(f"\n{'='*60}")
        print(f"[{name.upper()}] (Difficulty: {difficulty})")
        print(f"{'='*60}")
        print(f"Task: {base_user_prompt}")

        # =====================================================================
        # PHASE 1: Existing JSON + MLIR pipeline (unchanged logic)
        # =====================================================================
        system_prompt_json = prompt_builder.build_prompt(
            base_user_prompt, MlirResponse.model_json_schema()
        )

        mlir_feedback = None
        json_raw = ""
        mlir_code = ""
        mlir_obj = None
        mlir_success = False

        max_json_attempts = 2
        for json_attempt in range(max_json_attempts):
            print(f"\n  [Phase 1/3] MLIR Generation Attempt {json_attempt+1}/{max_json_attempts}")
            current_user_prompt = base_user_prompt if json_attempt == 0 else (
                base_user_prompt + f"\n\nPrevious JSON attempt failed. Fix these errors and regenerate valid JSON."
            )

            try:
                # Call Gemini with constrained JSON (existing logic)
                print("    -> Calling LLM for JSON MLIR generation...")
                json_raw = generate_llm_response(
                    "gemini", system_prompt_json, current_user_prompt, schema=MlirResponse
                )
                print(f"    -> LLM responded ({len(json_raw)} chars)")

                # Save raw JSON response
                with open(f"output/{name}_mlir_attempt{json_attempt+1}_raw.json", "w") as f:
                    f.write(json_raw)

                # Extract and parse JSON (same logic as run_benchmarks.py)
                clean_json = _extract_json(json_raw)

                # Try json.loads
                try:
                    response_json = json.loads(clean_json)
                except json.JSONDecodeError as e:
                    # Try to find outermost braces as fallback
                    brace_match = re.search(r'(\{.*\})', clean_json, re.DOTALL)
                    if brace_match:
                        response_json = json.loads(brace_match.group(1).strip())
                    else:
                        raise

                mlir_obj = MlirResponse(**response_json)
                print("    -> Pydantic validation PASSED")

                # Save parsed JSON
                with open(f"output/{name}_mlir_attempt{json_attempt+1}_parsed.json", "w") as f:
                    json.dump(response_json, f, indent=2)

                # Semantic validation (existing)
                print("    -> Running semantic validation...")
                semantic_errors = validator.validate(mlir_obj)
                if semantic_errors:
                    error_msg = "\n".join(semantic_errors)
                    raise RuntimeError(f"Semantic validation failed:\n{error_msg}")
                print("    -> Semantic validation PASSED")

                # Translate to MLIR and verify (existing)
                print("    -> Translating to MLIR and running .verify()...")
                mlir_code = translator.translate_to_module(mlir_obj.code)
                print("    -> MLIR .verify() PASSED")

                # Save verified MLIR
                with open(f"output/{name}_mlir_attempt{json_attempt+1}.mlir", "w") as f:
                    f.write(mlir_code)

                mlir_success = True
                print(f"\n  [Phase 1/3] MLIR pipeline SUCCESS after {json_attempt+1} attempt(s)")
                break

            except Exception as e:
                error_str = str(e)
                print(f"    -> FAILED: {error_str[:300]}")
                traceback.print_exc()

                # Save error for debugging
                with open(f"output/{name}_mlir_attempt{json_attempt+1}_error.txt", "w") as f:
                    f.write(traceback.format_exc())

                if json_attempt < max_json_attempts - 1:
                    print(f"    -> Retrying with error feedback...")
                else:
                    print(f"    -> Max MLIR retries reached. Proceeding with failure feedback.")

        # Build feedback from Phase 1 result
        mlir_feedback = _build_mlir_feedback(
            success=mlir_success,
            mlir_code=mlir_code,
            error_msg=error_str if not mlir_success else "",
            mlir_obj=mlir_obj
        )

        # Save MLIR feedback for inspection
        with open(f"output/{name}_mlir_feedback.txt", "w") as f:
            f.write(mlir_feedback)

        # =====================================================================
        # PHASE 2: Generate Triton Python using MLIR feedback
        # =====================================================================
        system_prompt_triton = prompt_builder.build_triton_python_prompt()

        triton_user_prompt = (
            f"Task Description:\n{base_user_prompt}\n\n"
            f"MLIR Structural Feedback (from compiler verification):\n"
            f"{'='*60}\n{mlir_feedback}\n{'='*60}\n\n"
            f"Based on the task and the MLIR structural feedback above, generate a complete "
            f"Triton Python kernel. {'The MLIR was verified, so follow its exact structure.' if mlir_success else 'The MLIR failed verification, so carefully avoid the semantic errors described above.'} "
            f"Output ONLY valid Python code with imports and a launcher function."
        )

        success = False
        triton_error_history = ""
        triton_code = ""

        max_triton_attempts = 3
        for triton_attempt in range(max_triton_attempts):
            print(f"\n  [Phase 2-3/3] Triton Python Attempt {triton_attempt+1}/{max_triton_attempts}")

            try:
                print("    -> Calling LLM for Triton Python generation (unconstrained)...")
                triton_raw = generate_llm_response(
                    "gemini", system_prompt_triton, triton_user_prompt, schema=None
                )
                print(f"    -> LLM responded ({len(triton_raw)} chars)")

                # Extract Python code
                triton_code = _extract_python(triton_raw)

                # Save Triton Python artifact
                artifact_path = f"output/{name}_triton_attempt{triton_attempt+1}.py"
                with open(artifact_path, "w") as f:
                    f.write(triton_code)
                print(f"    -> Saved to {artifact_path}")

                # =====================================================================
                # PHASE 3: Compile and benchmark (existing TritonExecutor)
                # =====================================================================
                print("    -> Compiling and benchmarking with Triton...")
                exec_result = executor.run(triton_code, n_elements=1024, warmup=2, reps=10)

                if exec_result["success"]:
                    success = True
                    results[name] = {
                        "status": "success",
                        "mlir_verified": mlir_success,
                        "json_attempts": json_attempt + 1 if mlir_success else max_json_attempts,
                        "triton_attempts": triton_attempt + 1,
                        "correct": exec_result["correct"],
                        "speedup": exec_result.get("speedup"),
                        "kernel_time_ms": exec_result.get("kernel_time_ms"),
                        "ref_time_ms": exec_result.get("ref_time_ms"),
                    }
                    print(f"\n  [Phase 2-3/3] SUCCESS!")
                    print(f"    -> Correct: {exec_result['correct']}")
                    print(f"    -> Speedup: {exec_result.get('speedup', 'N/A')}x")
                    print(f"    -> Kernel time: {exec_result.get('kernel_time_ms', 'N/A')} ms")

                    if tracker:
                        tracker.log_iteration(
                            triton_attempt, base_user_prompt, triton_raw, triton_code,
                            True, None
                        )
                    break
                else:
                    error_msg = exec_result["error"]
                    print(f"    -> Triton compilation/execution FAILED")
                    print(f"    -> Error: {error_msg[:400]}")
                    triton_error_history += f"\n- Triton attempt {triton_attempt+1}:\n{error_msg}\n"

                    # Add failure feedback for retry
                    triton_user_prompt += (
                        f"\n\nYour previous Triton Python code FAILED to compile or execute with this error:\n"
                        f"```\n{error_msg}\n```\n"
                        f"Fix the code and regenerate. Pay attention to:\n"
                    )

                    # Extract actionable guidance from Triton error
                    if "expected" in error_msg.lower() and "got" in error_msg.lower():
                        triton_user_prompt += "- Type/shape mismatch in the kernel signature or body.\n"
                    if "undefined" in error_msg.lower() or "not defined" in error_msg.lower():
                        triton_user_prompt += "- A variable is used before being defined.\n"
                    if "mask" in error_msg.lower():
                        triton_user_prompt += "- The mask shape or type is incorrect.\n"
                    if "pointer" in error_msg.lower():
                        triton_user_prompt += "- Pointer arithmetic is wrong. Ensure `ptr + offsets` not `ptr * offsets`.\n"

                    if tracker:
                        tracker.log_iteration(
                            triton_attempt, base_user_prompt, triton_raw, triton_code,
                            False, error_msg
                        )

            except Exception as e:
                print(f"    -> Exception in Triton phase: {str(e)[:400]}")
                traceback.print_exc()
                triton_error_history += f"\n- Triton attempt {triton_attempt+1} exception:\n{traceback.format_exc()}\n"
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
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    success_count = 0
    for k, v in results.items():
        status = v["status"]
        if status == "success":
            success_count += 1
            mlir_ok = "VERIFIED" if v.get("mlir_verified") else "FAILED"
            correct = v.get("correct", False)
            speedup = v.get("speedup")
            speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
            print(
                f"{k:20s} | SUCCESS | MLIR: {mlir_ok:7s} | "
                f"Correct: {correct} | Speedup: {speedup_str} | "
                f"Attempts: J={v.get('json_attempts', 0)} T={v['triton_attempts']}"
            )
        else:
            mlir_ok = "VERIFIED" if v.get("mlir_verified") else "FAILED"
            print(f"{k:20s} | FAILED  | MLIR: {mlir_ok:7s}")

    total = len(results)
    print(f"\nSuccess Rate: {success_count}/{total} ({100*success_count/total:.1f}%)")

    # Save full results
    with open("output/triton_direct_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to output/triton_direct_results.json")

    if tracker:
        tracker.finish()

    print("\n=== Pipeline Finished ===")
    return results


if __name__ == "__main__":
    run_triton_direct_pipeline()
