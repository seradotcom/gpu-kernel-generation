import os
import sys
import json
import traceback
import requests

from core.llm_client import generate_llm_response
from core.schemas import MlirResponse
from core.mlir_translator import MLIRTranslator
from core.prompt_builder import PromptBuilder
from core.semantic_validator import SemanticValidator
from core.triton_python_generator import TritonPythonGenerator
from core.triton_executor import TritonExecutor

from core.mlops_tracker import MLOpsTracker


def run_benchmarks():
    print("=== Starting TritonBench LLM-MLIR Evaluator with Triton Python Feedback Loop ===")
    
    prompts_file = "benchmark_prompts.json"
    if not os.path.exists(prompts_file):
        print(f"Error: {prompts_file} not found. Creating a default one with Phase 1 prompts.")
        default_prompts = {
            "vector_add": {
                "difficulty": "easy",
                "prompt": "Generate a Triton kernel that performs element-wise addition of two float32 vectors (A + B = C) using pointer arithmetic with a block size of 256."
            },
            "element_mul": {
                "difficulty": "easy",
                "prompt": "Generate a Triton kernel that performs element-wise multiplication of two float32 vectors (A * B = C) using pointer arithmetic with a block size of 256."
            }
        }
        with open(prompts_file, "w") as f:
            json.dump(default_prompts, f, indent=2)
        print(f"Created {prompts_file}. Please edit it to add more benchmarks.")
        return
        
    with open(prompts_file, "r") as f:
        benchmarks = json.load(f)
        
    validator = SemanticValidator()
    translator = MLIRTranslator()
    triton_generator = TritonPythonGenerator()
    triton_executor = TritonExecutor()
    prompt_builder = PromptBuilder()
    
    try:
        tracker = MLOpsTracker(project_name="llm-mlir-compiler")
    except Exception as e:
        print(f"[Info] Running without MLOps tracking: {e}")
        tracker = None
    
    results = {}
    
    for name, data in benchmarks.items():
        difficulty = data.get("difficulty", "unknown")
        prompt_text = data["prompt"]
        
        print(f"\n[{name.upper()}] (Difficulty: {difficulty})")
        
        base_user_prompt = f"Implement the following Triton kernel logic: {prompt_text}"
        current_user_prompt = base_user_prompt
        system_prompt = prompt_builder.build_prompt(base_user_prompt, MlirResponse.model_json_schema())
        
        success = False
        max_retries = 3
        error_history = ""
        
        for attempt in range(max_retries):
            raw_response = ""
            try:
                print(f"  [Attempt {attempt+1}/{max_retries}] Generating JSON via LLM...")
                raw_response = generate_llm_response("gemini", system_prompt, current_user_prompt, schema=MlirResponse)
                
                clean_json = raw_response.strip()
                if clean_json.startswith("```json"): clean_json = clean_json[7:]
                if clean_json.startswith("```"): clean_json = clean_json[3:]
                if clean_json.endswith("```"): clean_json = clean_json[:-3]
                clean_json = clean_json.strip()
                
                response_json = json.loads(clean_json)
                mlir_obj = MlirResponse(**response_json)
                
                print("  [2/5] Validating MLIR Semantics...")
                semantic_errors = validator.validate(mlir_obj)
                if semantic_errors:
                    error_msg = "\n".join(semantic_errors)
                    print(f"    -> Semantic validation failed")
                    error_history += f"\n- Attempt {attempt+1} semantic errors:\n{error_msg}\n"
                    feedback = _build_feedback(error_history, raw_response, error_msg)
                    current_user_prompt = base_user_prompt + feedback
                    if tracker:
                        tracker.log_iteration(attempt, base_user_prompt, raw_response, "", False, error_msg)
                    continue
                
                print("  [3/5] Translating to MLIR and verifying...")
                mlir_code = translator.translate_to_module(mlir_obj.code)
                print(f"    -> MLIR verification passed")
                
                print("  [4/5] Generating Triton Python from verified JSON...")
                triton_python = triton_generator.generate(mlir_obj.code)
                print(f"    -> Generated Triton Python:\n{triton_python}")
                
                print("  [5/5] Compiling and benchmarking Triton kernel...")
                exec_result = triton_executor.run(triton_python, n_elements=1024, warmup=2, reps=10)
                
                if not exec_result["success"]:
                    error_msg = exec_result["error"]
                    print(f"    -> Triton execution failed:\n{error_msg}")
                    error_history += f"\n- Attempt {attempt+1} Triton execution error:\n{error_msg}\n"
                    feedback = _build_feedback(error_history, raw_response, error_msg)
                    current_user_prompt = base_user_prompt + feedback
                    if tracker:
                        tracker.log_iteration(attempt, base_user_prompt, raw_response, triton_python, False, error_msg)
                    continue
                
                # Success!
                success = True
                results[name] = {
                    "status": "success",
                    "attempts": attempt + 1,
                    "correct": exec_result["correct"],
                    "speedup": exec_result.get("speedup"),
                    "kernel_time_ms": exec_result.get("kernel_time_ms"),
                    "ref_time_ms": exec_result.get("ref_time_ms"),
                }
                print(f"    -> SUCCESS! Correct: {exec_result['correct']}, Speedup: {exec_result.get('speedup', 'N/A')}x")
                if tracker:
                    tracker.log_iteration(attempt, base_user_prompt, raw_response, triton_python, True, None)
                break
                
            except requests.exceptions.RequestException as e:
                print(f"  -> Remote server error: {e}")
                print("     Retrying after 10 seconds...")
                import time
                time.sleep(10)
                if attempt == max_retries - 1:
                    results[name] = {"status": "remote_server_error", "error": str(e)}
            except TimeoutError as e:
                print(f"  -> Job timed out: {e}")
                results[name] = {"status": "timeout_error"}
                break
            except Exception as e:
                error_str = str(e)
                print(f"  -> Pipeline exception: {error_str}")
                
                error_history += f"\n- Attempt {attempt+1} exception:\n{error_str}\n"
                
                code_json_str = ""
                try:
                    if 'clean_json' in locals():
                        parsed = json.loads(clean_json)
                        if "code" in parsed:
                            code_json_str = json.dumps({"code": parsed["code"]}, indent=2)
                except:
                    pass
                
                feedback = _build_feedback(error_history, raw_response, error_str)
                current_user_prompt = base_user_prompt + feedback
                
                if attempt == max_retries - 1:
                    results[name] = {"status": "exception", "error": error_str}
                    if tracker:
                        tracker.log_iteration(attempt, base_user_prompt, raw_response, "", False, error_str)
                    
        if not success and name not in results:
            results[name] = {"status": "validation_failed_after_retries"}
            
    if tracker: tracker.finish()
    
    print("\n=== Benchmark Summary ===")
    for k, v in results.items():
        status = v["status"]
        attempts = v.get("attempts", "N/A")
        if status == "success":
            correct = v.get("correct", False)
            speedup = v.get("speedup")
            print(f"{k.ljust(20)}: {status} (Attempts: {attempts}, Correct: {correct}, Speedup: {speedup:.2f}x)")
        else:
            print(f"{k.ljust(20)}: {status} (Attempts: {attempts})")


def _build_feedback(error_history: str, raw_response: str, error_msg: str) -> str:
    """Build a feedback string for the LLM based on errors."""
    snippet = raw_response if len(raw_response) < 1000 else raw_response[:500] + "\n...[TRUNCATED]...\n" + raw_response[-500:]
    
    feedback = f"\n\n--- PREVIOUS ATTEMPTS HISTORY ---{error_history}\n"
    feedback += f"\nIn your last attempt, you generated this JSON:\n{snippet}\n\n"
    
    # Add specific guidance based on error patterns
    if "not found in environment" in error_msg or "never defined in this scope" in error_msg:
        feedback += "CRITICAL RULE VIOLATION: You used a register that DOES NOT EXIST. Every operand must be the 'result' of a previous operation.\n"
    if "attributes.value" in error_msg:
        feedback += "CRITICAL RULE: 'arith.constant' MUST have a 'value' field (e.g. \"value\": 0.0) so the compiler knows the numeric value.\n"
    if "must be floating-point-like, but got '!tt.ptr<f32>'" in error_msg:
        feedback += "CRITICAL RULE VIOLATION: The compiler failed because you tried to do Math on POINTERS. Add explicit 'out_type': 'tensor<...xf32>' to 'tt.load'.\n"
    if "failed to verify that result type matches ptr type" in error_msg:
        feedback += "CRITICAL RULE: 'tt.addptr' MUST return the same type as its pointer operand!\n"
    if "Triton compiler" in error_msg or "@triton.jit" in error_msg:
        feedback += "CRITICAL RULE: The generated Triton Python failed to compile. Check that all variables are defined before use and types are consistent.\n"
    if "tl.load" in error_msg or "tl.store" in error_msg:
        feedback += "CRITICAL RULE: Check your pointer arithmetic. Ensure tt.make_range, tt.splat, and tt.addptr produce valid pointer tensors before load/store.\n"
    if "Correctness check failed" in error_msg or "allclose" in error_msg:
        feedback += "CRITICAL RULE: The kernel compiles but produces wrong output. Verify your algorithm logic, especially operand order in arithmetic ops.\n"
        
    feedback += "\nAnalyze ALL past errors, correct your JSON, and ensure strict compliance with MLIR rules."
    return feedback


if __name__ == "__main__":
    run_benchmarks()
