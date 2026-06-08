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
    
    # Setup output directory for artifacts
    os.makedirs("output", exist_ok=True)
    print("[Info] Artifacts will be saved to ./output/")
    
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
        tracker = MLOpsTracker(job_type="llm-mlir-compiler")
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
                print(f"  [Attempt {attempt+1}/{max_retries}] Stage 1/5: Calling Gemini API...")
                raw_response = generate_llm_response("gemini", system_prompt, current_user_prompt, schema=MlirResponse)
                print(f"    -> LLM responded ({len(raw_response)} chars)")
                
                # Save raw response for debugging
                with open(f"output/{name}_attempt{attempt+1}_raw.txt", "w") as f:
                    f.write(raw_response)
                
                clean_json = raw_response.strip()
                import re
                
                # Try to extract from ```json ... ``` first
                json_match = re.search(r'```json\s*(.*?)\s*```', clean_json, flags=re.DOTALL)
                if json_match:
                    clean_json = json_match.group(1).strip()
                else:
                    # Remove markdown code blocks
                    clean_json = re.sub(r'```.*?```', '', clean_json, flags=re.DOTALL).strip()
                    # Try to find outermost braces
                    brace_match = re.search(r'(\{.*\})', clean_json, flags=re.DOTALL)
                    if brace_match:
                        clean_json = brace_match.group(1).strip()
                    else:
                        clean_json = clean_json.strip()
                
                print(f"    -> Extracted JSON ({len(clean_json)} chars)")
                print(f"    -> JSON preview: {clean_json[:200]}...")
                
                response_json = json.loads(clean_json)
                mlir_obj = MlirResponse(**response_json)
                print("    -> Pydantic validation PASSED")
                
                # Save parsed JSON
                with open(f"output/{name}_attempt{attempt+1}_parsed.json", "w") as f:
                    json.dump(response_json, f, indent=2)
                
                print("  [2/5] Stage 2/5: Validating MLIR Semantics...")
                semantic_errors = validator.validate(mlir_obj)
                if semantic_errors:
                    error_msg = "\n".join(semantic_errors)
                    print(f"    -> Validation failed:\n{error_msg}")
                    
                    if tracker: tracker.log_iteration(attempt, current_user_prompt, clean_json, "", False, error_msg)
                    
                    code_json_str = clean_json
                    try:
                        parsed = json.loads(clean_json)
                        if "code" in parsed:
                            code_json_str = json.dumps({"code": parsed["code"]}, indent=2)
                    except:
                        pass
                        
                    # Accumulate history so the SLM doesn't regress
                    error_history += f"\n- Attempt {attempt+1} errors:\n{error_msg}\n"
                    
                    snippet = raw_response if len(raw_response) < 1000 else raw_response[:500] + "\n...[TRUNCATED]...\n" + raw_response[-500:]
                    feedback = f"\n\n--- PREVIOUS ATTEMPTS HISTORY ---{error_history}\n"
                    feedback += f"\nIn your last attempt, you generated this JSON:\n{snippet}\n\n"
                    
                    if "scf.for loop defines" in error_msg and "iter_args but returns" in error_msg:
                        feedback += "CRITICAL RULE: The number of 'results' in scf.for MUST EXACTLY MATCH the number of 'iter_args'.\n"
                    if "missing an scf.yield operation" in error_msg:
                        feedback += "CRITICAL RULE: The LAST operation inside a 'scf.for' or 'scf.if' body MUST be 'scf.yield'. Do NOT forget to add the yield operation.\n"
                    if "requires a pointer or tensor of pointers" in error_msg:
                        feedback += "CRITICAL RULE: 'tt.load' or 'tt.store' MUST receive a pointer. If you have a base pointer like '%arg0_ptr', you must broadcast it using 'tt.splat' and then add offsets using 'tt.addptr'. Never pass raw scalars or standard tensors.\n"
                    if "used in 'scf.yield' but was never defined in this scope" in error_msg:
                        feedback += "CRITICAL RULE VIOLATION: You yielded the final result variable of the scf.for loop itself (e.g., '%final_max') inside the loop body. Inside the loop body, those final variables do not exist yet! You MUST yield the *newly computed values* for the current iteration (e.g., '%new_max' or '%current_max') so they can be passed to the next iteration.\n"
                    elif "not found in environment" in error_msg or "never defined in this scope" in error_msg:
                        feedback += "CRITICAL RULE VIOLATION: You used a register that DOES NOT EXIST. In MLIR, you cannot invent variables like '%is_max'. If you need a boolean condition, compute it first using 'arith.cmpf'. Every operand must be the 'result' of a previous operation.\n"
                    if "requires an 'axis' attribute" in error_msg:
                        feedback += "CRITICAL RULE: 'tt.reduce' MUST have an 'axis' attribute (e.g. {\"axis\": 0}). Do not forget it!\n"
                    if "incorrect number of indices for extract_element" in error_msg:
                        feedback += "CRITICAL RULE VIOLATION: You used 'tensor.extract' with the wrong number of indices. DO NOT use 'tensor.extract' to slice a row! To slice a row, you MUST use Triton pointer arithmetic ('tt.make_range', 'tt.splat', 'tt.addptr', 'tt.load').\n"
                    if "requires a single operand" in error_msg:
                        feedback += "CRITICAL RULE: Correct the arity of the operation.\n"
                    
                    feedback += "\nAnalyze ALL past errors, correct your JSON, and ensure strict compliance with MLIR rules."
                    current_user_prompt = base_user_prompt + feedback
                    if tracker:
                        tracker.log_iteration(attempt, base_user_prompt, raw_response, triton_python, False, error_msg)
                    continue
                
                print("  [4/4] Compiling to PTX via Triton Backend...")
                if backend:
                    ptx_code = backend.compile_ttir_to_ptx(ttir_code)
                    success = True
                    results[name] = {"status": "success", "attempts": attempt + 1}
                    
                    if tracker: tracker.log_iteration(attempt, current_user_prompt, clean_json, ttir_code, True)
                    
                    os.makedirs("output_ptx", exist_ok=True)
                    with open(f"output_ptx/{name}.ptx", "w") as f:
                        f.write(ptx_code)
                    print(f"    -> SUCCESS! Saved to output_ptx/{name}.ptx")
                    if tracker: tracker.save_artifact(f"output_ptx/{name}.ptx", f"{name}_ptx")
                    break
                else:
                    if tracker: tracker.log_iteration(attempt, current_user_prompt, clean_json, ttir_code, True)
                    results[name] = {"status": "ttir_generated"}
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"  -> Remote server error: {e}")
                print("     Retrying after 10 seconds...")
                import time
                time.sleep(10)
                if attempt == max_retries - 1:
                    results[name] = {"status": "remote_server_error", "error": str(e), "attempts": max_retries}
            except TimeoutError as e:
                print(f"  -> Job timed out: {e}")
                results[name] = {"status": "timeout_error", "attempts": attempt + 1}
                break
            except Exception as e:
                import traceback
                error_str = str(e)
                full_traceback = traceback.format_exc()
                print(f"  -> Pipeline exception: {error_str}")
                
                if "CUDA out of memory" in error_str:
                    results[name] = {"status": "remote_server_error", "error": "CUDA out of memory", "attempts": attempt + 1}
                    break
                
                error_history += f"\n- Attempt {attempt+1} exception:\n{error_str}\n"
                
                code_json_str = ""
                try:
                    if 'clean_json' in locals():
                        parsed = json.loads(clean_json)
                        if "code" in parsed:
                            code_json_str = json.dumps({"code": parsed["code"]}, indent=2)
                except:
                    pass
                
                feedback = f"\n\n--- PREVIOUS ATTEMPTS HISTORY ---{error_history}\n"
                snippet = code_json_str if len(code_json_str) < 500 else code_json_str[:250] + "\n...[TRUNCATED]...\n" + code_json_str[-250:]
                feedback += f"\nYou generated this code in the last attempt:\n```json\n{snippet}\n```\n\n"
                
                if "not found in environment" in error_str:
                    feedback += "CRITICAL RULE VIOLATION: You used a register that DOES NOT EXIST. Every operand must be the 'result' of a previous operation.\n"
                elif "attributes.value" in error_str:
                    feedback += "CRITICAL RULE: 'arith.constant' MUST have a 'value' field (e.g. \"value\": 0.0) so the compiler knows the numeric value.\n"
                if "must be floating-point-like, but got '!tt.ptr<f32>'" in error_str:
                    feedback += "CRITICAL RULE VIOLATION: The compiler failed because you tried to do Math (like arith.addf) on POINTERS. This happened because you forgot to add explicit 'out_type': 'tensor<...xf32>' to your 'tt.load' operation, so the compiler assumed it returned a pointer instead of math data.\n"
                if "literal_error" in error_str or "validation errors for MlirResponse" in error_str:
                    feedback += "CRITICAL RULE VIOLATION: Pydantic Schema Validation Failed. Make sure you included 'operands': [] even if the operation takes no operands (like tt.make_range), and ensure your 'out_type' strictly follows the MLIR syntax.\n"
                if "failed to verify that result type matches ptr type" in error_str:
                    feedback += "CRITICAL RULE VIOLATION: 'tt.addptr' MUST return EXACTLY the same type as its pointer operand! If your input pointer is 'tensor<...x!tt.ptr<f32>>', your 'out_type' MUST also be exactly 'tensor<...x!tt.ptr<f32>>'. Do not change the type or shape!\n"
                if "operand #1 must be 1-bit signless integer" in error_str and "tt.load" in error_str:
                    feedback += "CRITICAL RULE VIOLATION: The mask operand (operand #1) of 'tt.load' or 'tt.store' MUST be a boolean tensor (i1), e.g., 'tensor<1024xi1>'. You passed an i32 tensor instead. Use 'arith.cmpi' to create a boolean mask first!\n"
                if "'tt.addptr' op operand #0 must be ptr" in error_str:
                    feedback += "CRITICAL RULE VIOLATION: The first operand of 'tt.addptr' MUST be a pointer (e.g. '!tt.ptr<f32>'). You passed a math tensor (like 'f32'). You must pass a base pointer, NOT a loaded value!\n"
                if "JSONDecodeError" in error_str or "Expecting value:" in error_str or "Unterminated string" in error_str:
                    feedback += "CRITICAL RULE VIOLATION: The JSON is invalid or truncated. This happens when you hit the token limit! You MUST be more concise, use 'scf.for' loops instead of unrolling manually, DO NOT generate redundant operations or duplicate constants, and ensure the JSON is fully closed.\n"
                if "@triton.jit" in error_str or "jit functions should be defined" in error_str:
                    feedback += "CRITICAL RULE: The Triton kernel must be defined in a proper Python file. Do not use eval/exec.\n"
                    
                current_user_prompt = base_user_prompt + feedback + "\nAnalyze ALL past errors, correct your JSON, and ensure strict compliance with MLIR rules."
                
                if attempt == max_retries - 1:
                    results[name] = {"status": "exception", "error": error_str, "attempts": max_retries}
                    if tracker: tracker.log_iteration(attempt, base_user_prompt, raw_response, "", False, error_str)
                    
        if not success and name not in results:
            results[name] = {"status": "validation_failed_after_retries", "attempts": max_retries}
            
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
    
    # List generated artifacts
    print("\n=== Generated Artifacts ===")
    if os.path.exists("output"):
        artifacts = sorted(os.listdir("output"))
        if artifacts:
            for f in artifacts:
                print(f"  output/{f}")
        else:
            print("  (no artifacts generated)")
    else:
        print("  (output directory missing)")


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
