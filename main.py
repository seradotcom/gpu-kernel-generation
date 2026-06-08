import json
import traceback
from google import genai
from pydantic import ValidationError

from core.llm_client import generate_llm_response
from core.mlops_tracker import MLOpsTracker
from core.schemas import MlirResponse
from core.mlir_translator import MLIRTranslator

def main():
    """
    Main Orchestrator - Generation Loop and Strict Validation.
    Phase 1 and 2 of the LLM-Guided Compiler Project.
    """
    print("Starting MLOps and MLIR infrastructure...")
    
    # 1. Setup MLOps Tracker (Optional)
    try:
        # Try to initialize W&B only if the key exists in .env
        import core.config as config
        if config.WANDB_API_KEY:
            tracker = MLOpsTracker(job_type="vector-add-experiment")
        else:
            tracker = None
            print("[Info] WANDB_API_KEY not found. Running without MLOps.")
    except Exception as e:
        tracker = None
        print(f"[Info] Running without MLOps due to error: {e}")
    
    # 2. Instantiate the MLIR translator
    try:
        translator = MLIRTranslator()
    except ImportError as e:
        print(f"[!] Environment error: {e}")
        if tracker: tracker.finish()
        return

    # 3. Prompt Definition and Allowed Triton Operations
    from core.prompt_builder import PromptBuilder
    
    user_prompt = (
        "Kernel: One-Pass Online Softmax (Advanced).\n"
        "Objective: Calculate the Softmax operation across the rows of an input matrix "
        "in a single global memory pass, using an online reduction algorithm "
        "to dynamically update the Local Maximum and the Sum of Exponentials.\n"
        "Mandatory Hardware Condition:\n"
        "1. Define the appropriate thread geometry to process one row per block.\n"
        "2. Calculate and assign the exact shared memory bytes needed for threads "
        "to cooperate in parallel reduction using two independent intermediate arrays (one for "
        "maximums and another for accumulated sums) assuming 32-bit float data type.\n"
        "3. Detail in the logical steps the strict use of synchronization barriers (__syncthreads) "
        "to avoid race conditions during block reduction."
    )
    
    prompt_builder = PromptBuilder()
    system_prompt = prompt_builder.build_prompt(user_prompt, MlirResponse.model_json_schema())
    
    # Experiment parameters
    max_retries = 3
    success = False
    
    base_user_prompt = user_prompt
    current_user_prompt = user_prompt
    
    error_history = ""
    
    for attempt in range(max_retries):
        print(f"\n--- Attempt {attempt + 1}/{max_retries} ---")
        if tracker: tracker.start_timer()
        
        # Call the model using Constrained Decoding (XGrammar / Structured Outputs)
        print("Waiting for LLM generation (Constrained Decoding enabled)...")
        try:
            client = genai.Client(project="project-2f46fba3-c3a3-456a-9d1", location="us-central1")
            # Pass the actual Pydantic schema to the client (XGrammar at API level)
            raw_response = generate_llm_response("gemini", system_prompt, current_user_prompt, schema=MlirResponse)
        except Exception as e:
            print(f"[!] Error calling the API: {e}")
            break
            
        print("LLM responded. Parsing JSON contract...")
        mlir_code = ""
        error_msg = None
        
        try:
            # 1. Syntactic Validation (Almost always passes thanks to Constrained Decoding)
            # Strip markdown formatting to ensure clean JSON parsing
            clean_json = raw_response.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.startswith("```"):
                clean_json = clean_json[3:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()
            
            print(f"[DEBUG] Raw response length: {len(raw_response)}")
            print(f"[DEBUG] First 100 chars of clean_json: {clean_json[:100]}")
            
            parsed_json = json.loads(clean_json)
            response_obj = MlirResponse(**parsed_json)
            
            # 2. Semantic AST Validation
            from core.semantic_validator import SemanticValidator
            semantic_errors = SemanticValidator.validate(response_obj)
            if semantic_errors:
                raise RuntimeError("Multiple Semantic Errors Found:\n" + "\n".join(semantic_errors))
            
            # 3. Translation to MLIR
            print("Translating to MLIR Dialects...")
            mlir_code = translator.translate_to_module(response_obj.code)
            
            success = True
            print("[✓] Successful MLIR Compilation!")
            print(mlir_code)
            
            if tracker: tracker.log_iteration(attempt, user_prompt, raw_response, mlir_code, success, None)
            break
            
        except json.JSONDecodeError as e:
            error_msg = f"Malformed JSON: {e}"
        except ValidationError as e:
            error_msg = f"JSON Schema not respected: {e}"
        except RuntimeError as e:
            error_msg = str(e)
        except Exception as e:
            error_msg = f"General error: {traceback.format_exc()}"
            
        success = False
        print(f"[X] Semantic/Syntactic failure intercepted:\n{error_msg}")
        if tracker: tracker.log_iteration(attempt, user_prompt, raw_response, mlir_code, success, error_msg)

        code_json_str = clean_json
        try:
            parsed = json.loads(clean_json)
            if "code" in parsed:
                code_json_str = json.dumps({"code": parsed["code"]}, indent=2)
        except:
            pass

        error_history += f"\n- Attempt {attempt+1} errors:\n{error_msg}\n"

        feedback_context = (
            f"\n\n--- PREVIOUS ATTEMPTS HISTORY ---{error_history}\n"
            f"You generated this code in the last attempt:\n```json\n{code_json_str}\n```\n\n"
        )

        if "scf.for loop defines" in error_msg and "iter_args but returns" in error_msg:
            feedback_context += "CRITICAL RULE: The number of 'results' in scf.for MUST EXACTLY MATCH the number of 'iter_args'.\n"
        if "missing an scf.yield operation" in error_msg:
            feedback_context += "CRITICAL RULE: The LAST operation inside a 'scf.for' or 'scf.if' body MUST be 'scf.yield'. Do NOT forget to add the yield operation.\n"
        if "requires a pointer or tensor of pointers" in error_msg:
            feedback_context += "CRITICAL RULE: 'tt.load' or 'tt.store' MUST receive a pointer. If you have a base pointer like '%arg0_ptr', you must broadcast it using 'tt.splat' and then add offsets using 'tt.addptr'. Never pass raw scalars or standard tensors.\n"
        if "used in 'scf.yield' but was never defined in this scope" in error_msg:
            feedback_context += "CRITICAL RULE VIOLATION: You yielded the final result variable of the scf.for loop itself (e.g., '%final_max') inside the loop body. Inside the loop body, those final variables do not exist yet! You MUST yield the *newly computed values* for the current iteration (e.g., '%new_max' or '%current_max') so they can be passed to the next iteration.\n"
        elif "not found in environment" in error_msg or "never defined in this scope" in error_msg:
            feedback_context += "CRITICAL RULE VIOLATION: You used a register that DOES NOT EXIST. In MLIR, you cannot invent variables like '%is_max'. If you need a boolean condition, compute it first using 'arith.cmpf'. Every operand must be the 'result' of a previous operation.\n"
        if "requires an 'axis' attribute" in error_msg:
            feedback_context += "CRITICAL RULE: 'tt.reduce' MUST have an 'axis' attribute (e.g. {\"axis\": 0}). Do not forget it!\n"
        if "incorrect number of indices for extract_element" in error_msg:
            feedback_context += "CRITICAL RULE VIOLATION: You used 'tensor.extract' with the wrong number of indices. DO NOT use 'tensor.extract' to slice a row! To slice a row, you MUST use Triton pointer arithmetic ('tt.make_range', 'tt.splat', 'tt.addptr', 'tt.load').\n"
        if "requires a single operand" in error_msg:
            feedback_context += "CRITICAL RULE: Correct the arity of the operation.\n"
        if "must be floating-point-like, but got '!tt.ptr<f32>'" in error_msg:
            feedback_context += "CRITICAL RULE VIOLATION: The compiler failed because you tried to do Math (like arith.addf) on POINTERS. This happened because you forgot to add explicit 'out_type': 'tensor<...xf32>' to your 'tt.load' operation, so the compiler assumed it returned a pointer instead of math data.\n"
        if "literal_error" in error_msg or "validation errors for MlirResponse" in error_msg:
            feedback_context += "CRITICAL RULE VIOLATION: Pydantic Schema Validation Failed. Make sure you included 'operands': [] even if the operation takes no operands (like tt.make_range), and ensure your 'out_type' strictly follows the MLIR syntax.\n"
        if "failed to verify that result type matches ptr type" in error_msg:
            feedback_context += "CRITICAL RULE VIOLATION: 'tt.addptr' MUST return EXACTLY the same type as its pointer operand! If your input pointer is 'tensor<...x!tt.ptr<f32>>', your 'out_type' MUST also be exactly 'tensor<...x!tt.ptr<f32>>'. Do not change the type or shape!\n"
        if "Malformed JSON:" in error_msg or "JSONDecodeError" in error_msg or "Expecting value:" in error_msg or "Unterminated string" in error_msg:
            feedback_context += "CRITICAL RULE VIOLATION: The JSON is invalid or truncated. This happens when you hit the token limit! You MUST be more concise, use 'scf.for' loops instead of unrolling manually, DO NOT generate redundant operations or duplicate constants, and ensure the JSON is fully closed.\n"

        current_user_prompt = base_user_prompt + feedback_context + "\nAnalyze ALL past errors, correct the logical flaw in the JSON, and output the fixed version."

    if tracker: tracker.finish()
    print("\nProcess finished.")

if __name__ == "__main__":
    main()
