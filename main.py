import json
import traceback
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
    schema_str = json.dumps(MlirResponse.model_json_schema(), indent=2)
    
    system_prompt = (
        "You are an expert LLM compiler in Triton MLIR. You must output ONLY a valid JSON object "
        f"that EXACTLY complies with the following JSON Schema:\n{schema_str}\n\n"
        "STRICT RULES:\n"
        "1. DO NOT include markdown code blocks (e.g., ```json) or text outside the JSON.\n"
        "2. In 'arguments' and 'returns', use only strings (e.g., '%arg0'), NOT dictionaries.\n"
        "3. You MUST ONLY use the opcodes provided in the JSON Schema Enum.\n"
        "4. In iter_args and operands, you MUST ONLY use valid registers (e.g., '%0'). NEVER use raw numbers (e.g., '0.0'). If you need a constant, create it first with 'arith.constant'.\n"
    )
    
    user_prompt = (
        "Generate a kernel that performs one-pass online softmax. "
        "Goal: compute the softmax operation across the rows of an input matrix in a single global memory pass, "
        "using an online reduction algorithm to dynamically update the local maximum and the sum of exponentials."
    )
    
    # Experiment parameters
    max_retries = 3
    success = False
    
    for attempt in range(max_retries):
        print(f"\n--- Attempt {attempt + 1}/{max_retries} ---")
        if tracker: tracker.start_timer()
        
        # Call the model using Constrained Decoding (XGrammar / Structured Outputs)
        print("Waiting for LLM generation (Constrained Decoding enabled)...")
        try:
            # Pass the actual Pydantic schema to the client (XGrammar at API level)
            raw_response = generate_llm_response("ollama", system_prompt, user_prompt, schema=MlirResponse)
        except Exception as e:
            print(f"[!] Error calling the API: {e}")
            break
            
        print("LLM responded. Parsing JSON contract...")
        mlir_code = ""
        error_msg = None
        
        try:
            # 1. Syntactic Validation (Almost always passes thanks to Constrained Decoding)
            # Limpieza básica por si el modelo remoto responde con bloques de markdown
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
            
            # 2. Translation to MLIR
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
        
        user_prompt += f"\n\nYour previous attempt failed with this error:\n{error_msg}\nCorrect the JSON to fix it."

    if tracker: tracker.finish()
    print("\nProcess finished.")

if __name__ == "__main__":
    main()
