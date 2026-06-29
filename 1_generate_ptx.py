import os
import re
import sys
import json
import argparse
import traceback
import requests
import time

from core.llm_client import generate_llm_response
from core.schemas import MlirResponse
from core.semantic_validator import SemanticValidator
from core.mlir_translator import MLIRTranslator
from core.prompt_builder import PromptBuilder
from core.feedback_engine import generate_feedback_prompt

try:
    from core.triton_backend import TritonBackend, HAS_TRITON
except ImportError:
    HAS_TRITON = False

def extract_json(raw: str) -> str:
    s = raw.strip()
    m = re.search(r"```json\s*(.*?)\s*```", s, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    s = re.sub(r"```.*?```", "", s, flags=re.DOTALL).strip()
    m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    return m.group(1).strip() if m else s

def parse_wrapper_info(instruction: str) -> str:
    return instruction.split("Wrapper Entry Information:")[-1].strip()

def generate_one(model: str, instruction: str, func_name: str, max_retries: int,
                 validator: SemanticValidator, translator: MLIRTranslator, backend: TritonBackend,
                 builder: PromptBuilder, output_dir: str) -> tuple[bool, dict]:
    base_user = instruction
    system_prompt = builder.build_prompt(base_user, MlirResponse.model_json_schema())
    user_prompt = base_user
    error_history = ""
    last_err = ""

    stats = {
        "function_name": func_name,
        "success": False,
        "attempts": 0,
        "failure_stage": None,
        "errors_by_attempt": []
    }

    for attempt in range(max_retries):
        stats["attempts"] = attempt + 1
        print(f"  -> Attempt {attempt + 1}/{max_retries}...")
        try:
            stats["failure_stage"] = "LLM Generation"
            raw = generate_llm_response(model, system_prompt, user_prompt, schema=MlirResponse)
            
            stats["failure_stage"] = "JSON Parsing"
            clean_json = extract_json(raw)
            obj = MlirResponse(**json.loads(clean_json))

            stats["failure_stage"] = "Semantic Validation"
            errs = validator.validate(obj)
            if errs:
                print(f"     [Semantic Error] {len(errs)} rule violations detected. Feeding back to LLM...")
                error_msg = "\n".join(errs)
                error_history += f"\n- Attempt {attempt+1} errors:\n{error_msg}\n"
                last_err = error_msg
                
                stats["errors_by_attempt"].append({"attempt": attempt + 1, "stage": "Semantic Validation", "errors": errs})
                
                snippet = raw if len(raw) < 1000 else raw[:500] + "\n...[TRUNCATED]...\n" + raw[-500:]
                feedback = generate_feedback_prompt(error_msg, error_history, snippet, is_semantic=True)
                user_prompt = base_user + feedback
                continue

            print("     [MLIR] Semantics passed. Lowering to TTIR...")
            stats["failure_stage"] = "TTIR Translation"
            ttir_code = translator.translate_to_module(obj.code)
            
            if backend:
                print("     [PTX] Compiling TTIR to PTX...")
                stats["failure_stage"] = "PTX Compilation"
                ptx_code = backend.compile_ttir_to_ptx(ttir_code)
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f"{func_name}.ptx"), "w") as f:
                    f.write(ptx_code)
                with open(os.path.join(output_dir, f"{func_name}.json"), "w") as f:
                    json.dump({"instruction": instruction, "mlir_args": [a.name for a in obj.code.arguments]}, f)
                print(f"     [SUCCESS] {func_name}.ptx generated!")
                stats["success"] = True
                stats["failure_stage"] = None
                return True, stats
            else:
                print("     [ERROR] No TritonBackend available!")
                stats["errors_by_attempt"].append({"attempt": attempt + 1, "stage": "Environment", "errors": ["No TritonBackend available"]})
                return False, stats

        except requests.exceptions.RequestException as e:
            print(f"  -> Remote server error: {e}")
            stats["errors_by_attempt"].append({"attempt": attempt + 1, "stage": "Network", "errors": [str(e)]})
            print("     Retrying after 10 seconds...")
            time.sleep(10)
            if attempt == max_retries - 1:
                last_err = str(e)
            continue
            
        except TimeoutError as e:
            print(f"  -> Job timed out: {e}")
            stats["errors_by_attempt"].append({"attempt": attempt + 1, "stage": "Network", "errors": ["TimeoutError"]})
            last_err = str(e)
            break

        except Exception as e:
            error_str = str(e)
            if "CUDA out of memory" in error_str:
                print("  -> CUDA OOM, aborting prompt")
                stats["errors_by_attempt"].append({"attempt": attempt + 1, "stage": "PTX Compilation", "errors": ["CUDA OOM"]})
                return False, stats
                
            print(f"     [Compilation Error] Pipeline crashed. Feeding back to LLM...")
            error_history += f"\n- Attempt {attempt+1} errors:\n[PYTHON EXCEPTION]: {error_str}\n"
            last_err = error_str
            
            stats["errors_by_attempt"].append({"attempt": attempt + 1, "stage": stats.get("failure_stage", "Unknown"), "errors": [error_str]})
            
            code_json_str = ""
            try:
                if 'clean_json' in locals():
                    parsed = json.loads(clean_json)
                    if "code" in parsed:
                        code_json_str = json.dumps({"code": parsed["code"]}, indent=2)
            except:
                pass
                
            snippet = code_json_str if len(code_json_str) < 500 else code_json_str[:250] + "\n...[TRUNCATED]...\n" + code_json_str[-250:]
            feedback = generate_feedback_prompt(error_str, error_history, snippet, is_semantic=False)
            user_prompt = base_user + feedback

    print(f"  -> [FAILED] Failed to generate {func_name} after {max_retries} attempts.")
    return False, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpaca", required=True)
    ap.add_argument("--out_dir", default="benchmark_ptx_output")
    ap.add_argument("--model", default="gemini")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--start", type=int, default=0, help="Start index")
    ap.add_argument("--limit", type=int, default=0, help="Number of items to process")
    args = ap.parse_args()

    full_alpaca = json.load(open(args.alpaca, "r", encoding="utf-8"))
    
    start_idx = args.start
    end_idx = start_idx + args.limit if args.limit > 0 else len(full_alpaca)
    alpaca = full_alpaca[start_idx:end_idx]

    validator = SemanticValidator()
    translator = MLIRTranslator()
    backend = TritonBackend() if HAS_TRITON else None
    builder = PromptBuilder()

    log_file_name = f"generation_stats_{start_idx}_to_{end_idx}.jsonl" if (args.start > 0 or args.limit > 0) else "generation_stats.jsonl"
    log_file_path = os.path.join(args.out_dir, log_file_name)
    os.makedirs(args.out_dir, exist_ok=True)
    # File clearing removed to safely append if ran multiple times

    n_ok = 0
    for i, entry in enumerate(alpaca):
        instruction = entry["instruction"]
        wrapper_info = parse_wrapper_info(instruction)
        func_name = wrapper_info.split("(")[0].strip()
        print(f"[{start_idx + i + 1}/{len(full_alpaca)}] generating PTX for {func_name}...", flush=True)
        
        success, stats = generate_one(args.model, instruction, func_name, args.max_retries,
                                      validator, translator, backend, builder, args.out_dir)
        
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
            
        if success:
            n_ok += 1

    print(f"\nDone. Successfully generated {n_ok}/{len(alpaca)} PTX kernels in {args.out_dir}/.")
    print(f"Statistics and logs saved to: {log_file_path}")

if __name__ == "__main__":
    main()
