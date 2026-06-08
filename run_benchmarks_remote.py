"""
run_benchmarks_remote.py

Same pipeline as the repo's run_benchmarks.py (3-attempt generate -> validate ->
translate -> compile, with error feedback between attempts), adapted to the
decoupled setup: generation and compilation/benchmarking happen on the T4 over
the ngrok endpoint, translation and semantic validation stay local on the Mac.

Run from the repo ROOT, venv active:
    python run_benchmarks_remote.py

Requires in .env:
    USE_REMOTE_MODEL=1
    GEMMA_API_URL=<the PUBLIC URL your Kaggle notebook printed>
"""
import os
import json
import requests

import core.config as config
from core.llm_client import generate_llm_response
from core.schemas import MlirResponse
from core.mlir_translator import MLIRTranslator
from core.prompt_builder import PromptBuilder
from core.semantic_validator import SemanticValidator

MAX_RETRIES = 3
BENCH_URL = config.REMOTE_MODEL_URL.rstrip("/") + "/benchmark"
HEADERS = {"ngrok-skip-browser-warning": "true"}


def clean_json(raw):
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def build_feedback(error_msg, error_history, last_code_json):
    """Targeted feedback rules — ported from the repo's run_benchmarks.py,
    plus the compile errors that now arrive from the /benchmark endpoint."""
    fb = f"\n\n--- PREVIOUS ATTEMPTS HISTORY ---{error_history}\n"
    fb += f"\nIn your last attempt, you generated this code:\n```json\n{last_code_json}\n```\n\n"

    e = error_msg or ""
    if "scf.for loop defines" in e and "iter_args but returns" in e:
        fb += "CRITICAL RULE: The number of 'results' in scf.for MUST EXACTLY MATCH the number of 'iter_args'.\n"
    if "missing an scf.yield operation" in e:
        fb += "CRITICAL RULE: The LAST operation inside a 'scf.for' or 'scf.if' body MUST be 'scf.yield'.\n"
    if "requires a pointer or tensor of pointers" in e:
        fb += "CRITICAL RULE: 'tt.load'/'tt.store' MUST receive a pointer. Broadcast a base pointer with 'tt.splat' then add offsets with 'tt.addptr'.\n"
    if "used in 'scf.yield' but was never defined in this scope" in e:
        fb += "CRITICAL RULE VIOLATION: Inside the loop body, yield the NEWLY computed values for this iteration, not the loop's final result registers.\n"
    elif "not found in environment" in e or "never defined in this scope" in e:
        fb += "CRITICAL RULE VIOLATION: You used a register that does not exist. Every operand must be the 'result' of a previous operation.\n"
    if "requires an 'axis' attribute" in e:
        fb += "CRITICAL RULE: 'tt.reduce' MUST have an 'axis' attribute (e.g. {\"axis\": 0}).\n"
    if "incorrect number of indices for extract_element" in e:
        fb += "CRITICAL RULE VIOLATION: Do NOT use 'tensor.extract' to slice rows. Use Triton pointer arithmetic (make_range/splat/addptr/load).\n"
    if "must be floating-point-like, but got '!tt.ptr<f32>'" in e:
        fb += "CRITICAL RULE VIOLATION: You did math on POINTERS. Add explicit 'out_type': 'tensor<...xf32>' to 'tt.load' so it returns data, not a pointer.\n"
    if "failed to verify that result type matches ptr type" in e:
        fb += "CRITICAL RULE VIOLATION: 'tt.addptr' out_type MUST equal its pointer operand's type exactly (same shape and !tt.ptr<...>).\n"
    if "operand #1 must be 1-bit signless integer" in e and "tt.load" in e:
        fb += "CRITICAL RULE VIOLATION: The mask operand of tt.load/tt.store must be a boolean tensor (i1). Build it with 'arith.cmpi'.\n"
    if "'tt.addptr' op operand #0 must be ptr" in e:
        fb += "CRITICAL RULE VIOLATION: The first operand of 'tt.addptr' must be a pointer, not a loaded value.\n"
    # compile error we are hitting now, surfaced from the endpoint:
    if "make_range" in e and "'end'" in e:
        fb += ("CRITICAL RULE: 'tt.make_range' REQUIRES 'start' AND 'end' inside 'attributes', "
               "e.g. {\"opcode\":\"tt.make_range\",\"operands\":[],\"attributes\":{\"start\":0,\"end\":256},"
               "\"result\":\"%offsets\",\"out_type\":\"tensor<256xi32>\"}. Do not omit them.\n")
    if "literal_error" in e or "validation errors for MlirResponse" in e:
        fb += "CRITICAL RULE VIOLATION: Pydantic schema failed. Include 'operands': [] even for zero-operand ops, and use exact MLIR type strings.\n"

    # --- arith.constant missing value (his rule, yours was missing it) ---
    if "attributes.value" in e or ("arith.constant" in e and "value" in e):
        fb += ("CRITICAL RULE: 'arith.constant' MUST include a 'value' in attributes, "
               "e.g. {\"opcode\":\"arith.constant\",\"attributes\":{\"value\":0.0},"
               "\"result\":\"%zero\",\"out_type\":\"f32\"}.\n")

    # --- tensor-vs-scalar type mismatch in binary math (your relu failure) ---
    if "EXACT SAME type/shape" in e or "requires operands of the EXACT SAME" in e:
        fb += ("CRITICAL RULE: Binary ops like 'arith.maximumf'/'arith.addf' need BOTH operands "
               "to be the SAME shape. To use a scalar constant with a tensor, FIRST create the "
               "constant with 'arith.constant', THEN broadcast it to a tensor with 'tt.splat' "
               "(out_type 'tensor<256xf32>'), and only then apply the math op.\n")

    # --- all make_range variants, not just the 'end' one ---
    if "make_range" in e and ("zero regions" in e or "ranked tensor of integer" in e or "got 'index'" in e):
        fb += ("CRITICAL RULE: 'tt.make_range' takes EXACTLY 0 operands (\"operands\": []), needs "
               "'start' and 'end' in attributes, and its 'out_type' MUST be an integer tensor like "
               "'tensor<256xi32>' (NOT 'index', NOT a region). Example: "
               "{\"opcode\":\"tt.make_range\",\"operands\":[],\"attributes\":{\"start\":0,\"end\":256},"
               "\"result\":\"%off\",\"out_type\":\"tensor<256xi32>\"}.\n")

    fb += "\nAnalyze ALL past errors, correct your JSON, and ensure strict compliance with the MLIR rules."


    return fb


def run():
    print("=== Remote LLM-MLIR Benchmark Runner (repo pipeline, 3 attempts) ===")
    print("benchmark endpoint:", BENCH_URL)

    if not os.path.exists("benchmark_prompts.json"):
        print("ERROR: benchmark_prompts.json not found in repo root."); return
    benchmarks = json.load(open("benchmark_prompts.json"))

    validator = SemanticValidator()
    translator = MLIRTranslator()
    prompt_builder = PromptBuilder()
    schema = MlirResponse.model_json_schema()

    results = {}

    for name, data in benchmarks.items():
        difficulty = data.get("difficulty", "?")
        prompt_text = data["prompt"]
        print(f"\n[{name}] (difficulty: {difficulty})")

        base_user_prompt = f"Implement the following Triton kernel logic: {prompt_text}"
        current_user_prompt = base_user_prompt
        system_prompt = prompt_builder.build_prompt(base_user_prompt, schema)

        success = False
        error_history = ""
        last_row = None

        for attempt in range(MAX_RETRIES):
            print(f"  [attempt {attempt+1}/{MAX_RETRIES}] generating via remote LLM...")
            error_msg = None
            last_code_json = "{}"

            # 1. GENERATE (remote, ChatML-wrapped inside run_remote)
            try:
                raw = generate_llm_response("ollama", system_prompt, current_user_prompt, schema=MlirResponse)
            except Exception as e:
                print(f"    -> generation request failed: {e}")
                last_row = {"name": name, "status": "gen_request_failed", "error": str(e)}
                error_msg = str(e)
                error_history += f"\n- Attempt {attempt+1}: {error_msg}\n"
                current_user_prompt = base_user_prompt + build_feedback(error_msg, error_history, last_code_json)
                continue

            # 2. PARSE + 3. SEMANTIC VALIDATION (local)
            try:
                cj = clean_json(raw)
                parsed = json.loads(cj)
                last_code_json = json.dumps({"code": parsed.get("code")}, indent=2)
                obj = MlirResponse(**parsed)
                sem_errors = validator.validate(obj)
                if sem_errors:
                    raise RuntimeError("Semantic errors:\n" + "\n".join(sem_errors))
                # 4. TRANSLATE (local)
                ttir = translator.translate_to_module(obj.code)
            except Exception as e:
                error_msg = str(e)
                print(f"    -> local validation/translation failed:\n{error_msg[:300]}")
                last_row = {"name": name, "status": "local_failed", "error": error_msg}
                error_history += f"\n- Attempt {attempt+1}: {error_msg}\n"
                current_user_prompt = base_user_prompt + build_feedback(error_msg, error_history, last_code_json)
                continue

            # 5. COMPILE + LAUNCH + BENCHMARK (remote on T4)
            try:
                resp = requests.post(BENCH_URL, json={"name": name, "ttir": ttir},
                                     headers=HEADERS, timeout=180)
                row = resp.json()
            except Exception as e:
                error_msg = f"benchmark endpoint error: {e}"
                print(f"    -> {error_msg}")
                last_row = {"name": name, "status": "endpoint_unreachable", "error": str(e)}
                error_history += f"\n- Attempt {attempt+1}: {error_msg}\n"
                current_user_prompt = base_user_prompt + build_feedback(error_msg, error_history, last_code_json)
                continue

            last_row = row
            if row.get("status") == "success":
                row["attempts"] = attempt + 1
                results[name] = row
                success = True
                print(f"    -> SUCCESS  correct={row['correct']}  speedup={row.get('speedup')}")
                break

            # not success: feed the remote error back into the loop
            error_msg = row.get("error") or row.get("status")
            print(f"    -> {row.get('status')}: {str(error_msg)[:300]}")
            error_history += f"\n- Attempt {attempt+1}: {error_msg}\n"
            current_user_prompt = base_user_prompt + build_feedback(error_msg, error_history, last_code_json)

        if not success:
            last_row = last_row or {"name": name, "status": "failed_after_retries"}
            last_row["attempts"] = MAX_RETRIES
            results[name] = last_row

    # persist + summary
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Benchmark Summary ===")
    for k, v in results.items():
        print(f"{k.ljust(18)}: {v.get('status')}  (attempts: {v.get('attempts', '?')}, "
              f"correct: {v.get('correct')}, speedup: {v.get('speedup')})")
    print("\nwrote results.json")


if __name__ == "__main__":
    run()
