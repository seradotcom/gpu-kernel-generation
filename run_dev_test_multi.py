"""
run_dev_test_multi.py

Dev mode: runs the full pipeline with multiple hardcoded sample JSONs.
No LLM calls, no WandB, no API keys needed.
Tests various kernel families and opcodes to see what compiles and runs.

Usage:
    python run_dev_test_multi.py

Or in Colab:
    !python /content/gpu-kernel-generation/run_dev_test_multi.py
"""

import json
import os
import traceback

import core.config  # noqa: F401 - Sets up MLIR bindings in sys.path

from core.schemas import MlirResponse
from core.semantic_validator import SemanticValidator
from core.mlir_translator import MLIRTranslator
from core.triton_python_generator import TritonPythonGenerator
from core.triton_executor import TritonExecutor


def test_kernel(name: str, sample_json: dict, n_elements: int = 1024) -> dict:
    print(f"\n{'='*60}")
    print(f"KERNEL: {name}")
    print(f"{'='*60}")

    result = {
        "name": name,
        "pydantic_ok": False,
        "semantic_ok": False,
        "mlir_ok": False,
        "triton_gen_ok": False,
        "exec_ok": False,
        "correct": None,
        "speedup": None,
        "error": None,
    }

    # Stage 1: Parse JSON
    print("[1/5] Parsing JSON...")
    try:
        mlir_obj = MlirResponse(**sample_json)
        print("    -> Pydantic PASSED")
        result["pydantic_ok"] = True
    except Exception as e:
        print(f"    -> Pydantic FAILED: {e}")
        result["error"] = f"Pydantic: {e}"
        return result

    # Stage 2: Semantic Validation
    print("[2/5] Semantic validation...")
    validator = SemanticValidator()
    semantic_errors = validator.validate(mlir_obj)
    if semantic_errors:
        print(f"    -> Semantic FAILED: {semantic_errors}")
        result["error"] = f"Semantic: {semantic_errors}"
        return result
    print("    -> Semantic PASSED")
    result["semantic_ok"] = True

    # Stage 3: MLIR Translation
    print("[3/5] MLIR translation...")
    try:
        translator = MLIRTranslator()
        mlir_code = translator.translate_to_module(mlir_obj.code)
        print("    -> MLIR PASSED")
        result["mlir_ok"] = True
    except Exception as e:
        print(f"    -> MLIR FAILED: {e}")
        result["error"] = f"MLIR: {e}"
        return result

    # Stage 4: Triton Python Generation
    print("[4/5] Triton Python generation...")
    try:
        gen = TritonPythonGenerator()
        triton_python = gen.generate(mlir_obj.code)
        print("    -> Triton Python PASSED")
        print(f"\n--- Generated Triton Python ---\n{triton_python}\n---")
        result["triton_gen_ok"] = True
    except Exception as e:
        print(f"    -> Triton Python FAILED: {traceback.format_exc()}")
        result["error"] = f"TritonGen: {traceback.format_exc()}"
        return result

    # Stage 5: Compile and Execute
    print("[5/5] Compile and execute...")
    try:
        executor = TritonExecutor()
        exec_result = executor.run(triton_python, n_elements=n_elements, warmup=2, reps=10)
        if exec_result["success"]:
            print(f"    -> Execution PASSED")
            print(f"    -> Correct: {exec_result['correct']}")
            print(f"    -> Speedup: {exec_result.get('speedup', 'N/A')}x")
            result["exec_ok"] = True
            result["correct"] = exec_result["correct"]
            result["speedup"] = exec_result.get("speedup")
        else:
            print(f"    -> Execution FAILED")
            print(f"    -> Error: {exec_result['error'][:500]}")
            result["error"] = exec_result["error"]
    except Exception as e:
        print(f"    -> Execution FAILED: {traceback.format_exc()}")
        result["error"] = traceback.format_exc()

    return result


def main():
    print("=== MULTI-KERNEL DEV TEST ===")
    print("Testing pipeline on various kernel families without LLM calls.\n")

    kernels = {}

    # Kernel 1: Vector Add (element-wise, Phase 1)
    kernels["vector_add"] = {
        "reasoning": "A + B = C",
        "code": {
            "function_name": "vec_sum_kernel",
            "arguments": [
                {"name": "%arg0_A", "type": "!tt.ptr<f32>"},
                {"name": "%arg1_B", "type": "!tt.ptr<f32>"},
                {"name": "%arg2_C", "type": "!tt.ptr<f32>"}
            ],
            "operations": [
                {"opcode": "tt.make_range", "operands": [], "attributes": {"start": 0, "end": 256}, "result": "%offsets", "out_type": "tensor<256xi32>"},
                {"opcode": "tt.splat", "operands": ["%arg0_A"], "result": "%ptrs_A", "out_type": "tensor<256x!tt.ptr<f32>>"},
                {"opcode": "tt.addptr", "operands": ["%ptrs_A", "%offsets"], "result": "%ptrs_A_off", "out_type": "tensor<256x!tt.ptr<f32>>"},
                {"opcode": "tt.load", "operands": ["%ptrs_A_off"], "result": "%val_A", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.splat", "operands": ["%arg1_B"], "result": "%ptrs_B", "out_type": "tensor<256x!tt.ptr<f32>>"},
                {"opcode": "tt.addptr", "operands": ["%ptrs_B", "%offsets"], "result": "%ptrs_B_off", "out_type": "tensor<256x!tt.ptr<f32>>"},
                {"opcode": "tt.load", "operands": ["%ptrs_B_off"], "result": "%val_B", "out_type": "tensor<256xf32>"},
                {"opcode": "arith.addf", "operands": ["%val_A", "%val_B"], "result": "%val_C", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.splat", "operands": ["%arg2_C"], "result": "%ptrs_C", "out_type": "tensor<256x!tt.ptr<f32>>"},
                {"opcode": "tt.addptr", "operands": ["%ptrs_C", "%offsets"], "result": "%ptrs_C_off", "out_type": "tensor<256x!tt.ptr<f32>>"},
                {"opcode": "tt.store", "operands": ["%ptrs_C_off", "%val_C"], "result": "none"}
            ],
            "returns": []
        }
    }

    # Kernel 2: Element-wise Mul (Phase 1)
    kernels["element_mul"] = {
        "reasoning": "A * B = C",
        "code": {
            "function_name": "element_mul_kernel",
            "arguments": [
                {"name": "%A_ptr", "type": "!tt.ptr<f32>"},
                {"name": "%B_ptr", "type": "!tt.ptr<f32>"},
                {"name": "%C_ptr", "type": "!tt.ptr<f32>"}
            ],
            "operations": [
                {"opcode": "tt.make_range", "operands": [], "attributes": {"start": 0, "end": 256}, "result": "%off", "out_type": "tensor<256xi32>"},
                {"opcode": "tt.load", "operands": ["%A_ptr", "%off"], "result": "%a", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.load", "operands": ["%B_ptr", "%off"], "result": "%b", "out_type": "tensor<256xf32>"},
                {"opcode": "arith.mulf", "operands": ["%a", "%b"], "result": "%c", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.store", "operands": ["%C_ptr", "%off", "%c"], "result": "none"}
            ],
            "returns": []
        }
    }

    # Kernel 3: Element-wise Sqrt (Phase 1 + math.sqrt)
    kernels["element_sqrt"] = {
        "reasoning": "sqrt(A) = B",
        "code": {
            "function_name": "element_sqrt_kernel",
            "arguments": [
                {"name": "%A_ptr", "type": "!tt.ptr<f32>"},
                {"name": "%B_ptr", "type": "!tt.ptr<f32>"}
            ],
            "operations": [
                {"opcode": "tt.make_range", "operands": [], "attributes": {"start": 0, "end": 256}, "result": "%off", "out_type": "tensor<256xi32>"},
                {"opcode": "tt.load", "operands": ["%A_ptr", "%off"], "result": "%a", "out_type": "tensor<256xf32>"},
                {"opcode": "math.sqrt", "operands": ["%a"], "result": "%b", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.store", "operands": ["%B_ptr", "%off", "%b"], "result": "none"}
            ],
            "returns": []
        }
    }

    # Kernel 4: Element-wise Max (Phase 1 + arith.maximumf)
    kernels["element_max"] = {
        "reasoning": "max(A, B) = C",
        "code": {
            "function_name": "element_max_kernel",
            "arguments": [
                {"name": "%A_ptr", "type": "!tt.ptr<f32>"},
                {"name": "%B_ptr", "type": "!tt.ptr<f32>"},
                {"name": "%C_ptr", "type": "!tt.ptr<f32>"}
            ],
            "operations": [
                {"opcode": "tt.make_range", "operands": [], "attributes": {"start": 0, "end": 256}, "result": "%off", "out_type": "tensor<256xi32>"},
                {"opcode": "tt.load", "operands": ["%A_ptr", "%off"], "result": "%a", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.load", "operands": ["%B_ptr", "%off"], "result": "%b", "out_type": "tensor<256xf32>"},
                {"opcode": "arith.maximumf", "operands": ["%a", "%b"], "result": "%c", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.store", "operands": ["%C_ptr", "%off", "%c"], "result": "none"}
            ],
            "returns": []
        }
    }

    # Kernel 5: Scale and Add (A * scale + B = C, tests arith.constant + tt.splat)
    kernels["scale_add"] = {
        "reasoning": "A * 2.0 + B = C",
        "code": {
            "function_name": "scale_add_kernel",
            "arguments": [
                {"name": "%A_ptr", "type": "!tt.ptr<f32>"},
                {"name": "%B_ptr", "type": "!tt.ptr<f32>"},
                {"name": "%C_ptr", "type": "!tt.ptr<f32>"}
            ],
            "operations": [
                {"opcode": "tt.make_range", "operands": [], "attributes": {"start": 0, "end": 256}, "result": "%off", "out_type": "tensor<256xi32>"},
                {"opcode": "tt.load", "operands": ["%A_ptr", "%off"], "result": "%a", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.load", "operands": ["%B_ptr", "%off"], "result": "%b", "out_type": "tensor<256xf32>"},
                {"opcode": "arith.constant", "operands": [], "attributes": {"value": 2.0}, "result": "%scale", "out_type": "f32"},
                {"opcode": "tt.splat", "operands": ["%scale"], "result": "%scale_t", "out_type": "tensor<256xf32>"},
                {"opcode": "arith.mulf", "operands": ["%a", "%scale_t"], "result": "%scaled", "out_type": "tensor<256xf32>"},
                {"opcode": "arith.addf", "operands": ["%scaled", "%b"], "result": "%c", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.store", "operands": ["%C_ptr", "%off", "%c"], "result": "none"}
            ],
            "returns": []
        }
    }

    # Kernel 6: Simple scf.for loop (sum reduction pattern)
    kernels["simple_loop"] = {
        "reasoning": "Loop to demonstrate scf.for",
        "code": {
            "function_name": "simple_loop_kernel",
            "arguments": [
                {"name": "%A_ptr", "type": "!tt.ptr<f32>"},
                {"name": "%B_ptr", "type": "!tt.ptr<f32>"}
            ],
            "operations": [
                {"opcode": "tt.make_range", "operands": [], "attributes": {"start": 0, "end": 256}, "result": "%off", "out_type": "tensor<256xi32>"},
                {"opcode": "tt.load", "operands": ["%A_ptr", "%off"], "result": "%a", "out_type": "tensor<256xf32>"},
                {"opcode": "arith.constant", "operands": [], "attributes": {"value": 0.0}, "result": "%zero", "out_type": "f32"},
                {"opcode": "tt.splat", "operands": ["%zero"], "result": "%zero_t", "out_type": "tensor<256xf32>"},
                {"opcode": "scf.for", "lower_bound": 0, "upper_bound": 4, "step": 1, "loop_var": "%i",
                 "iter_args": {"%acc": "%zero_t"}, "results": ["%final_acc"],
                 "body": [
                     {"opcode": "arith.addf", "operands": ["%acc", "%a"], "result": "%new_acc", "out_type": "tensor<256xf32>"},
                     {"opcode": "scf.yield", "operands": ["%new_acc"]}
                 ]},
                {"opcode": "tt.store", "operands": ["%B_ptr", "%off", "%final_acc"], "result": "none"}
            ],
            "returns": []
        }
    }

    # Kernel 7: Masked load (arith.cmpf for boundary check)
    kernels["masked_add"] = {
        "reasoning": "A + B = C with boundary masking",
        "code": {
            "function_name": "masked_add_kernel",
            "arguments": [
                {"name": "%A_ptr", "type": "!tt.ptr<f32>"},
                {"name": "%B_ptr", "type": "!tt.ptr<f32>"},
                {"name": "%C_ptr", "type": "!tt.ptr<f32>"}
            ],
            "operations": [
                {"opcode": "tt.make_range", "operands": [], "attributes": {"start": 0, "end": 256}, "result": "%off", "out_type": "tensor<256xi32>"},
                {"opcode": "arith.constant", "operands": [], "attributes": {"value": 1024}, "result": "%limit", "out_type": "i32"},
                {"opcode": "tt.splat", "operands": ["%limit"], "result": "%limit_t", "out_type": "tensor<256xi32>"},
                {"opcode": "arith.cmpi", "operands": ["%off", "%limit_t"], "result": "%mask", "out_type": "tensor<256xi1>", "attributes": {"predicate": 4}},
                {"opcode": "tt.load", "operands": ["%A_ptr", "%off", "%mask"], "result": "%a", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.load", "operands": ["%B_ptr", "%off", "%mask"], "result": "%b", "out_type": "tensor<256xf32>"},
                {"opcode": "arith.addf", "operands": ["%a", "%b"], "result": "%c", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.store", "operands": ["%C_ptr", "%off", "%c", "%mask"], "result": "none"}
            ],
            "returns": []
        }
    }

    # Kernel 8: Element-wise Exp (math.exp)
    kernels["element_exp"] = {
        "reasoning": "exp(A) = B",
        "code": {
            "function_name": "element_exp_kernel",
            "arguments": [
                {"name": "%A_ptr", "type": "!tt.ptr<f32>"},
                {"name": "%B_ptr", "type": "!tt.ptr<f32>"}
            ],
            "operations": [
                {"opcode": "tt.make_range", "operands": [], "attributes": {"start": 0, "end": 256}, "result": "%off", "out_type": "tensor<256xi32>"},
                {"opcode": "tt.load", "operands": ["%A_ptr", "%off"], "result": "%a", "out_type": "tensor<256xf32>"},
                {"opcode": "math.exp", "operands": ["%a"], "result": "%b", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.store", "operands": ["%B_ptr", "%off", "%b"], "result": "none"}
            ],
            "returns": []
        }
    }

    # Kernel 9: Conditional Select (arith.select + arith.cmpf)
    kernels["conditional_select"] = {
        "reasoning": "C = A > 0.0 ? A : 0.0",
        "code": {
            "function_name": "conditional_select_kernel",
            "arguments": [
                {"name": "%A_ptr", "type": "!tt.ptr<f32>"},
                {"name": "%C_ptr", "type": "!tt.ptr<f32>"}
            ],
            "operations": [
                {"opcode": "tt.make_range", "operands": [], "attributes": {"start": 0, "end": 256}, "result": "%off", "out_type": "tensor<256xi32>"},
                {"opcode": "tt.load", "operands": ["%A_ptr", "%off"], "result": "%a", "out_type": "tensor<256xf32>"},
                {"opcode": "arith.constant", "operands": [], "attributes": {"value": 0.0}, "result": "%zero", "out_type": "f32"},
                {"opcode": "tt.splat", "operands": ["%zero"], "result": "%zero_t", "out_type": "tensor<256xf32>"},
                {"opcode": "arith.cmpf", "operands": ["%a", "%zero_t"], "result": "%mask", "out_type": "tensor<256xi1>", "attributes": {"predicate": 1}},
                {"opcode": "arith.select", "operands": ["%mask", "%a", "%zero_t"], "result": "%c", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.store", "operands": ["%C_ptr", "%off", "%c"], "result": "none"}
            ],
            "returns": []
        }
    }

    # Kernel 10: Tensor Sum Reduction (tt.reduce)
    kernels["sum_reduction"] = {
        "reasoning": "Sum elements of A into scalar B",
        "code": {
            "function_name": "sum_reduction_kernel",
            "arguments": [
                {"name": "%A_ptr", "type": "!tt.ptr<f32>"},
                {"name": "%B_ptr", "type": "!tt.ptr<f32>"}
            ],
            "operations": [
                {"opcode": "tt.make_range", "operands": [], "attributes": {"start": 0, "end": 256}, "result": "%off", "out_type": "tensor<256xi32>"},
                {"opcode": "tt.load", "operands": ["%A_ptr", "%off"], "result": "%a", "out_type": "tensor<256xf32>"},
                {"opcode": "tt.reduce", "operands": ["%a"], "result": "%sum", "out_type": "f32", "attributes": {"axis": 0}, "region_combiner": "arith.addf"},
                {"opcode": "tt.store", "operands": ["%B_ptr", "%off", "%sum"], "result": "none"}
            ],
            "returns": []
        }
    }

    # Run all kernels
    results = []
    for name, json_data in kernels.items():
        r = test_kernel(name, json_data, n_elements=1024)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total = len(results)
    passed = sum(1 for r in results if r["exec_ok"])
    print(f"Total kernels: {total}")
    print(f"Passed execution: {passed}")
    print(f"Failed: {total - passed}\n")

    for r in results:
        status = "PASS" if r["exec_ok"] else "FAIL"
        stages = []
        if r["pydantic_ok"]: stages.append("P")
        if r["semantic_ok"]: stages.append("S")
        if r["mlir_ok"]: stages.append("M")
        if r["triton_gen_ok"]: stages.append("T")
        if r["exec_ok"]: stages.append("E")
        stage_str = "-".join(stages) if stages else "none"
        print(f"  {r['name']:<20} [{status}]  Stages: {stage_str}")
        if not r["exec_ok"] and r["error"]:
            err = r["error"].split("\n")[0][:80]
            print(f"    Error: {err}")

    # Save detailed results
    os.makedirs("output", exist_ok=True)
    with open("output/dev_test_multi_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: output/dev_test_multi_results.json")


if __name__ == "__main__":
    main()
