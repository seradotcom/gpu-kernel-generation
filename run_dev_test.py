"""
run_dev_test.py

Dev mode: runs the full pipeline with a hardcoded sample JSON.
No LLM calls, no WandB, no API keys needed.
Use this to verify the end-to-end pipeline works before adding LLM generation.

Usage:
    python run_dev_test.py

Or in Colab:
    !python /content/gpu-kernel-generation/run_dev_test.py
"""

import json
import os
import traceback

from core.schemas import MlirResponse
from core.semantic_validator import SemanticValidator
from core.mlir_translator import MLIRTranslator
from core.triton_python_generator import TritonPythonGenerator
from core.triton_executor import TritonExecutor


def run_dev_test():
    print("=== DEV TEST: Vector Add Kernel ===")
    print("No LLM calls. Testing full pipeline with hardcoded sample JSON.\n")
    
    # Hardcoded sample: vector add (A + B = C)
    sample_json = {
        "reasoning": "Element-wise addition of two vectors using pointer arithmetic.",
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
    
    # Setup output directory
    os.makedirs("output", exist_ok=True)
    
    # Stage 1: Parse JSON
    print("[1/5] Parsing sample JSON...")
    try:
        mlir_obj = MlirResponse(**sample_json)
        print("    -> Pydantic validation PASSED")
    except Exception as e:
        print(f"    -> FAILED: {e}")
        return False
    
    # Save parsed JSON
    with open("output/dev_test_parsed.json", "w") as f:
        json.dump(sample_json, f, indent=2)
    
    # Stage 2: Semantic Validation
    print("\n[2/5] Semantic validation...")
    validator = SemanticValidator()
    semantic_errors = validator.validate(mlir_obj)
    if semantic_errors:
        print(f"    -> FAILED: {semantic_errors}")
        return False
    print("    -> Semantic validation PASSED")
    
    # Stage 3: MLIR Translation
    print("\n[3/5] Translating to MLIR and verifying...")
    try:
        translator = MLIRTranslator()
        mlir_code = translator.translate_to_module(mlir_obj.code)
        print("    -> MLIR verification PASSED")
        print(f"\n=== MLIR OUTPUT ===\n{mlir_code}\n=== END MLIR ===\n")
        
        # Save MLIR
        with open("output/dev_test.mlir", "w") as f:
            f.write(mlir_code)
    except Exception as e:
        print(f"    -> FAILED: {traceback.format_exc()}")
        return False
    
    # Stage 4: Triton Python Generation
    print("\n[4/5] Generating Triton Python...")
    try:
        gen = TritonPythonGenerator()
        triton_python = gen.generate(mlir_obj.code)
        print("    -> Triton Python generation PASSED")
        print(f"\n=== TRITON PYTHON ===\n{triton_python}\n=== END TRITON PYTHON ===\n")
        
        # Save Triton Python
        with open("output/dev_test.py", "w") as f:
            f.write(triton_python)
    except Exception as e:
        print(f"    -> FAILED: {traceback.format_exc()}")
        return False
    
    # Stage 5: Compile and Benchmark
    print("\n[5/5] Compiling and benchmarking Triton kernel...")
    try:
        executor = TritonExecutor()
        result = executor.run(triton_python, n_elements=1024, warmup=2, reps=10)
        
        if result["success"]:
            print(f"    -> SUCCESS!")
            print(f"    -> Correct: {result['correct']}")
            print(f"    -> Speedup: {result.get('speedup', 'N/A')}x")
            print(f"    -> Kernel time: {result.get('kernel_time_ms', 'N/A')} ms")
            print(f"    -> Ref time: {result.get('ref_time_ms', 'N/A')} ms")
            return True
        else:
            print(f"    -> FAILED: {result['error']}")
            with open("output/dev_test_error.txt", "w") as f:
                f.write(result["error"])
            return False
    except Exception as e:
        print(f"    -> FAILED: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = run_dev_test()
    if success:
        print("\n✅ DEV TEST PASSED: End-to-end pipeline works!")
    else:
        print("\n❌ DEV TEST FAILED: Check output/ directory for artifacts and error logs.")
