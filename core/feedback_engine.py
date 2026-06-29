def generate_feedback_prompt(error_msg: str, error_history: str, snippet: str, is_semantic: bool) -> str:
    """
    Generates an intelligent feedback prompt by diagnosing compilation or validation errors
    and injecting critical rules for the LLM to learn from.
    """
    feedback = f"\n\n--- PREVIOUS ATTEMPTS HISTORY ---{error_history}\n"
    
    if is_semantic:
        feedback += f"\nIn your last attempt, you generated this JSON:\n{snippet}\n\n"
        if "scf.for loop defines" in error_msg and "iter_args but returns" in error_msg:
            feedback += "CRITICAL RULE: The number of 'results' in scf.for MUST EXACTLY MATCH the number of 'iter_args'.\n"
        if "missing an scf.yield operation" in error_msg:
            feedback += "CRITICAL RULE: The LAST operation inside a 'scf.for' or 'scf.if' body MUST be 'scf.yield'. Do NOT forget to add the yield operation.\n"
        if "Floating point offsets are invalid" in error_msg:
            feedback += "CRITICAL RULE VIOLATION: You used floating point offsets (f32) for 'tt.addptr'. Offsets MUST be integers (e.g. 'i32' or 'tensor<256xi32>'). Do NOT use arith.mulf or f32 variables to calculate pointer offsets!\n"
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
    else:
        # Pipeline / Compiler errors
        feedback += f"\nYou generated this code in the last attempt:\n```json\n{snippet}\n```\n\n"
        
        if "not found in environment" in error_msg:
            feedback += "CRITICAL RULE VIOLATION: You used a register that DOES NOT EXIST. Every operand must be the 'result' of a previous operation.\n"
        elif "attributes.value" in error_msg:
            feedback += "CRITICAL RULE: 'arith.constant' MUST have a 'value' field (e.g. \"value\": 0.0) so the compiler knows the numeric value.\n"
        if "must be floating-point-like, but got '!tt.ptr<f32>'" in error_msg:
            feedback += "CRITICAL RULE VIOLATION: The compiler failed because you tried to do Math (like arith.addf) on POINTERS. This happened because you forgot to add explicit 'out_type': 'tensor<...xf32>' to your 'tt.load' operation, so the compiler assumed it returned a pointer instead of math data.\n"
        if "literal_error" in error_msg or "validation errors for MlirResponse" in error_msg:
            feedback += "CRITICAL RULE VIOLATION: Pydantic Schema Validation Failed. Make sure you included 'operands': [] even if the operation takes no operands (like tt.make_range), and ensure your 'out_type' strictly follows the MLIR syntax.\n"
        if "failed to verify that result type matches ptr type" in error_msg and "'tt.addptr'" in error_msg:
            feedback += "CRITICAL RULE VIOLATION: 'tt.addptr' MUST return EXACTLY the same type as its pointer operand! If your input pointer is 'tensor<...x!tt.ptr<f32>>', your 'out_type' MUST also be exactly 'tensor<...x!tt.ptr<f32>>'. Do not change the type or shape!\n"
        if "failed to verify that result matches ptr type" in error_msg and "'tt.load'" in error_msg:
            feedback += "CRITICAL RULE VIOLATION: 'tt.load' result type MUST exactly match the underlying type of the pointer! If your pointer is 'tensor<256x!tt.ptr<f32>>', the 'out_type' of tt.load MUST be 'tensor<256xf32>'. Do not output 'f32' for a tensor pointer, and do not output a tensor for a scalar pointer!\n"
        if "operand #1 must be 1-bit signless integer" in error_msg and "tt.load" in error_msg:
            feedback += "CRITICAL RULE VIOLATION: The mask operand (operand #1) of 'tt.load' or 'tt.store' MUST be a boolean tensor (i1), e.g., 'tensor<1024xi1>'. You passed an i32 tensor instead. Use 'arith.cmpi' to create a boolean mask first!\n"
        if "requires the same type for all operands and results" in error_msg or "requires the same shape for all operands and results" in error_msg:
            feedback += "CRITICAL RULE VIOLATION: You attempted to perform an operation (like arith.addi, arith.muli, or tt.addptr) where the operands have DIFFERENT shapes or types (e.g. adding a scalar to a tensor). In MLIR/Triton, you MUST broadcast the scalar using 'tt.splat' so both operands have the exact same shape.\n"
        if "must be signless-non-zero-bitwidth-integer-like, but got" in error_msg or "must be floating-point-like, but got" in error_msg:
            feedback += "CRITICAL RULE VIOLATION: Type mismatch! You passed a floating-point (f32) value to an integer operation (like arith.muli/addi) OR an integer (i32) to a floating-point operation (like arith.mulf/addf). Check your operand types and arith.constant values.\n"
        if "'tt.addptr' op operand #0 must be ptr" in error_msg:
            feedback += "CRITICAL RULE VIOLATION: The first operand of 'tt.addptr' MUST be a pointer (e.g. '!tt.ptr<f32>'). You passed a math tensor (like 'f32'). You must pass a base pointer, NOT a loaded value!\n"
        if "JSONDecodeError" in error_msg or "Expecting value:" in error_msg or "Unterminated string" in error_msg:
            feedback += "CRITICAL RULE VIOLATION: The JSON is invalid or truncated. This happens when you hit the token limit! You MUST be more concise, use 'scf.for' loops instead of unrolling manually, DO NOT generate redundant operations or duplicate constants, and ensure the JSON is fully closed.\n"

    feedback += "\nAnalyze ALL past errors, correct your JSON, and ensure strict compliance with MLIR rules."
    return feedback
