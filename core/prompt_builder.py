import json

class PromptBuilder:
    """
    Constructs the system prompt for the LLM dynamically based on the user's query.
    Implements a simple RAG (Retrieval-Augmented Generation) heuristic.
    """
    
    def __init__(self):
        self.base_rules = (
            "CRITICAL RULES:\n"
            "1. Immutability & Scoping: Each operation saves its result in a new temporary register (e.g., '%0'). "
            "Registers defined inside a block (e.g., scf.for) CANNOT be accessed outside.\n"
            "2. TOPOLOGICAL ORDERING: You MUST define a register BEFORE you use it. For example, create '%mask' using 'arith.cmpf' BEFORE using it in 'tt.load'.\n"
            "3. NO UNDECLARED VARIABLES: You CANNOT use any register as an operand unless EXPLICITLY defined beforehand. No forward references. ALWAYS double-check your variable names.\n"
            "4. TYPE MATCHING: 'arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf', 'arith.cmpf' require EXACTLY the same shape and type for all operands. You CANNOT directly add a tensor and a scalar ('f32'). You MUST first broadcast the scalar using 'tt.splat', then apply math. NO EXCEPTIONS.\n"
            "5. FLOAT OPERATIONS: 'arith.divf', 'arith.addf', 'math.exp' expect float types ('f32', 'tensor<...xf32>'). DO NOT pass integers ('i32', 'index') to float math ops. NOTE: The result of 'tt.reduce' is a SCALAR (e.g., 'f32'). You MUST use 'tt.splat' on it before using it in math operations with other tensors!\n"
            "6. POINTERS: 'tt.load' and 'tt.store' operate on pointers. 'tt.load' returns math values (tensors/f32), NOT pointers. 'tt.store' consumes math values and writes to pointers.\n"
            "7. TT.STORE OPERAND ORDER: For 'tt.store', the FIRST operand MUST be the pointer ('%ptr'), and the SECOND operand MUST be the value ('%val'). e.g., operands: ['%ptr', '%val']. DO NOT swap them!\n"
            "8. SPMD PARADIGM: Do NOT use 'scf.for' to iterate over the grid or PIDs! The Triton kernel runs per-block. Use 'tt.get_program_id' (with 'axis' attribute) to get the block index. ONLY use 'scf.for' for loops WITHIN a block (e.g., reduction over sequence length).\n"
            "9. SCF.FOR CORRECTNESS: If you use 'scf.for', it MUST have an 'scf.yield' as its LAST operation. The number of 'results' must EXACTLY match the number of 'iter_args', and 'scf.yield' must return exactly that many operands.\n"
            "10. REDUCE COMBINERS: 'tt.reduce' MUST include an 'axis' attribute in 'attributes' (e.g. {\"axis\": 0}). Its region_combiner MUST be a binary op (e.g. 'arith.addf', 'arith.maximumf').\n"
            "11. CMP ATTRIBUTES: 'arith.cmpf' (for floats) and 'arith.cmpi' (for ints/index) REQUIRE a 'predicate' attribute inside the 'attributes' dict (e.g., {\"attributes\": {\"predicate\": 1}} for OGT or {\"attributes\": {\"predicate\": 2}} for SLT).\n"
            "12. CONSTANTS: You CANNOT pass numeric literals directly as operands. You MUST create them first using 'arith.constant'. EVERY 'arith.constant' MUST define its numeric value in attributes. For floats, DO NOT use extremely long decimals; round to 4 decimal places or use scientific notation.\n"
            "13. OUT_TYPE IS MANDATORY FOR MEMORY: Operations like 'tt.make_range', 'tt.splat', 'tt.addptr', and 'tt.load' MUST have an explicit 'out_type'. If you omit it, the compiler will assume they return pointers and math operations like 'arith.addf' will fail!\n"
            "14. MAKE_RANGE STRICT RULES: 'tt.make_range' takes EXACTLY 0 operands (pass []). It REQUIRES 'start' and 'end' in 'attributes'. Its out_type MUST be a tensor of standard integers (e.g., 'tensor<256xi32>'). DO NOT use 'index' or 'tensor<...xindex>'!\n"
            "15. KERNEL RETURNS: The main 'code' block must ALWAYS return an empty list: \"returns\": []. Triton GPU kernels modify global memory via 'tt.store', they do not return variables.\n"
            "16. INTEGER TYPES: Always use 'i32' for all loop steps, bounds, offsets, arithmetic logic, and program ids. Do NOT mix 'index' and 'i32'. Use 'i32' explicitly.\n"
            "17. HARDWARE BLOCK SIZES: GPUs operate on strict powers of 2. Pick a power of 2 from the allowed schema (e.g., 128, 256). Pad using boolean masks if bounds are arbitrary.\n"
            "18. CONCISENESS AND NO UNROLLING: DO NOT unroll loops manually. Use 'scf.for'. DO NOT hallucinate or generate repetitive/redundant constants (e.g., %c_result_splat_out1 ... %c_result_splat_out88). Be extremely concise to avoid exceeding token limits.\n"
            "19. JSON VALIDITY: Your JSON MUST be complete and properly closed before the end of the response. DO NOT generate more operations than necessary.\n"
        )
        
        self.few_shot_examples = (
            "GENERIC MLIR PARADIGMS (Use these as reference for your JSON structure):\n"
            "- SPMD KERNEL PARADIGM (No loops for grid): Use `tt.get_program_id` with `\"attributes\": {\"axis\": 0}` (and NO operands) to get the block ID, then multiply by BLOCK_SIZE to get the base offset for this block.\n"
            "- MEMORY BLOCK PARADIGM: To read global memory, strictly follow this sequence:\n"
            "  1. tt.make_range -> %offsets\n"
            "  2. tt.splat(%base_ptr) -> %splat_ptr\n"
            "  3. tt.addptr(%splat_ptr, %offsets) -> %block_ptrs\n"
            "  4. arith.constant(N) -> %c_limit\n"
            "  5. tt.splat(%c_limit) -> %splat_limit\n"
            "  6. arith.cmpi (with attributes: {\"predicate\": 2}) on [%offsets, %splat_limit] -> %mask\n"
            "  7. tt.load(%block_ptrs, %mask) -> %values\n\n"
            "- SCF.FOR LOOP PARADIGM (ONLY for loops WITHIN a block, like reducing along K dimension):\n"
            "  {\"opcode\":\"scf.for\", \"lower_bound\":0, \"upper_bound\":128, \"step\":1, \"loop_var\":\"%i\", \"iter_args\":{\"%sum\":0.0}, \"results\":[\"%out\"], \"body\": [\n"
            "    {\"opcode\":\"tt.load\", \"operands\":[\"%ptr\"], \"result\":\"%v\"},\n"
            "    {\"opcode\":\"arith.addf\", \"operands\":[\"%sum\",\"%v\"], \"result\":\"%n\"},\n"
            "    {\"opcode\":\"scf.yield\", \"operands\":[\"%n\"]}\n"
            "  ]}\n\n"
            "- TT.REDUCE PARADIGM:\n"
            "  {\"opcode\": \"tt.reduce\", \"operands\": [\"%tensor\"], \"result\": \"%row_sum\", \"region_combiner\": \"arith.addf\", \"attributes\": {\"axis\": 0}}\n"
            "  NOTE: `region_combiner` MUST be a valid binary op like `arith.addf`, `arith.maximumf`. NEVER use unary ops like `math.exp`.\n\n"
            "- 2D TILE / MATRIX PARADIGM (For tt.dot):\n"
            "  To load a 2D block, you must create 1D ranges, expand them, broadcast, and add to the base pointer:\n"
            "  1. tt.make_range -> %rm (0 to 128)\n"
            "  2. tt.make_range -> %rn (0 to 128)\n"
            "  3. tt.expand_dims(%rm) {\"dimension_to_expand\": 1} -> %rm_exp (shape 128x1)\n"
            "  4. tt.expand_dims(%rn) {\"dimension_to_expand\": 0} -> %rn_exp (shape 1x128)\n"
            "  5. tt.broadcast(%rm_exp) -> %rm_bcast (shape 128x128)\n"
            "  6. tt.broadcast(%rn_exp) -> %rn_bcast (shape 128x128)\n"
            "  7. Multiply offsets by strides and add to base pointer to get the 2D pointer tensor.\n\n"
            "- PERFECT VECTOR ADD JSON (Use this exact structure for simple kernels):\n"
            "{\n"
            "  \"code\": {\n"
            "    \"function_name\": \"vec_sum_kernel\",\n"
            "    \"arguments\": [\n"
            "      {\"name\": \"%arg0_A\", \"type\": \"!tt.ptr<f32>\"},\n"
            "      {\"name\": \"%arg1_B\", \"type\": \"!tt.ptr<f32>\"},\n"
            "      {\"name\": \"%arg2_C\", \"type\": \"!tt.ptr<f32>\"}\n"
            "    ],\n"
            "    \"operations\": [\n"
            "      {\"opcode\": \"tt.make_range\", \"operands\": [], \"attributes\": {\"start\": 0, \"end\": 256}, \"result\": \"%offsets\", \"out_type\": \"tensor<256xi32>\"},\n"
            "      {\"opcode\": \"tt.splat\", \"operands\": [\"%arg0_A\"], \"result\": \"%ptrs_A\", \"out_type\": \"tensor<256x!tt.ptr<f32>>\"},\n"
            "      {\"opcode\": \"tt.addptr\", \"operands\": [\"%ptrs_A\", \"%offsets\"], \"result\": \"%ptrs_A_off\", \"out_type\": \"tensor<256x!tt.ptr<f32>>\"},\n"
            "      {\"opcode\": \"tt.load\", \"operands\": [\"%ptrs_A_off\"], \"result\": \"%val_A\", \"out_type\": \"tensor<256xf32>\"},\n"
            "      {\"opcode\": \"tt.splat\", \"operands\": [\"%arg1_B\"], \"result\": \"%ptrs_B\", \"out_type\": \"tensor<256x!tt.ptr<f32>>\"},\n"
            "      {\"opcode\": \"tt.addptr\", \"operands\": [\"%ptrs_B\", \"%offsets\"], \"result\": \"%ptrs_B_off\", \"out_type\": \"tensor<256x!tt.ptr<f32>>\"},\n"
            "      {\"opcode\": \"tt.load\", \"operands\": [\"%ptrs_B_off\"], \"result\": \"%val_B\", \"out_type\": \"tensor<256xf32>\"},\n"
            "      {\"opcode\": \"arith.addf\", \"operands\": [\"%val_A\", \"%val_B\"], \"result\": \"%val_C\", \"out_type\": \"tensor<256xf32>\"},\n"
            "      {\"opcode\": \"tt.splat\", \"operands\": [\"%arg2_C\"], \"result\": \"%ptrs_C\", \"out_type\": \"tensor<256x!tt.ptr<f32>>\"},\n"
            "      {\"opcode\": \"tt.addptr\", \"operands\": [\"%ptrs_C\", \"%offsets\"], \"result\": \"%ptrs_C_off\", \"out_type\": \"tensor<256x!tt.ptr<f32>>\"},\n"
            "      {\"opcode\": \"tt.store\", \"operands\": [\"%ptrs_C_off\", \"%val_C\"], \"result\": \"none\"}\n"
            "    ],\n"
            "    \"returns\": []\n"
            "  }\n"
            "}\n"
        )
        
        self.scf_rules = ""
        self.reduce_rules = ""
        self.block_rules = ""
        
        import os
        self.mlir_ops_csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlir_operations.csv")
        
        self.triton_ops_registry = {
            "arith.cmpi": "Compares integers. Requires 'predicate' attribute inside 'attributes' dict. Operands: [lhs, rhs].",
            "arith.cmpf": "Compares floats. Requires 'predicate' attribute inside 'attributes' dict. Operands: [lhs, rhs].",
            "tt.get_program_id": "Returns the id of the current program. Requires 'axis' attribute (int). Takes EXACTLY 0 operands (pass []). Example attributes: {\"axis\": 0}",
            "tt.make_range": "Returns a tensor of indices. Requires 'start' and 'end' attributes (int). Takes EXACTLY 0 operands (pass []). Example attributes: {\"start\": 0, \"end\": 1024}",
            "tt.splat": "Broadcasts a scalar to a tensor of a given shape. Requires 'shape' attribute (list of ints). Operands: [value].",
            "tt.expand_dims": "Inserts a size-1 dimension into a tensor's shape. Requires 'dimension_to_expand' attribute (int). Operands: [tensor].",
            "tt.broadcast": "Broadcasts a tensor to a larger shape. Operands: [tensor]. Requires correct out_type.",
            "tt.addptr": "Adds an offset to a tensor of pointers. Operands: [base_ptr, offset].",
            "tt.load": "Loads data from a pointer or tensor of pointers. Operands: [ptr_tensor].",
            "tt.store": "Stores data to a pointer or tensor of pointers. Operands: [ptr_tensor, value_tensor].",
            "tt.dot": "Performs matrix multiplication. Operands: [tensor_a, tensor_b]. Both operands MUST be 2D tensors of the same inner dimension (e.g. MxK and KxN).",
            "tt.reduce": "Reduces a tensor along a specific axis. Requires 'axis' attribute and 'region_combiner' attribute (like 'arith.addf'). Operands: [tensor]."
        }

    def _fetch_rag_context(self, user_prompt: str) -> str:
        prompt_words = set(user_prompt.lower().replace(",", " ").replace(".", " ").split())
        context_lines = []
        
        # 1. Search Triton dictionary
        for op, desc in self.triton_ops_registry.items():
            op_keywords = set(op.replace("tt.", "").split("_"))
            if op_keywords.intersection(prompt_words):
                context_lines.append(f"- {op}: {desc}")
                
        # 2. Search MLIR CSV
        import csv
        import os
        if os.path.exists(self.mlir_ops_csv_path):
            with open(self.mlir_ops_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    op_json = row.get("Operacion_JSON_op", "")
                    desc = row.get("Descripcion", "")
                    if not op_json or not desc or desc.strip() == "Sin descripción":
                        continue
                    
                    # Extract keywords from opcode
                    op_keywords = set(op_json.split(".")[-1].split("_"))
                    if op_keywords.intersection(prompt_words):
                        context_lines.append(f"- {op_json}: {desc}")
                        
        if context_lines:
            return "DYNAMIC RAG CONTEXT FOR OPERATIONS:\n" + "\n".join(context_lines) + "\n"
        return ""

    def build_prompt(self, user_prompt: str, json_schema: dict) -> str:
        """
        Builds the system prompt by dynamically injecting rules based on keywords.
        """
        import copy
        schema_copy = copy.deepcopy(json_schema)
        
        # Minify massive Enums to save context tokens (Guided Decoding still uses the full backend schema)
        if "$defs" in schema_copy:
            if "MLIRType" in schema_copy["$defs"]:
                schema_copy["$defs"]["MLIRType"]["enum"] = [
                    "<REDACTED: Any valid scalar (e.g., 'f32'), pointer ('!tt.ptr<i32>'), or tensor shape (e.g., 'tensor<128x256xf32>')>"
                ]
            if "MlirOpcode" in schema_copy["$defs"]:
                # Keep only a few examples in the prompt to save tokens, the backend FSM will enforce the rest
                schema_copy["$defs"]["MlirOpcode"]["enum"] = [
                    "arith.addf", "arith.cmpi", "tt.load", "tt.store", "tt.splat", "tt.make_range", "tt.addptr", "..."
                ]

        schema_str = json.dumps(schema_copy, indent=2)
        
        prompt = (
            "Role: You are a Block-Level GPU Architect and expert LLM compiler in Triton MLIR. "
            "Your job is to translate high-level mathematical descriptions to a strict intermediate language in JSON format (Static Single Assignment - SSA).\n\n"
            f"You must output ONLY a valid JSON object that EXACTLY complies with the following JSON Schema:\n{schema_str}\n\n"
        )
        
        prompt += self.base_rules + "\n"
        
        user_prompt_lower = user_prompt.lower()
        
        if "online" in user_prompt_lower or "dynamic" in user_prompt_lower:
            prompt += (
                "ONLINE ALGORITHM RULES:\n"
                "- When an 'online' algorithm is requested (like online softmax), use an 'scf.for' loop to iterate and maintain running state via 'iter_args'.\n"
            )

        prompt += self.few_shot_examples + "\n"
            
        if any(word in user_prompt_lower for word in ["mask", "dynamic", "any size", "size"]):
            prompt += (
                "MEMORY MASKING RULE:\n"
                "- If the tensor size is dynamic, compute a boolean mask using arith.cmpi (comparing %offsets against tensor size).\n"
                "- Pass this mask to tt.load: tt.load(%ptrs, %mask).\n"
            )
            
        prompt += self._fetch_rag_context(user_prompt) + "\n"
            
        prompt += (
            "Output Format: DO NOT include markdown code blocks (e.g., ```json) or text outside the JSON. "
            "Your output must be parseable by json.loads().\n"
        )
        
        return prompt

    def build_triton_python_prompt(self) -> str:
        """
        Builds a system prompt for direct Triton Python kernel generation.
        This is used in the second phase after MLIR verify feedback has been collected.
        """
        prompt = (
            "Role: You are an expert GPU kernel engineer specializing in Triton (OpenAI Triton / PyTorch Triton).\n"
            "Your task is to write complete, valid, high-performance Triton Python kernels based on the task description "
            "and MLIR structural feedback provided.\n\n"
            "CRITICAL RULES FOR TRITON PYTHON:\n"
            "1. KERNEL SIGNATURE: Use `@triton.jit` decorator. Kernels take raw pointers and metadata (sizes, strides). "
            "They NEVER return Python values.\n"
            "2. SPMD MODEL: Each kernel instance processes ONE block. Get the block ID with `pid = tl.program_id(axis=0)`. "
            "Compute the block's start offset as `block_start = pid * BLOCK_SIZE`.\n"
            "3. BLOCK SIZE: Use `BLOCK_SIZE: tl.constexpr` as a parameter, or hardcode a power of 2 (e.g., 128, 256, 1024).\n"
            "4. POINTER ARITHMETIC: Create offsets with `offsets = block_start + tl.arange(0, BLOCK_SIZE)`. "
            "Load with `tl.load(ptr + offsets, mask=..., other=...)`.\n"
            "5. MASKING IS MANDATORY: For dynamic sizes, always compute a boolean mask: `mask = offsets < n_elements`. "
            "Pass it to `tl.load(ptr + offsets, mask=mask, other=0.0)` and `tl.store(ptr + offsets, value, mask=mask)`.\n"
            "6. NO RETURNS: Triton kernels modify memory in-place via `tl.store`. Do NOT use `return` statements.\n"
            "7. TYPE CONSISTENCY: All arithmetic should be on float32 (`torch.float32`) unless specified otherwise.\n"
            "8. REDUCTIONS: Use `tl.sum`, `tl.max`, `tl.min` for reductions. For `tl.dot`, ensure shapes are compatible.\n"
            "9. SHARED MEMORY: For reductions within a block, you can use `tl.zeros` + accumulate.\n"
            "10. CONSTANTS: Define scalar constants normally (e.g., `eps = 1e-5`). For tensor constants, use `tl.full`.\n"
            "11. 2D KERNELS: For matrix ops, map 1D program_id to 2D tile coordinates using row/col strides.\n"
            "12. MEMORY ORDER: When loading/storing 2D tiles, respect row-major memory layout and use proper strides.\n"
            "13. LAUNCHER: Include a host-side Python function that creates torch tensors and calls the kernel with proper grid.\n\n"
            "OUTPUT FORMAT:\n"
            "- Output ONLY valid Python code.\n"
            "- Do NOT wrap in markdown code blocks (no ```python).\n"
            "- Include all necessary imports: `import triton`, `import triton.language as tl`, `import torch`.\n"
            "- The code must be syntactically valid and compilable by `triton.compile`.\n"
            "- Include a launcher function (e.g., `run_kernel(...)`) that prepares inputs and calls the kernel.\n\n"
            "HOW TO USE THE MLIR FEEDBACK:\n"
            "- If the MLIR feedback says verification SUCCEEDED, it shows the exact types and operations needed. "
            "Mirror that structure in your Triton Python code (e.g., if MLIR uses `tensor<256xf32>`, use `BLOCK_SIZE=256` and `tl.arange(0, 256)`).\n"
            "- If the MLIR feedback says verification FAILED, the error tells you exactly what semantic mistake to avoid. "
            "For example, if MLIR says 'operand #1 must be ptr', ensure your `tl.load` receives a proper pointer expression, not a tensor value.\n"
            "- If MLIR says `scf.for` iter_args mismatch, ensure your Python `for` loop variables are initialized correctly and updated each iteration.\n"
            "- If MLIR says a register was never defined, ensure all variables are assigned before use in your Python code.\n\n"
            "EXAMPLE VECTOR ADD KERNEL:\n"
            "import triton\n"
            "import triton.language as tl\n"
            "import torch\n\n"
            "@triton.jit\n"
            "def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):\n"
            "    pid = tl.program_id(axis=0)\n"
            "    block_start = pid * BLOCK_SIZE\n"
            "    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n"
            "    mask = offsets < n_elements\n"
            "    a = tl.load(a_ptr + offsets, mask=mask)\n"
            "    b = tl.load(b_ptr + offsets, mask=mask)\n"
            "    c = a + b\n"
            "    tl.store(c_ptr + offsets, c, mask=mask)\n\n"
            "def run_vector_add(n=1024):\n"
            "    a = torch.rand(n, device='cuda')\n"
            "    b = torch.rand(n, device='cuda')\n"
            "    c = torch.empty(n, device='cuda')\n"
            "    BLOCK_SIZE = 256\n"
            "    grid = (triton.cdiv(n, BLOCK_SIZE),)\n"
            "    vector_add_kernel[grid](a.data_ptr(), b.data_ptr(), c.data_ptr(), n, BLOCK_SIZE=BLOCK_SIZE)\n"
            "    return c\n"
        )
        return prompt
