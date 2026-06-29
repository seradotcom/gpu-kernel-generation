from typing import List, Union, Any
from core.schemas import MlirResponse, ScfForLoop, ScfIf, ScfYield

class SemanticValidator:
    @staticmethod
    def validate(response: MlirResponse) -> List[str]:
        errors = []
        # Validate function arguments
        for arg in response.code.arguments:
            arg_type_str = SemanticValidator._normalize_type(arg.type)
            if "tensor" in arg_type_str:
                errors.append(f"[SEMANTIC ERROR] Function arguments CANNOT be tensors! Triton kernel signatures only take scalars (e.g. 'i32') and memory pointers (e.g. '!tt.ptr<f32>'). You passed '{arg_type_str}' for '{arg.name}'.")
        
        # Initialize scope with function arguments
        initial_scope = {arg.name: {"type": arg.type, "opcode": "argument"} for arg in response.code.arguments}
        SemanticValidator._walk_operations(response.code.operations, errors, initial_scope)
        
        # Validate function returns
        for ret in response.code.returns:
            if ret not in initial_scope:
                errors.append(f"[SEMANTIC ERROR] Function returns variable '{ret}', but it was never defined in the global scope.")
                
        return errors

    @staticmethod
    def _normalize_type(t: Any) -> str:
        if hasattr(t, "value"):
            return t.value
        return str(t)

    @staticmethod
    def _walk_operations(ops: List[Any], errors: List[str], current_scope: dict, expected_yield_types: List[str] = None):
        i = 0
        while i < len(ops):
            op = ops[i]
            opcode = getattr(op, "opcode", None)
            if hasattr(opcode, "value"):
                opcode = opcode.value
            
            # --- 1. UNDECLARED OPERAND VALIDATION ---
            operands = getattr(op, "operands", [])
            resolved_types = []
            
            for operand in operands:
                if isinstance(operand, str) and operand.startswith("%"):
                    if operand not in current_scope:
                        errors.append(f"[SEMANTIC ERROR] Variable '{operand}' is used in '{opcode}' but was never defined in this scope.")
                        resolved_types.append("unknown")
                    else:
                        resolved_types.append(SemanticValidator._normalize_type(current_scope[operand].get("type", "unknown")))
                else:
                    # Numeric literal resolution
                    if isinstance(operand, float):
                        resolved_types.append("f32")
                    elif isinstance(operand, int):
                        resolved_types.append("i32")
                    elif operand != "none":
                        resolved_types.append("unknown")

            # --- 2. BINARY OPERATION TYPE VALIDATION ---
            if opcode in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.maximumf", "arith.minimumf", "arith.cmpf", "arith.addi", "arith.subi", "arith.muli", "arith.divsi", "arith.divui", "arith.cmpi", "arith.remui", "arith.remsi"):
                if len(resolved_types) == 2 and "unknown" not in resolved_types:
                    t1, t2 = resolved_types[0], resolved_types[1]
                    
                    if opcode.endswith("f") and "f" not in t1 and "f" not in t2:
                        errors.append(f"[SEMANTIC ERROR] '{opcode}' requires floating-point operands, but both are integers ('{t1}', '{t2}'). If you want integer math, use '{opcode[:-1]}i'.")

                    if t1 != t2:
                        # Auto-inject index_cast if index vs integer
                        if (t1 == "index" and "i" in t2) or ("i" in t1 and t2 == "index"):
                            cast_idx = 0 if t1 == "index" else 1
                            idx_op_name = op.operands[cast_idx]
                            new_reg = f"%cst_cast_{i}" if isinstance(idx_op_name, (int, float)) or (isinstance(idx_op_name, str) and not idx_op_name.startswith("%")) else f"{idx_op_name}_c_{i}"
                            from core.schemas import GenericOperation
                            cast_op = GenericOperation(
                                opcode="arith.index_cast",
                                operands=[idx_op_name],
                                result=new_reg,
                                out_type="i32"
                            )
                            ops.insert(i, cast_op)
                            op.operands[cast_idx] = new_reg
                            continue
                            
                        # Auto-inject sitofp if integer vs float in float ops
                        if opcode.endswith("f") and (("i" in t1 and "f" in t2) or ("f" in t1 and "i" in t2)):
                            cast_idx = 0 if "i" in t1 else 1
                            int_op_name = op.operands[cast_idx]
                            new_reg = f"%cst_s2f_{i}" if isinstance(int_op_name, (int, float)) or (isinstance(int_op_name, str) and not int_op_name.startswith("%")) else f"{int_op_name}_f_{i}"
                            
                            int_type = t1 if cast_idx == 0 else t2
                            import re
                            new_out_type = re.sub(r'i\d+', 'f32', int_type)
                            
                            from core.schemas import GenericOperation
                            cast_op = GenericOperation(
                                opcode="arith.sitofp",
                                operands=[int_op_name],
                                result=new_reg,
                                out_type=new_out_type
                            )
                            ops.insert(i, cast_op)
                            op.operands[cast_idx] = new_reg
                            continue

                        # Auto-inject tt.splat if scalar vs tensor
                        if "tensor" in t1 and "tensor" not in t2:
                            scalar_idx, tensor_type = 1, t1
                        elif "tensor" in t2 and "tensor" not in t1:
                            scalar_idx, tensor_type = 0, t2
                        else:
                            scalar_idx = -1
                            
                        if scalar_idx != -1:
                            scalar_op_name = op.operands[scalar_idx]
                            new_reg = f"%cst_splt_{i}" if isinstance(scalar_op_name, (int, float)) or (isinstance(scalar_op_name, str) and not scalar_op_name.startswith("%")) else f"{scalar_op_name}_s_{i}"
                            from core.schemas import GenericOperation, MlirOpcode
                            splat_op = GenericOperation(
                                opcode=MlirOpcode.TT_SPLAT,
                                operands=[scalar_op_name],
                                result=new_reg,
                                out_type=tensor_type
                            )
                            ops.insert(i, splat_op)
                            op.operands[scalar_idx] = new_reg
                            
                            if hasattr(op, "out_type") and op.out_type and "tensor" not in op.out_type:
                                op.out_type = tensor_type
                                
                            continue # Reprocess the newly inserted splat_op

                        errors.append(f"[SEMANTIC ERROR] '{opcode}' requires operands of the EXACT SAME type/shape. Got '{t1}' and '{t2}'. Use tt.splat to broadcast scalars to tensors if needed.")

            # --- 3. OP-SPECIFIC VALIDATION ---
            if isinstance(op, ScfForLoop):
                if len(op.results) != len(op.iter_args):
                    errors.append(f"[SEMANTIC ERROR] scf.for loop defines {len(op.iter_args)} iter_args but returns {len(op.results)} results. They must match exactly.")
                
                # Validate initial iter_args operands
                for arg_val in op.iter_args.values():
                    if isinstance(arg_val, str) and arg_val.startswith("%") and arg_val not in current_scope:
                        errors.append(f"[SEMANTIC ERROR] iter_arg initial value '{arg_val}' is undefined.")

                yields = [o for o in op.body if getattr(o, "opcode", None) == "scf.yield"]
                if not yields:
                    errors.append(f"[SEMANTIC ERROR] scf.for loop body is missing an scf.yield operation.")
                else:
                    last_op = op.body[-1]
                    if getattr(last_op, "opcode", None) != "scf.yield":
                        errors.append(f"[SEMANTIC ERROR] The LAST operation in an scf.for body MUST be scf.yield.")
                    for y in yields:
                        if len(y.operands) != len(op.iter_args):
                            errors.append(f"[SEMANTIC ERROR] scf.for body scf.yield returns {len(y.operands)} values, but loop iter_args expects {len(op.iter_args)} values.")
                
                if op.loop_var in current_scope:
                    errors.append(f"[SEMANTIC ERROR] Loop variable '{op.loop_var}' shadows an existing register.")
                
                new_scope = current_scope.copy()
                new_scope[op.loop_var] = {"type": "index", "opcode": "loop_var"}  # Register the loop iterator
                iter_types = []
                for arg_name, arg_val in op.iter_args.items():
                    init_type = "unknown"
                    if isinstance(arg_val, str) and arg_val.startswith("%"):
                        if arg_val in current_scope:
                            init_type = current_scope[arg_val].get("type", "unknown")
                    elif isinstance(arg_val, float):
                        init_type = "f32"
                    elif isinstance(arg_val, int):
                        init_type = "i32"
                    new_scope[arg_name] = {"type": init_type, "opcode": "iter_arg"}
                    iter_types.append(SemanticValidator._normalize_type(init_type))
                
                SemanticValidator._walk_operations(op.body, errors, new_scope, expected_yield_types=iter_types)
                
                for res, itype in zip(op.results, iter_types):
                    current_scope[res] = {"type": itype, "opcode": "scf.for_result"} # Register global results
                    
                    
            elif isinstance(op, ScfIf):
                # Validate condition
                if op.condition not in current_scope:
                    errors.append(f"[SEMANTIC ERROR] Condition '{op.condition}' for scf.if was never defined.")

                # Validate yields in Then block
                yields_then = [o for o in op.then_body if getattr(o, "opcode", None) == "scf.yield"]
                if not yields_then and len(op.results) > 0:
                     errors.append(f"[SEMANTIC ERROR] scf.if 'then' block must have an scf.yield because the operation expects {len(op.results)} results.")
                if yields_then:
                    for y in yields_then:
                        if len(y.operands) != len(op.results):
                            errors.append(f"[SEMANTIC ERROR] scf.if 'then' block yields {len(y.operands)} values, but expects {len(op.results)}.")
                
                # Validate yields in Else block
                if op.else_body:
                    yields_else = [o for o in op.else_body if getattr(o, "opcode", None) == "scf.yield"]
                    if not yields_else and len(op.results) > 0:
                         errors.append(f"[SEMANTIC ERROR] scf.if 'else' block must have an scf.yield because the operation expects {len(op.results)} results.")
                    if yields_else:
                        for y in yields_else:
                            if len(y.operands) != len(op.results):
                                errors.append(f"[SEMANTIC ERROR] scf.if 'else' block yields {len(y.operands)} values, but expects {len(op.results)}.")
                
                SemanticValidator._walk_operations(op.then_body, errors, current_scope.copy())
                if op.else_body:
                    SemanticValidator._walk_operations(op.else_body, errors, current_scope.copy())
                
                for res in op.results:
                    current_scope[res] = {"type": "unknown", "opcode": "scf.if_result"}
                    
            elif opcode == "scf.yield":
                if expected_yield_types is not None:
                    for idx, (y_t, e_t) in enumerate(zip(resolved_types, expected_yield_types)):
                        if y_t != e_t and "unknown" not in (y_t, e_t):
                            errors.append(f"[SEMANTIC ERROR] 'scf.yield' operand #{idx} type ('{y_t}') does not match the expected type ('{e_t}'). If you want to accumulate a tensor in a loop, you MUST initialize the iter_arg with a tensor (e.g. using tt.splat before the loop) instead of a scalar.")
                
            else:
                # --- 4. POINTER AND TENSOR VALIDATION ---
                if opcode in ("tt.load", "tt.store"):
                    target_ptr = operands[0] if operands else None
                    if target_ptr in current_scope:
                        ptr_info = current_scope[target_ptr]
                        creator_op = ptr_info.get("opcode")
                        ptr_type = SemanticValidator._normalize_type(ptr_info.get("type", ""))
                        
                        if "ptr" not in ptr_type:
                            errors.append(f"[SEMANTIC ERROR] '{opcode}' operand #0 MUST be a memory pointer (e.g. '!tt.ptr<f32>' or 'tensor<256x!tt.ptr<f32>>'). Got '{ptr_type}'.")
                        elif creator_op == "argument" and "ptr" not in ptr_type:
                            pass 
                        elif creator_op in ("math.exp", "math.log", "arith.addf", "arith.mulf", "arith.maximumf", "arith.minimumf", "tt.load", "tt.reduce", "tt.dot", "arith.constant"):
                            errors.append(f"[SEMANTIC ERROR] '{opcode}' requires memory pointers. '{target_ptr}' was generated by '{creator_op}', which outputs data/math values, not pointers.")
                            
                if opcode == "tt.store":
                    if len(resolved_types) >= 2:
                        ptr_type = resolved_types[0]
                        val_type = resolved_types[1]
                        if "tensor" in val_type and "tensor" not in ptr_type:
                            errors.append(f"[SEMANTIC ERROR] 'tt.store' value is a tensor ('{val_type}') but the pointer is a scalar ('{ptr_type}'). You MUST broadcast the pointer to a tensor of the same shape using 'tt.splat' and 'tt.addptr' BEFORE storing.")
                        elif "tensor" in ptr_type and "tensor" not in val_type:
                            errors.append(f"[SEMANTIC ERROR] 'tt.store' pointer is a tensor ('{ptr_type}') but the value is a scalar ('{val_type}'). Use 'tt.splat' on the value.")
                            
                if opcode == "arith.sitofp":
                    if resolved_types and "f" in resolved_types[0]:
                        errors.append(f"[SEMANTIC ERROR] 'arith.sitofp' requires an integer operand (e.g. 'i32' or 'tensor<256xi32>'), but got '{resolved_types[0]}'.")
                            
                if opcode == "tt.addptr":
                    if len(resolved_types) >= 2:
                        ptr_type = resolved_types[0]
                        offset_type = resolved_types[1]
                        if "f32" in offset_type or "f16" in offset_type or "f64" in offset_type:
                            errors.append(f"[SEMANTIC ERROR] 'tt.addptr' operand #1 (offsets) MUST be an integer or tensor of integers, but got '{offset_type}'. Floating point offsets are invalid.")
                        if "tensor" not in ptr_type and "tensor" in offset_type:
                            errors.append(f"[SEMANTIC ERROR] 'tt.addptr' operands must have the same shape. You passed a scalar pointer ('{ptr_type}') and a tensor offset ('{offset_type}'). You MUST use 'tt.splat' to broadcast the pointer first.")
                            
                if opcode == "tt.reduce":
                    target_tensor = operands[0] if operands else None
                    combiner = getattr(op, "region_combiner", None)
                    if combiner and combiner not in ("arith.addf", "arith.maximumf", "arith.minimumf", "arith.mulf"):
                        errors.append(f"[SEMANTIC ERROR] 'tt.reduce' region_combiner '{combiner}' is invalid. Reductions MUST use a valid binary combiner like 'arith.addf' or 'arith.maximumf'. You cannot use unary ops like 'math.exp'.")

                    # Auto-inject axis=0 if missing to prevent LLM loops
                    if not getattr(op, "attributes", None):
                        op.attributes = {"axis": 0}
                    elif "axis" not in op.attributes:
                        op.attributes["axis"] = 0

                    if target_tensor in current_scope:
                        ptr_info = current_scope[target_tensor]
                        tensor_type_str = SemanticValidator._normalize_type(ptr_info.get("type", "unknown"))
                        # Removed the axis error since it's auto-injected
                                
                if opcode == "arith.cmpf":
                    attrs = getattr(op, "attributes", {}) or {}
                    if "predicate" not in attrs:
                        errors.append(f"[SEMANTIC ERROR] 'arith.cmpf' requires a 'predicate' attribute in 'attributes' (e.g., {{\"predicate\": 1}} for OGT).")

                if opcode == "tt.dot":
                    if len(resolved_types) >= 2:
                        type_a = resolved_types[0]
                        type_b = resolved_types[1]
                        
                        import re
                        match_a = re.search(r"tensor<(\d+)x(\d+)x(.+)>", type_a)
                        match_b = re.search(r"tensor<(\d+)x(\d+)x(.+)>", type_b)
                        
                        if not match_a or not match_b:
                            errors.append(f"[SEMANTIC ERROR] 'tt.dot' operands MUST be 2D tensors (e.g., 'tensor<128x64xf16>'). Got '{type_a}' and '{type_b}'.")
                        else:
                            m_a, k_a, t_a = match_a.groups()
                            k_b, n_b, t_b = match_b.groups()
                            
                            if k_a != k_b:
                                errors.append(f"[SEMANTIC ERROR] 'tt.dot' inner dimension mismatch. Operand A has K={k_a}, Operand B has K={k_b}. They must be equal.")
                            
                            if len(resolved_types) >= 3:
                                type_c = resolved_types[2]
                                match_c = re.search(r"tensor<(\d+)x(\d+)x(.+)>", type_c)
                                if match_c:
                                    m_c, n_c, t_c = match_c.groups()
                                    if m_c != m_a or n_c != n_b:
                                        errors.append(f"[SEMANTIC ERROR] 'tt.dot' accumulator (operand #3) shape mismatch. Expected {m_a}x{n_b}, got {m_c}x{n_c}.")

                # Infer output type for propagation
                out_type = getattr(op, "out_type", None)
                
                if opcode == "tt.splat":
                    if resolved_types and "tensor" in resolved_types[0]:
                        errors.append(f"[SEMANTIC ERROR] 'tt.splat' can only be applied to a scalar. You passed a tensor ('{resolved_types[0]}').")

                inferred_type = "unknown"
                if opcode == "tt.splat":
                    inferred_type = resolved_types[0] if resolved_types and "tensor" in resolved_types[0] else (f"tensor<256x{resolved_types[0]}>" if resolved_types else "unknown")
                elif opcode == "tt.addptr":
                    if len(resolved_types) >= 2:
                        ptr_type = resolved_types[0]
                        offset_type = resolved_types[1]
                        if "tensor" in offset_type and "tensor" not in ptr_type:
                            inferred_type = f"tensor<256x{ptr_type}>"
                        elif "tensor" in ptr_type:
                            inferred_type = ptr_type
                        else:
                            inferred_type = ptr_type
                    else:
                        inferred_type = resolved_types[0] if resolved_types else "unknown"
                elif opcode == "tt.load":
                    if resolved_types:
                        t = resolved_types[0]
                        if t.startswith("tensor<") and "!tt.ptr<" in t:
                            inferred_type = t.replace("!tt.ptr<", "").replace(">", "", 1)
                        elif "!tt.ptr<" in t:
                            inferred_type = t.replace("!tt.ptr<", "").replace(">", "", 1)
                        else:
                            inferred_type = "f32"
                elif opcode == "tt.reduce":
                    if resolved_types:
                        t = resolved_types[0]
                        import re
                        match = re.search(r"tensor<\d+x(.+)>", t)
                        inferred_type = match.group(1) if match else t
                elif opcode in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.maximumf", "arith.minimumf", "math.exp", "math.log", "math.sqrt", "math.floor", "math.ceil", "math.cos", "math.sin", "math.absf", "arith.addi", "arith.subi", "arith.muli", "arith.divsi", "arith.divui", "arith.remui", "arith.remsi"):
                    inferred_type = resolved_types[0] if resolved_types else "unknown"
                elif opcode == "arith.select":
                    inferred_type = resolved_types[1] if len(resolved_types) > 1 else "unknown"
                elif opcode in ("arith.cmpf", "arith.cmpi"):
                    if resolved_types and "tensor" in resolved_types[0]:
                        import re
                        match = re.search(r"tensor<(\d+)x.+>", resolved_types[0])
                        inferred_type = f"tensor<{match.group(1)}xi1>" if match else "i1"
                    else:
                        inferred_type = "i1"
                elif opcode == "tt.make_range":
                    start = 0
                    end = 0
                    attrs = getattr(op, "attributes", {}) or {}
                    if "start" in attrs: start = attrs["start"]
                    if "end" in attrs: end = attrs["end"]
                    inferred_type = f"tensor<{abs(end-start)}xi32>"
                elif opcode == "tt.get_program_id":
                    inferred_type = "i32"
                elif opcode == "arith.constant":
                    attrs = getattr(op, "attributes", {}) or {}
                    val = attrs.get("value", 0)
                    if isinstance(val, str):
                        try:
                            val = float(val) if '.' in val or 'e' in val.lower() else int(val)
                        except ValueError:
                            pass
                    if isinstance(val, float): inferred_type = "f32"
                    elif isinstance(val, int): inferred_type = "i32"
                    elif isinstance(val, bool): inferred_type = "i1"

                # Keep LLM's out_type for casts, since we can't infer the exact target
                elif opcode in ("arith.extf", "arith.truncf", "arith.sitofp", "arith.fptosi", "arith.extsi", "arith.extui", "arith.trunci", "tt.ptr_to_int", "tt.int_to_ptr"):
                    inferred_type = SemanticValidator._normalize_type(out_type) if out_type else "unknown"
                    
                # Now reconcile inferred_type with LLM's out_type
                if out_type:
                    out_type = SemanticValidator._normalize_type(out_type)
                    if inferred_type != "unknown" and out_type != inferred_type:
                        if opcode in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.maximumf", "arith.minimumf", "math.exp", "math.log", "arith.addi", "arith.subi", "arith.muli"):
                            errors.append(f"[SEMANTIC ERROR] '{opcode}' requires the EXACT SAME type for all operands and results. Operands are '{inferred_type}' but you returned '{out_type}'. Use tt.splat to broadcast operands first if you want a larger tensor.")
                        op.out_type = inferred_type
                        out_type = inferred_type
                else:
                    out_type = inferred_type
                    op.out_type = out_type

                # Register the result in current scope
                result = getattr(op, "result", None)
                if result and result != "none":
                    if out_type == "unknown":
                        errors.append(f"[SEMANTIC ERROR] Could not infer 'out_type' for '{opcode}'. You MUST provide an explicit 'out_type' parameter.")
                        
                    if result in current_scope:
                        errors.append(f"[SEMANTIC ERROR] Register '{result}' is assigned multiple times (shadowing).")
                    current_scope[result] = {"type": out_type, "opcode": opcode}
                    
            i += 1
