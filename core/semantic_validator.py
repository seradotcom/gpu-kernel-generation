from typing import List, Union, Any
from core.schemas import MlirResponse, ScfForLoop, ScfIf, ScfYield

class SemanticValidator:
    @staticmethod
    def validate(response: MlirResponse) -> List[str]:
        errors = []
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
    def _walk_operations(ops: List[Any], errors: List[str], current_scope: dict):
        for op in ops:
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
            if opcode in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.maximumf", "arith.minimumf", "arith.cmpf"):
                if len(resolved_types) == 2 and "unknown" not in resolved_types:
                    t1, t2 = resolved_types[0], resolved_types[1]
                    if t1 != t2:
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
                
                SemanticValidator._walk_operations(op.body, errors, new_scope)
                
                for res in op.results:
                    current_scope[res] = {"type": "unknown", "opcode": "scf.for_result"} # Register global results
                    
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
                pass
                
            else:
                # --- 4. POINTER AND TENSOR VALIDATION ---
                if opcode in ("tt.load", "tt.store"):
                    target_ptr = operands[0] if operands else None
                    if target_ptr in current_scope:
                        ptr_info = current_scope[target_ptr]
                        creator_op = ptr_info.get("opcode")
                        ptr_type = SemanticValidator._normalize_type(ptr_info.get("type", ""))
                        
                        if creator_op == "argument" and "ptr" not in ptr_type:
                            # Note: we coerce this later, but warning is still useful
                            pass 
                        elif creator_op in ("math.exp", "math.log", "arith.addf", "arith.mulf", "arith.maximumf", "arith.minimumf", "tt.load", "tt.reduce", "tt.dot"):
                            errors.append(f"[SEMANTIC ERROR] '{opcode}' requires memory pointers. '{target_ptr}' was generated by '{creator_op}', which outputs data/math values, not pointers.")
                            
                if opcode == "tt.reduce":
                    target_tensor = operands[0] if operands else None
                    combiner = getattr(op, "region_combiner", None)
                    if combiner and combiner not in ("arith.addf", "arith.maximumf", "arith.minimumf", "arith.mulf"):
                        errors.append(f"[SEMANTIC ERROR] 'tt.reduce' region_combiner '{combiner}' is invalid. Reductions MUST use a valid binary combiner like 'arith.addf' or 'arith.maximumf'. You cannot use unary ops like 'math.exp'.")

                    if target_tensor in current_scope:
                        ptr_info = current_scope[target_tensor]
                        tensor_type_str = SemanticValidator._normalize_type(ptr_info.get("type", "unknown"))
                        if "x" in tensor_type_str and "tensor" in tensor_type_str:
                            attrs = getattr(op, "attributes", {}) or {}
                            if "axis" not in attrs:
                                errors.append(f"[SEMANTIC ERROR] 'tt.reduce' on tensor '{tensor_type_str}' requires an 'axis' attribute in 'attributes' (e.g., {{\"axis\": 0}}).")
                                
                if opcode == "arith.cmpf":
                    attrs = getattr(op, "attributes", {}) or {}
                    if "predicate" not in attrs:
                        errors.append(f"[SEMANTIC ERROR] 'arith.cmpf' requires a 'predicate' attribute in 'attributes' (e.g., {{\"predicate\": 1}} for OGT).")

                # Infer output type for propagation
                out_type = getattr(op, "out_type", None)
                if not out_type:
                    if opcode in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.maximumf", "arith.minimumf", "math.exp", "math.log", "math.sqrt", "tt.load"):
                        out_type = resolved_types[0] if resolved_types else "unknown"
                    elif opcode == "arith.cmpf":
                        out_type = "i1"
                    elif opcode == "tt.splat":
                        out_type = "unknown" # requires explicit out_type
                    elif opcode == "tt.make_range":
                        start = 0
                        end = 0
                        attrs = getattr(op, "attributes", {}) or {}
                        if "start" in attrs: start = attrs["start"]
                        if "end" in attrs: end = attrs["end"]
                        out_type = f"tensor<{abs(end-start)}xi32>"
                    elif opcode == "tt.get_program_id":
                        out_type = "i32"
                    elif opcode == "tt.addptr":
                        out_type = resolved_types[0] if resolved_types else "unknown"
                    else:
                        out_type = "unknown"
                else:
                    out_type = SemanticValidator._normalize_type(out_type)

                # Register the result in current scope
                result = getattr(op, "result", None)
                if result and result != "none":
                    if result in current_scope:
                        errors.append(f"[SEMANTIC ERROR] Register '{result}' is assigned multiple times (shadowing).")
                    current_scope[result] = {"type": out_type, "opcode": opcode}
