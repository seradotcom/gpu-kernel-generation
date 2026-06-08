import re
from typing import Dict, Any, List

try:
    from mlir.ir import Context, Module, Location, InsertionPoint
    import mlir.ir as ir
    from mlir.dialects import arith, scf
    HAS_MLIR = True
except ImportError:
    HAS_MLIR = False

from core.schemas import MLIRFunctionBody, AnyOperation, ScfForLoop, ScfIf, ScfYield

class ScopeStack:
    def __init__(self):
        self.scopes = [{}]
    def push(self):
        self.scopes.append({})
    def pop(self):
        self.scopes.pop()
    def clear(self):
        self.scopes = [{}]
    def __contains__(self, key):
        for scope in reversed(self.scopes):
            if key in scope: return True
        return False
    def __getitem__(self, key):
        for scope in reversed(self.scopes):
            if key in scope: return scope[key]
        raise KeyError(key)
    def __setitem__(self, key, value):
        self.scopes[-1][key] = value
    def get_available_vars(self):
        keys = []
        for scope in self.scopes:
            keys.extend(scope.keys())
        return list(set(keys))

class MLIRTranslator:
    """
    Deterministic translator from structured JSON to MLIR Dialects using mlir-py.
    """
    
    def __init__(self):
        """
        Initializes the global MLIR context and allows unregistered dialects.

        Raises:
            ImportError: If MLIR bindings are not found.
        """
        if not HAS_MLIR:
            raise ImportError("MLIR dependency not found.")
            
        self.context = Context()
        self.context.allow_unregistered_dialects = True # Important for 'tt', 'ttg' dialects
        self.value_env = ScopeStack()
        
    def _parse_type(self, type_str: Any) -> 'ir.Type':
        """
        Parses basic string types like 'f32', 'f16', 'index', 'tensor<64x128xf32>'.
        
        Args:
            type_str (Any): The MLIR type string or MLIRType enum.

        Returns:
            Type: The MLIR ir.Type object.
        """
        with self.context, Location.unknown():
            if hasattr(type_str, 'value'):
                type_str = type_str.value
            type_str = str(type_str).strip()
            if type_str == "f32": return ir.F32Type.get()
            if type_str == "f16": return ir.F16Type.get()
            if type_str == "i32": return ir.IntegerType.get_signless(32)
            if type_str == "i1": return ir.IntegerType.get_signless(1)
            if type_str == "index": return ir.IndexType.get()
            
            # Tensors: tensor<N...xType>
            # More generic parsing
            if type_str.startswith("tensor<") and type_str.endswith(">"):
                inner = type_str[7:-1]
                parts = inner.split("x")
                if len(parts) >= 2:
                    valid_dims = all(p.isdigit() or p == '?' for p in parts[:-1])
                    if not valid_dims:
                        raise RuntimeError(
                            f"Tipo de tensor inválido: '{type_str}'. "
                            f"El LLM generó un tipo malformado. "
                            f"Usa formatos como 'tensor<128xf32>' o 'tensor<64x128xf32>'."
                        )
                    type_part = parts[-1]
                    shape_part = parts[:-1]
                    shape = [int(s) if s.isdigit() else ir.ShapedType.get_dynamic_size() for s in shape_part]
                    element_type = self._parse_type(type_part)
                    return ir.RankedTensorType.get(shape, element_type)
            
            # Pointer type handling: !tt.ptr<f32>
            match_ptr = re.match(r"!tt\.ptr<(.+)>", type_str)
            if match_ptr:
                try:
                    return ir.Type.parse(type_str)
                except Exception:
                    pass

            # Fallback to native MLIR parsing if it's a valid general string
            return ir.Type.parse(type_str)

    def _infer_type(self, op: Any, resolved_operands: list = None) -> 'ir.Type':
        """
        Infers the return type. If there's an explicit cast, it uses it; otherwise it inherits from the operand.
        
        Args:
            op (Any): The Operation object.
            resolved_operands (list): List of ir.Value operands resolved from the environment.

        Returns:
            Type: The inferred MLIR ir.Type.
        """
        if getattr(op, "out_type", None):
            return self._parse_type(op.out_type)
            
        from core.schemas import MlirOpcode
        opcode = getattr(op, "opcode", None)
        
        if opcode == MlirOpcode.ARITH_CMPF:
            return ir.IntegerType.get_signless(1)
            
        if opcode == MlirOpcode.ARITH_SELECT:
            if resolved_operands and len(resolved_operands) >= 3:
                return resolved_operands[1].type
            return ir.F32Type.get()

        if opcode == MlirOpcode.TT_REDUCE:
            if resolved_operands:
                input_type = resolved_operands[0].type
                try:
                    return ir.ShapedType(input_type).element_type
                except Exception:
                    pass
            return ir.F32Type.get()
            
        if getattr(opcode, "value", opcode) == "tt.load":
            if resolved_operands:
                input_type = resolved_operands[0].type
                try:
                    # input_type is tensor<1024x!tt.ptr<f32>>
                    shaped = ir.ShapedType(input_type)
                    elem_type = str(shaped.element_type)
                    import re
                    match = re.search(r"!tt\.ptr<([^>]+)>", elem_type)
                    if match:
                        t = self._parse_type(match.group(1))
                        return ir.RankedTensorType.get(shaped.shape, t)
                except Exception:
                    pass
                
                input_type_str = str(input_type)
                import re
                match = re.search(r"!tt\.ptr<([^>]+)>", input_type_str)
                if match:
                    return self._parse_type(match.group(1))
            return ir.F32Type.get()

        if getattr(opcode, "value", opcode) == "tt.splat":
            if resolved_operands:
                # Find dominant shape from existing environment variables
                shape = [256]
                for scope in reversed(self.value_env.scopes):
                    for val in scope.values():
                        try:
                            shape = ir.ShapedType(val.type).shape
                            break
                        except Exception:
                            continue
                    else:
                        continue
                    break
                return ir.RankedTensorType.get(shape, resolved_operands[0].type)

        if getattr(opcode, "value", opcode) == "tt.make_range":
            # Infer tensor<Nxi32> from attributes
            start = 0
            end = 0
            if getattr(op, "attributes", None):
                start = op.attributes.get("start", 0)
                end = op.attributes.get("end", 0)
            size = abs(end - start)
            return ir.RankedTensorType.get([size], ir.IntegerType.get_signless(32))

        if getattr(opcode, "value", opcode) == "tt.get_program_id":
            return ir.IntegerType.get_signless(32)

        if resolved_operands:
            return resolved_operands[0].type
        # Default to index if no operands or out_type (e.g., constants)
        return ir.IndexType.get()

    def _process_operations(self, operations: list):
        """
        Processes a list of operations and inserts them into the current block.
        
        Args:
            operations (list): List of Operation or SCF objects.
        """
        with self.context, Location.unknown():
            for op_obj in operations:
                from core.schemas import UnaryOperation, BinaryOperation, GenericOperation
                if isinstance(op_obj, (UnaryOperation, BinaryOperation, GenericOperation)):
                    operands = []
                    for name in op_obj.operands:
                        if name is None or name == "none":
                            continue
                        parsed_num = None
                        if isinstance(name, (int, float)):
                            parsed_num = name
                        elif isinstance(name, str) and name.lstrip('-').replace('.','',1).isdigit():
                            parsed_num = float(name) if '.' in name else int(name)

                        if parsed_num is not None:
                            # Auto-inject arith.constant for literals
                            is_float = isinstance(parsed_num, float)
                            attr_type = ir.F32Type.get() if is_float else ir.IntegerType.get_signless(32)
                            attr_val = ir.FloatAttr.get(attr_type, float(parsed_num)) if is_float else ir.IntegerAttr.get(attr_type, int(parsed_num))
                            const_op = ir.Operation.create("arith.constant", results=[attr_type], operands=[], attributes={"value": attr_val})
                            operands.append(const_op.result)
                        else:
                            if name not in self.value_env:
                                available = ", ".join(self.value_env.get_available_vars())
                                raise RuntimeError(f"Operand '{name}' not found in environment. Available registers in this scope are: [{available}]")
                            operands.append(self.value_env[name])
                            
                    from core.schemas import MlirOpcode
                    
                    if op_obj.opcode == MlirOpcode.TT_REDUCE:
                        if not getattr(op_obj, "region_combiner", None):
                            raise RuntimeError(
                                f"'{op_obj.result}': tt.reduce requiere 'region_combiner' "
                                f"(ej: 'arith.addf' para suma, 'arith.maximumf' para max). "
                                f"Sin este campo el op es inválido semánticamente."
                            )

                    # Graceful fallback for tt.rand hallucination
                    if op_obj.opcode == MlirOpcode.TT_RAND or getattr(op_obj, "opcode", None) == "tt.rand":
                        val_type = self._parse_type(op_obj.out_type) if op_obj.out_type else ir.F32Type.get()
                        attr = ir.FloatAttr.get(val_type, 0.5) if isinstance(val_type, ir.FloatType) else ir.FloatAttr.get(ir.F32Type.get(), 0.5)
                        const_op = ir.Operation.create("arith.constant", results=[val_type], operands=[], attributes={"value": attr})
                        if op_obj.result != "none":
                            self.value_env[op_obj.result] = const_op.result
                        continue

                    if op_obj.opcode == MlirOpcode.ARITH_CONSTANT:
                        if not getattr(op_obj, "attributes", None) or "value" not in op_obj.attributes:
                            raise RuntimeError(
                                f"arith.constant en '{op_obj.result}' requiere 'attributes.value'. "
                                f"El LLM debe especificar el valor numérico."
                            )
                        val_type = self._parse_type(op_obj.out_type) if op_obj.out_type else ir.F32Type.get()
                        raw_val = op_obj.attributes["value"]
                        if isinstance(val_type, ir.FloatType) or (hasattr(ir, 'F32Type') and isinstance(val_type, ir.F32Type)):
                            attr = ir.FloatAttr.get(val_type, float(raw_val))
                        else:
                            attr = ir.IntegerAttr.get(val_type, int(raw_val))
                        const_op = ir.Operation.create("arith.constant", results=[val_type], operands=[], attributes={"value": attr})
                        self.value_env[op_obj.result] = const_op.result
                        continue

                    has_result = op_obj.result != "none"
                    results = []
                    if has_result:
                        result_type = self._infer_type(op_obj, resolved_operands=operands)
                        results = [result_type]
                    
                    # Parse attributes
                    mlir_attributes = {}
                    if getattr(op_obj, "attributes", None):
                        for k, v in op_obj.attributes.items():
                            if isinstance(v, list):
                                if all(isinstance(x, int) for x in v):
                                    mlir_attributes[k] = ir.ArrayAttr.get([ir.IntegerAttr.get(ir.IntegerType.get_signless(32), x) for x in v])
                                elif all(isinstance(x, float) for x in v):
                                    mlir_attributes[k] = ir.ArrayAttr.get([ir.FloatAttr.get(ir.F32Type.get(), x) for x in v])
                            elif isinstance(v, int):
                                if k == "predicate":
                                    mlir_attributes[k] = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), v)
                                else:
                                    mlir_attributes[k] = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), v)
                            elif isinstance(v, float):
                                mlir_attributes[k] = ir.FloatAttr.get(ir.F32Type.get(), v)
                            elif isinstance(v, bool):
                                mlir_attributes[k] = ir.BoolAttr.get(v)
                            elif isinstance(v, str):
                                mlir_attributes[k] = ir.StringAttr.get(v)
                    
                    regions = 1 if getattr(op_obj, "region_combiner", None) else 0

                    op_name = getattr(op_obj.opcode, "value", op_obj.opcode)

                    # Alias LLM hallucinations to valid MLIR opcodes
                    if op_name == "arith.maxf": op_name = "arith.maximumf"
                    if op_name == "arith.minf": op_name = "arith.minimumf"

                    if op_name == "tt.load":
                        sizes = [1, 0, 0] # [ptr, mask(optional), other(optional)]
                        if len(operands) == 2: sizes = [1, 1, 0]
                        elif len(operands) == 3: sizes = [1, 1, 1]
                        mlir_attributes["operandSegmentSizes"] = ir.DenseI32ArrayAttr.get(sizes)
                    elif op_name == "tt.store":
                        sizes = [1, 1, 0] # [ptr, value, mask(optional)]
                        if len(operands) == 3: sizes = [1, 1, 1]
                        mlir_attributes["operandSegmentSizes"] = ir.DenseI32ArrayAttr.get(sizes)


                    # Use generic constructor to support any dialect without hard Python bindings
                    # Since opcode is an Enum, we use .value
                    op = ir.Operation.create(
                        name=op_name,
                        results=results,
                        operands=operands,
                        attributes=mlir_attributes,
                        regions=regions
                    )
                    
                    if has_result:
                        self.value_env[op_obj.result] = op.result

                    # Handle custom region combiner (e.g. for tt.reduce)
                    if getattr(op_obj, "region_combiner", None):
                        region = op.regions[0]
                        
                        # Extract scalar element type
                        element_type = ir.F32Type.get()
                        if results:
                            try:
                                element_type = ir.ShapedType(results[0]).element_type
                            except Exception:
                                element_type = results[0]
                                
                        block_args = [element_type, element_type]
                        block = ir.Block.create_at_start(region, block_args)
                        with InsertionPoint(block):
                            combiner_op = ir.Operation.create(
                                name=op_obj.region_combiner,
                                results=[element_type],
                                operands=[block.arguments[0], block.arguments[1]]
                            )
                            ir.Operation.create(
                                name="tt.reduce.return",
                                results=[],
                                operands=[combiner_op.result]
                            )

                    
                elif isinstance(op_obj, ScfYield):
                    # scf.yield
                    operands = []
                    for name in op_obj.operands:
                        if name is None or name == "none":
                            continue
                        parsed_num = None
                        if isinstance(name, (int, float)):
                            parsed_num = name
                        elif isinstance(name, str) and name.lstrip('-').replace('.','',1).isdigit():
                            parsed_num = float(name) if '.' in name else int(name)

                        if parsed_num is not None:
                            is_float = isinstance(parsed_num, float)
                            attr_type = ir.F32Type.get() if is_float else ir.IntegerType.get_signless(32)
                            attr_val = ir.FloatAttr.get(attr_type, float(parsed_num)) if is_float else ir.IntegerAttr.get(attr_type, int(parsed_num))
                            const_op = ir.Operation.create("arith.constant", results=[attr_type], operands=[], attributes={"value": attr_val})
                            operands.append(const_op.result)
                        else:
                            if name not in self.value_env:
                                available = ", ".join(self.value_env.get_available_vars())
                                raise RuntimeError(f"Yield operand '{name}' not found in environment. Available registers in this scope are: [{available}]")
                            operands.append(self.value_env[name])
                    scf.YieldOp(operands)
                    
                elif isinstance(op_obj, ScfForLoop):
                    # scf.for loop
                    lb = self._get_or_create_index(op_obj.lower_bound)
                    ub = self._get_or_create_index(op_obj.upper_bound)
                    step = self._get_or_create_index(op_obj.step)
                    
                    # iter_args with literal fallback
                    iter_args_values = []
                    for init_val in op_obj.iter_args.values():
                        parsed_num = None
                        if isinstance(init_val, (int, float)):
                            parsed_num = init_val
                        elif isinstance(init_val, str) and init_val.lstrip('-').replace('.','',1).isdigit():
                            parsed_num = float(init_val) if '.' in init_val else int(init_val)

                        if parsed_num is not None:
                            is_float = isinstance(parsed_num, float)
                            attr_type = ir.F32Type.get() if is_float else ir.IntegerType.get_signless(32)
                            attr_val = ir.FloatAttr.get(attr_type, float(parsed_num)) if is_float else ir.IntegerAttr.get(attr_type, int(parsed_num))
                            const_op = ir.Operation.create("arith.constant", results=[attr_type], operands=[], attributes={"value": attr_val})
                            iter_args_values.append(const_op.result)
                        elif init_val in self.value_env:
                            iter_args_values.append(self.value_env[init_val])
                        else:
                            try:
                                f_val = float(init_val)
                                const_op = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), f_val))
                                iter_args_values.append(const_op.result)
                            except ValueError:
                                raise RuntimeError(f"Unknown iter_arg value: {init_val}.")
                    
                    for_op = scf.ForOp(lb, ub, step, iter_args_values)
                    
                    with InsertionPoint(for_op.body):
                        self.value_env.push()
                        # Register loop variable
                        self.value_env[op_obj.loop_var] = for_op.induction_variable
                        # Register iter_args inside the loop
                        for i, arg_name in enumerate(op_obj.iter_args.keys()):
                            self.value_env[arg_name] = for_op.inner_iter_args[i]
                            
                        self._process_operations(op_obj.body)
                        self.value_env.pop()
                        
                    # Register results generated by the loop
                    for i, res_name in enumerate(op_obj.results):
                        self.value_env[res_name] = for_op.results[i]
                        
                elif isinstance(op_obj, ScfIf):
                    # scf.if
                    if op_obj.condition not in self.value_env:
                        available = ", ".join(self.value_env.get_available_vars())
                        raise RuntimeError(f"Condition '{op_obj.condition}' for scf.if not found in environment. Available registers in this scope are: [{available}]")
                    cond_val = self.value_env[op_obj.condition]
                    has_else = bool(op_obj.else_body)
                    
                    # We default to f32 if no explicit types are provided
                    result_types = [ir.F32Type.get() for _ in op_obj.results]
                    if_op = scf.IfOp(cond_val, results_=result_types, hasElse=has_else)
                    
                    with InsertionPoint(if_op.then_block):
                        self.value_env.push()
                        self._process_operations(op_obj.then_body)
                        self.value_env.pop()
                        
                    if has_else:
                        with InsertionPoint(if_op.else_block):
                            self.value_env.push()
                            self._process_operations(op_obj.else_body)
                            self.value_env.pop()
                    
                    for i, res_name in enumerate(op_obj.results):
                        self.value_env[res_name] = if_op.results[i]

    def _get_or_create_index(self, val) -> 'ir.Value':
        """
        Converts int to arith.constant index, auto-casts integers to index, or fetches the register.
        """
        if isinstance(val, int):
            op = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), val))
            return op.result
        elif isinstance(val, str) and val.isdigit():
            op = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), int(val)))
            return op.result
            
        if val not in self.value_env:
            available = ", ".join(self.value_env.get_available_vars())
            raise RuntimeError(f"Index operand '{val}' not found in environment. Available registers in this scope are: [{available}]")
        v = self.value_env[val]
        # Auto-cast if it is an integer type but not an index
        if str(v.type).startswith('i') and not str(v.type).startswith('index'):
            cast_op = arith.IndexCastOp(ir.IndexType.get(), v)
            return cast_op.result
        return v

    def translate_to_module(self, function_body: MLIRFunctionBody) -> str:
        """
        Translation layer from abstract JSON to MLIR Dialects.
        
        Args:
            function_body (MLIRFunctionBody): The parsed Pydantic contract.

        Returns:
            str: The generated MLIR string.

        Raises:
            RuntimeError: If the module fails semantic verification.
        """
        with self.context, Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                # Extract input types and apply pointer coercion heuristics
                input_types = []
                for arg in function_body.arguments:
                    arg_type = self._parse_type(arg.type)
                    type_str = str(arg_type)
                    if arg.name.endswith("_ptr") and "ptr" not in type_str:
                        # Coerce to pointer
                        if "f16" in type_str:
                            arg_type = ir.Type.parse("!tt.ptr<f16>")
                        else:
                            arg_type = ir.Type.parse("!tt.ptr<f32>")
                    input_types.append(arg_type)
                
                # Start with empty results, we will update the signature after processing operations
                func_type = ir.FunctionType.get(inputs=input_types, results=[])
                func_op = ir.Operation.create(
                    name="tt.func",
                    results=[],
                    operands=[],
                    attributes={
                        "sym_name": ir.StringAttr.get(function_body.function_name),
                        "function_type": ir.TypeAttr.get(func_type)
                    },
                    regions=1
                )
                
                entry_block = ir.Block.create_at_start(func_op.regions[0], input_types)
                with InsertionPoint(entry_block):
                    self.value_env.clear()
                    # Map names to entry block Values
                    for i, arg in enumerate(function_body.arguments):
                        self.value_env[arg.name] = entry_block.arguments[i]
                        
                    self._process_operations(function_body.operations)
                    
                    # Generate return and infer return types
                    ret_vals = []
                    return_types = []
                    for r in function_body.returns:
                        if r not in self.value_env:
                            available = ", ".join(self.value_env.get_available_vars())
                            raise RuntimeError(f"Return value '{r}' not found in environment. Available registers in this scope are: [{available}]")
                        v = self.value_env[r]
                        ret_vals.append(v)
                        return_types.append(v.type)
                    
                    ir.Operation.create("tt.return", results=[], operands=ret_vals)
                    
                # Update function signature with correct return types
                new_func_type = ir.FunctionType.get(inputs=input_types, results=return_types)
                func_op.attributes["function_type"] = ir.TypeAttr.get(new_func_type)
                    
            import io
            import sys
            
            old_stderr = sys.stderr
            sys.stderr = buffer = io.StringIO()
            try:
                valid = module.operation.verify()
            finally:
                sys.stderr = old_stderr
                
            mlir_errors = buffer.getvalue()
            if not valid:
                raise RuntimeError(f"MLIR verification failed:\n{mlir_errors}")
                
            return str(module)
