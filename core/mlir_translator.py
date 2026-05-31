import re
from typing import Dict, Any, List

try:
    from mlir.ir import Context, Module, Location, InsertionPoint, Type, Value
    from mlir.dialects import arith, tensor, scf, func
    import mlir.ir as ir
    HAS_MLIR = True
except ImportError:
    HAS_MLIR = False

from core.schemas import MLIRFunctionBody, Operation, ScfForLoop, ScfIf, ScfYield

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
        self.value_env: Dict[str, Value] = {}
        
    def _parse_type(self, type_str: str) -> Type:
        """
        Parses basic string types like 'f32', 'f16', 'index', 'tensor<64x128xf32>'.
        
        Args:
            type_str (str): The MLIR type string.

        Returns:
            Type: The MLIR ir.Type object.
        """
        with self.context, Location.unknown():
            type_str = type_str.strip()
            if type_str == "f32": return ir.F32Type.get()
            if type_str == "f16": return ir.F16Type.get()
            if type_str == "i32": return ir.IntegerType.get_signless(32)
            if type_str == "i1": return ir.IntegerType.get_signless(1)
            if type_str == "index": return ir.IndexType.get()
            
            # Tensors: tensor<N...xType>
            match = re.match(r"tensor<(.+)x([a-z0-9]+)>", type_str)
            if match:
                shape_str = match.group(1).split("x")
                shape = [int(s) if s.isdigit() else ir.ShapedType.get_dynamic_size() for s in shape_str] # Dynamic dimension support
                element_type = self._parse_type(match.group(2))
                return ir.RankedTensorType.get(shape, element_type)
            
            # Fallback to native MLIR parsing if it's a valid general string
            return ir.Type.parse(type_str)

    def _infer_type(self, op: Operation) -> Type:
        """
        Infers the return type. If there's an explicit cast, it uses it; otherwise it inherits from the operand.
        
        Args:
            op (Operation): The Operation object.

        Returns:
            Type: The inferred MLIR ir.Type.
        """
        if op.out_type:
            return self._parse_type(op.out_type)
        if op.operands:
            return self.value_env[op.operands[0]].type
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
                if isinstance(op_obj, Operation):
                    # Standard operation (Triton, Arith, etc)
                    operands = [self.value_env[name] for name in op_obj.operands]
                    result_type = self._infer_type(op_obj)
                    
                    # Use generic constructor to support any dialect without hard Python bindings
                    # Since opcode is an Enum, we use .value
                    op = ir.Operation.create(
                        name=op_obj.opcode.value,
                        results=[result_type],
                        operands=operands
                    )
                    self.value_env[op_obj.result] = op.result
                    
                elif isinstance(op_obj, ScfYield):
                    # scf.yield
                    operands = [self.value_env[name] for name in op_obj.operands]
                    scf.YieldOp(operands)
                    
                elif isinstance(op_obj, ScfForLoop):
                    # scf.for loop
                    lb = self._get_or_create_index(op_obj.lower_bound)
                    ub = self._get_or_create_index(op_obj.upper_bound)
                    step = self._get_or_create_index(op_obj.step)
                    
                    # iter_args with literal fallback
                    iter_args_values = []
                    for init_val in op_obj.iter_args.values():
                        if init_val in self.value_env:
                            iter_args_values.append(self.value_env[init_val])
                        else:
                            # If LLM passed a string literal like "0.0", auto-create an f32 constant
                            try:
                                f_val = float(init_val)
                                const_op = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), f_val))
                                iter_args_values.append(const_op.result)
                            except ValueError:
                                raise KeyError(f"Unknown iter_arg value: {init_val}")
                    
                    for_op = scf.ForOp(lb, ub, step, iter_args_values)
                    
                    with InsertionPoint(for_op.body):
                        # Register loop variable
                        self.value_env[op_obj.loop_var] = for_op.induction_variable
                        # Register iter_args inside the loop
                        for i, arg_name in enumerate(op_obj.iter_args.keys()):
                            self.value_env[arg_name] = for_op.inner_iter_args[i]
                            
                        self._process_operations(op_obj.body)
                        
                    # Register results generated by the loop
                    for i, res_name in enumerate(op_obj.results):
                        self.value_env[res_name] = for_op.results[i]
                        
                elif isinstance(op_obj, ScfIf):
                    # scf.if
                    cond_val = self.value_env[op_obj.condition]
                    has_else = bool(op_obj.else_body)
                    
                    # Infers return types by looking for yields, for simplicity.
                    # In practice, the LLM should pass it or it's inferred statically.
                    if_op = scf.IfOp(cond_val, results_=[], hasElse=has_else)
                    
                    with InsertionPoint(if_op.then_block):
                        self._process_operations(op_obj.then_body)
                        
                    if has_else:
                        with InsertionPoint(if_op.else_block):
                            self._process_operations(op_obj.else_body)

    def _get_or_create_index(self, val) -> Value:
        """
        Converts int to arith.constant index, auto-casts integers to index, or fetches the register.
        """
        if isinstance(val, int):
            op = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), val))
            return op.result
        elif isinstance(val, str) and val.isdigit():
            op = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), int(val)))
            return op.result
            
        v = self.value_env[val]
        # Auto-cast if it is an integer type but not an index
        if ir.IntegerType.isinstance(v.type) and v.type != ir.IndexType.get():
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
                # Extract input types
                input_types = [self._parse_type(arg.type) for arg in function_body.arguments]
                
                # To infer returns, we assume the input type or fallback.
                # In a real compiler, a double pass is done, but for Triton we usually return ().
                return_types = [input_types[0] for _ in function_body.returns] if input_types else []
                
                func_type = ir.FunctionType.get(inputs=input_types, results=return_types)
                func_op = func.FuncOp(name=function_body.function_name, type=func_type)
                
                entry_block = func_op.add_entry_block()
                with InsertionPoint(entry_block):
                    self.value_env.clear()
                    # Map names to entry block Values
                    for i, arg in enumerate(function_body.arguments):
                        self.value_env[arg.name] = entry_block.arguments[i]
                        
                    self._process_operations(function_body.operations)
                    
                    # Generate return
                    ret_vals = [self.value_env[r] for r in function_body.returns]
                    func.ReturnOp(ret_vals)
                    
            if not module.operation.verify():
                raise RuntimeError("Generated MLIR Module is not semantically valid.")
                
            return str(module)
