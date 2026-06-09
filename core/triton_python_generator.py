import re
from typing import Dict, List, Any
from core.schemas import MLIRFunctionBody, AnyOperation, ScfForLoop, ScfIf, ScfYield


class TritonPythonGenerator:
    """
    Deterministic translator from JSON SSA MLIR representation to Triton Python code.

    Supports:
      - Memory: tt.load, tt.store, tt.make_range, tt.splat, tt.addptr, tt.advance, tt.get_program_id
      - Arithmetic: arith.addf, subf, mulf, divf, maximumf, minimumf, cmpf, cmpi, select, constant
      - Math: math.exp, sqrt, log, abs, cos, sin
      - Control Flow: scf.for (with iter_args/yield), scf.if (basic), scf.yield
      - Reductions: tt.reduce (sum, max, min)
      - Tensor: tt.broadcast, tt.expand_dims, tt.dot, tt.reshape, tt.trans
    """

    BINARY_OPS = {
        "arith.addf": "+",
        "arith.mulf": "*",
        "arith.subf": "-",
        "arith.divf": "/",
        "arith.maximumf": "tl.maximum",
        "arith.minimumf": "tl.minimum",
    }

    UNARY_OPS = {
        "math.exp": "tl.exp",
        "math.sqrt": "tl.sqrt",
        "math.log": "tl.log",
        "math.absf": "tl.abs",
        "math.cos": "tl.cos",
        "math.sin": "tl.sin",
        "math.rsqrt": "tl.rsqrt",
        "arith.extf": "float",
        "arith.truncf": "int",
        "arith.sitofp": "float",
        "arith.fptosi": "int",
    }

    CMPF_PREDICATES = {
        0: "==",   # OEQ
        1: ">",   # OGT
        2: ">=",  # OGE
        3: "<",   # OLT
        4: "<=",  # OLE
        5: "!=",  # ONE
    }

    CMPI_PREDICATES = {
        0: "==",
        1: ">",
        2: ">=",
        3: "<",
        4: "<=",
        5: "!=",
    }

    REDUCE_COMBINERS = {
        "arith.addf": "tl.sum",
        "arith.maximumf": "tl.max",
        "arith.minimumf": "tl.min",
    }

    def __init__(self):
        pass

    @staticmethod
    def _sanitize_name(name: str) -> str:
        name = name.lstrip("%")
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if name and name[0].isdigit():
            name = "_" + name
        return name

    @staticmethod
    def _parse_ptr_type(type_str: str) -> tuple:
        match = re.match(r"!tt\.ptr<(.+)>", type_str)
        if match:
            return True, match.group(1)
        return False, type_str

    @staticmethod
    def _parse_tensor_shape(type_str: str) -> tuple:
        match = re.match(r"tensor<(.+)x(.+)>", type_str)
        if match:
            dims_str = match.group(1)
            dtype = match.group(2)
            dims = [int(d) if d.isdigit() else -1 for d in dims_str.split("x")]
            return dims, dtype
        return [], type_str

    @staticmethod
    def _infer_block_size_from_type(type_str: str) -> int:
        dims, _ = TritonPythonGenerator._parse_tensor_shape(type_str)
        if dims:
            return dims[0]
        return 1024

    def generate(self, function_body: MLIRFunctionBody) -> str:
        func_name = function_body.function_name
        args = function_body.arguments
        ops = function_body.operations
        returns = function_body.returns

        arg_names = [self._sanitize_name(arg.name) for arg in args]
        sig = f"def {func_name}({', '.join(arg_names)}):"

        block_size = self._detect_block_size(ops)

        lines = []
        symbols = self._build_symbol_table(function_body)

        # Block indexing
        lines.append("    pid = tl.program_id(0)")
        lines.append(f"    block_start = pid * {block_size}")

        for op in ops:
            emitted = self._emit_op(op, symbols, block_size)
            if isinstance(emitted, list):
                lines.extend(emitted)
            elif emitted:
                lines.append(emitted)

        if returns:
            lines.append(f"    # WARNING: kernel has returns {returns} — Triton kernels should not return values.")

        code_lines = [
            "import triton",
            "import triton.language as tl",
            "import torch",
            "",
            "@triton.jit",
            sig,
        ]
        code_lines.extend(lines)
        code_lines.append("")

        return "\n".join(code_lines)

    def _build_symbol_table(self, function_body: MLIRFunctionBody) -> Dict[str, str]:
        symbols = {}
        for arg in function_body.arguments:
            py_name = self._sanitize_name(arg.name)
            is_ptr, _ = self._parse_ptr_type(arg.type)
            symbols[arg.name] = py_name
            symbols[f"__ptr__{arg.name}"] = is_ptr
        return symbols

    def _detect_block_size(self, ops: list) -> int:
        for op in ops:
            if getattr(op, "opcode", None) == "tt.make_range":
                attrs = getattr(op, "attributes", {}) or {}
                start = attrs.get("start", 0)
                end = attrs.get("end", 1024)
                return end - start
        return 1024

    def _emit_op(self, op: Any, symbols: Dict[str, Any], block_size: int) -> Any:
        opcode = getattr(op, "opcode", None)
        result = getattr(op, "result", "none")
        operands = list(getattr(op, "operands", []))
        out_type = getattr(op, "out_type", None)
        attrs = getattr(op, "attributes", None) or {}

        # Handle ScfForLoop (returns list of lines)
        if isinstance(op, ScfForLoop):
            return self._emit_scf_for(op, symbols, block_size)

        # Handle ScfIf
        if isinstance(op, ScfIf):
            return self._emit_scf_if(op, symbols, block_size)

        # Handle ScfYield
        if isinstance(op, ScfYield):
            return self._emit_scf_yield(op, symbols)

        # arith.constant
        if opcode == "arith.constant":
            val = attrs.get("value", 0)
            py_name = self._sanitize_name(result)
            symbols[result] = py_name
            return f"    {py_name} = {val}"

        # tt.get_program_id
        if opcode == "tt.get_program_id":
            axis = attrs.get("axis", 0)
            py_name = self._sanitize_name(result)
            symbols[result] = py_name
            return f"    {py_name} = tl.program_id({axis})"

        # tt.get_num_programs
        if opcode == "tt.get_num_programs":
            axis = attrs.get("axis", 0)
            py_name = self._sanitize_name(result)
            symbols[result] = py_name
            return f"    {py_name} = tl.num_programs({axis})"

        # tt.make_range
        if opcode == "tt.make_range":
            start = attrs.get("start", 0)
            end = attrs.get("end", 1024)
            py_name = self._sanitize_name(result)
            symbols[result] = py_name
            return f"    {py_name} = block_start + tl.arange({start}, {end})"

        # tt.splat
        if opcode == "tt.splat":
            if operands:
                src = operands[0]
                src_name = symbols.get(src, self._sanitize_name(src))
                symbols[result] = src_name
                symbols[f"__expr__{result}"] = src_name
            return ""

        # tt.addptr
        if opcode == "tt.addptr":
            if len(operands) >= 2:
                base = operands[0]
                offset = operands[1]
                base_name = symbols.get(base, self._sanitize_name(base))
                offset_name = symbols.get(offset, self._sanitize_name(offset))
                expr = f"{base_name} + {offset_name}"
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                symbols[f"__expr__{result}"] = expr
                return ""
            return "    # ERROR: tt.addptr needs 2 operands"

        # tt.advance
        if opcode == "tt.advance":
            if len(operands) >= 2:
                base = operands[0]
                offset = operands[1]
                base_name = symbols.get(base, self._sanitize_name(base))
                offset_name = symbols.get(offset, self._sanitize_name(offset))
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                symbols[f"__expr__{result}"] = f"{base_name} + {offset_name}"
                return ""
            return "    # ERROR: tt.advance needs 2 operands"

        # tt.load
        if opcode == "tt.load":
            if operands:
                ptr_reg = operands[0]
                ptr_expr = symbols.get(f"__expr__{ptr_reg}", None)
                if ptr_expr is None:
                    ptr_expr = symbols.get(ptr_reg, self._sanitize_name(ptr_reg))
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                if len(operands) >= 2:
                    mask_expr = symbols.get(operands[1], self._sanitize_name(operands[1]))
                    return f"    {py_name} = tl.load({ptr_expr}, mask={mask_expr})"
                return f"    {py_name} = tl.load({ptr_expr})"
            return "    # ERROR: tt.load needs a pointer operand"

        # tt.store
        if opcode == "tt.store":
            if len(operands) >= 2:
                ptr_reg = operands[0]
                val_reg = operands[1]
                ptr_expr = symbols.get(f"__expr__{ptr_reg}", None)
                if ptr_expr is None:
                    ptr_expr = symbols.get(ptr_reg, self._sanitize_name(ptr_reg))
                val_expr = symbols.get(val_reg, self._sanitize_name(val_reg))
                if len(operands) >= 3:
                    mask_expr = symbols.get(operands[2], self._sanitize_name(operands[2]))
                    return f"    tl.store({ptr_expr}, {val_expr}, mask={mask_expr})"
                return f"    tl.store({ptr_expr}, {val_expr})"
            return "    # ERROR: tt.store needs ptr and value operands"

        # arith.cmpf
        if opcode == "arith.cmpf":
            if len(operands) >= 2:
                left = symbols.get(operands[0], self._sanitize_name(operands[0]))
                right = symbols.get(operands[1], self._sanitize_name(operands[1]))
                pred = attrs.get("predicate", 2)
                op_symbol = self.CMPF_PREDICATES.get(pred, "<")
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = {left} {op_symbol} {right}"
            return "    # ERROR: arith.cmpf needs 2 operands"

        # arith.cmpi
        if opcode == "arith.cmpi":
            if len(operands) >= 2:
                left = symbols.get(operands[0], self._sanitize_name(operands[0]))
                right = symbols.get(operands[1], self._sanitize_name(operands[1]))
                pred = attrs.get("predicate", 2)
                op_symbol = self.CMPI_PREDICATES.get(pred, "<")
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = {left} {op_symbol} {right}"
            return "    # ERROR: arith.cmpi needs 2 operands"

        # arith.select
        if opcode == "arith.select":
            if len(operands) >= 3:
                cond = symbols.get(operands[0], self._sanitize_name(operands[0]))
                true_val = symbols.get(operands[1], self._sanitize_name(operands[1]))
                false_val = symbols.get(operands[2], self._sanitize_name(operands[2]))
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = tl.where({cond}, {true_val}, {false_val})"
            return "    # ERROR: arith.select needs 3 operands"

        # tt.reduce
        if opcode == "tt.reduce":
            if operands:
                tensor_expr = symbols.get(operands[0], self._sanitize_name(operands[0]))
                combiner = getattr(op, "region_combiner", "arith.addf")
                func_name = self.REDUCE_COMBINERS.get(combiner, "tl.sum")
                axis = attrs.get("axis", 0)
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = {func_name}({tensor_expr}, axis={axis})"
            return "    # ERROR: tt.reduce needs a tensor operand"

        # Binary arithmetic
        if opcode in self.BINARY_OPS:
            if len(operands) >= 2:
                left = symbols.get(operands[0], self._sanitize_name(operands[0]))
                right = symbols.get(operands[1], self._sanitize_name(operands[1]))
                func_or_op = self.BINARY_OPS[opcode]
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                if func_or_op.startswith("tl."):
                    return f"    {py_name} = {func_or_op}({left}, {right})"
                return f"    {py_name} = {left} {func_or_op} {right}"
            return f"    # ERROR: {opcode} needs 2 operands"

        # Unary math
        if opcode in self.UNARY_OPS:
            if operands:
                arg = symbols.get(operands[0], self._sanitize_name(operands[0]))
                func = self.UNARY_OPS[opcode]
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = {func}({arg})"
            return f"    # ERROR: {opcode} needs 1 operand"

        # tt.broadcast
        if opcode == "tt.broadcast":
            if operands:
                src = symbols.get(operands[0], self._sanitize_name(operands[0]))
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = tl.broadcast_to({src}, ...)"
            return "    # ERROR: tt.broadcast needs operand"

        # tt.expand_dims
        if opcode == "tt.expand_dims":
            if operands:
                src = symbols.get(operands[0], self._sanitize_name(operands[0]))
                dim = attrs.get("dimension_to_expand", 0)
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = tl.expand_dims({src}, {dim})"
            return "    # ERROR: tt.expand_dims needs operand"

        # tt.dot
        if opcode == "tt.dot":
            if len(operands) >= 2:
                a = symbols.get(operands[0], self._sanitize_name(operands[0]))
                b = symbols.get(operands[1], self._sanitize_name(operands[1]))
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = tl.dot({a}, {b})"
            return "    # ERROR: tt.dot needs 2 operands"

        # tt.reshape
        if opcode == "tt.reshape":
            if operands:
                src = symbols.get(operands[0], self._sanitize_name(operands[0]))
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = tl.reshape({src}, ...)"
            return "    # ERROR: tt.reshape needs operand"

        # tt.trans
        if opcode == "tt.trans":
            if operands:
                src = symbols.get(operands[0], self._sanitize_name(operands[0]))
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = tl.trans({src})"
            return "    # ERROR: tt.trans needs operand"

        # tt.make_block_ptr
        if opcode == "tt.make_block_ptr":
            return "    # PHASE 3 NOT IMPLEMENTED: tt.make_block_ptr"

        # tt.atomic_rmw
        if opcode == "tt.atomic_rmw":
            return "    # PHASE 3 NOT IMPLEMENTED: tt.atomic_rmw"

        # tt.atomic_cas
        if opcode == "tt.atomic_cas":
            return "    # PHASE 3 NOT IMPLEMENTED: tt.atomic_cas"

        # tt.rand
        if opcode == "tt.rand":
            return "    # PHASE 3 NOT IMPLEMENTED: tt.rand"

        # tt.reduce.return (internal, shouldn't appear at top level)
        if opcode == "tt.reduce.return":
            return ""

        return f"    # UNSUPPORTED OP: {opcode}"

    def _emit_scf_for(self, op: ScfForLoop, symbols: Dict[str, Any], block_size: int) -> List[str]:
        """Convert scf.for with iter_args/yield into a Python for loop."""
        lines = []

        lb = op.lower_bound
        ub = op.upper_bound
        step = op.step
        loop_var_py = self._sanitize_name(op.loop_var)

        # Initialize iter_args
        iter_arg_names = []
        for arg_name, init_val in op.iter_args.items():
            py_arg = self._sanitize_name(arg_name)
            iter_arg_names.append(arg_name)
            if isinstance(init_val, (int, float)):
                lines.append(f"    {py_arg} = {init_val}")
            elif isinstance(init_val, str) and init_val.startswith("%"):
                src = symbols.get(init_val, self._sanitize_name(init_val))
                lines.append(f"    {py_arg} = {src}")
            else:
                lines.append(f"    {py_arg} = {init_val}")

        # Emit for loop header
        lines.append(f"    for {loop_var_py} in range({lb}, {ub}, {step}):")

        # Emit body (indented)
        for body_op in op.body:
            if isinstance(body_op, ScfYield):
                # Map yield operands back to iter_args
                for i, yield_val in enumerate(body_op.operands):
                    if i < len(iter_arg_names):
                        src = symbols.get(yield_val, self._sanitize_name(yield_val))
                        py_target = self._sanitize_name(iter_arg_names[i])
                        lines.append(f"        {py_target} = {src}")
            else:
                emitted = self._emit_op(body_op, symbols, block_size)
                if isinstance(emitted, list):
                    for e in emitted:
                        if e:
                            lines.append(f"    {e}")
                elif emitted:
                    lines.append(f"    {emitted}")

        # After loop: map iter_args to results
        for i, res_name in enumerate(op.results):
            if i < len(iter_arg_names):
                py_res = self._sanitize_name(res_name)
                py_src = self._sanitize_name(iter_arg_names[i])
                symbols[res_name] = py_res
                lines.append(f"    {py_res} = {py_src}")

        return lines

    def _emit_scf_if(self, op: ScfIf, symbols: Dict[str, Any], block_size: int) -> List[str]:
        """Convert scf.if into Python if/else or tl.where."""
        lines = []
        cond_expr = symbols.get(op.condition, self._sanitize_name(op.condition))

        if op.results:
            # scf.if with results - use tl.where for each result
            # Extract yield values from then and else bodies
            then_yields = []
            else_yields = []

            for body_op in op.then_body:
                if isinstance(body_op, ScfYield):
                    then_yields = [symbols.get(v, self._sanitize_name(v)) for v in body_op.operands]

            if op.else_body:
                for body_op in op.else_body:
                    if isinstance(body_op, ScfYield):
                        else_yields = [symbols.get(v, self._sanitize_name(v)) for v in body_op.operands]

            for i, res_name in enumerate(op.results):
                py_res = self._sanitize_name(res_name)
                symbols[res_name] = py_res
                then_val = then_yields[i] if i < len(then_yields) else "None"
                else_val = else_yields[i] if i < len(else_yields) else "None"
                lines.append(f"    {py_res} = tl.where({cond_expr}, {then_val}, {else_val})")
        else:
            # scf.if without results - standard Python if/else
            lines.append(f"    if {cond_expr}:")
            for body_op in op.then_body:
                if not isinstance(body_op, ScfYield):
                    emitted = self._emit_op(body_op, symbols, block_size)
                    if emitted:
                        lines.append(f"    {emitted}")
            if op.else_body:
                lines.append("    else:")
                for body_op in op.else_body:
                    if not isinstance(body_op, ScfYield):
                        emitted = self._emit_op(body_op, symbols, block_size)
                        if emitted:
                            lines.append(f"    {emitted}")

        return lines

    def _emit_scf_yield(self, op: ScfYield, symbols: Dict[str, Any]) -> str:
        """scf.yield is handled inside scf.for body emission."""
        return ""

    def generate_callable(self, function_body: MLIRFunctionBody) -> str:
        """Generate both the kernel and a host-side launcher function."""
        kernel_code = self.generate(function_body)

        func_name = function_body.function_name
        arg_names = [self._sanitize_name(arg.name) for arg in function_body.arguments]
        arg_types = [arg.type for arg in function_body.arguments]

        block_size = 1024
        for op in function_body.operations:
            ot = getattr(op, "out_type", None)
            if ot and "tensor<" in str(ot):
                dims, _ = self._parse_tensor_shape(str(ot))
                if dims:
                    block_size = dims[0]
                    break

        n_elements = block_size

        launcher_lines = [
            "",
            f"def launch_{func_name}(n_elements={n_elements}):",
            "    import torch",
            "",
            "    # Create random inputs on GPU",
        ]

        launch_args = []
        for py_name, arg_type in zip(arg_names, arg_types):
            is_ptr, dtype = self._parse_ptr_type(arg_type)
            if is_ptr:
                torch_dtype = "torch.float32"
                if dtype == "f16":
                    torch_dtype = "torch.float16"
                elif dtype == "i32":
                    torch_dtype = "torch.int32"
                launcher_lines.append(f"    {py_name} = torch.empty(n_elements, device='cuda', dtype={torch_dtype})")
                launch_args.append(f"{py_name}.data_ptr()")
            else:
                launcher_lines.append(f"    {py_name} = ...  # TODO: handle non-pointer arg")
                launch_args.append(py_name)

        launcher_lines.extend([
            "",
            f"    grid = (triton.cdiv(n_elements, {block_size}),)",
            f"    {func_name}[grid]({', '.join(launch_args)})",
            f"    return {launch_args[0] if launch_args else 'None'}",
        ])

        return kernel_code + "\n".join(launcher_lines)
