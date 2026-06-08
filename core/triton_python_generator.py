import re
from typing import Dict, List, Any
from core.schemas import MLIRFunctionBody, AnyOperation, ScfForLoop, ScfIf, ScfYield


class TritonPythonGenerator:
    """
    Deterministic translator from JSON SSA MLIR representation to Triton Python code.

    Phase 1 subset (vector add / element-wise):
      - tt.get_program_id, tt.make_range, tt.splat, tt.addptr
      - tt.load, tt.store
      - arith.addf, arith.mulf, arith.subf
      - arith.constant, math.exp
      - No scf.for / scf.if yet (Phase 2)
    """

    # Map JSON opcodes to Python operator symbols or function names
    BINARY_OPS = {
        "arith.addf": "+",
        "arith.mulf": "*",
        "arith.subf": "-",
        "arith.divf": "/",
    }

    UNARY_OPS = {
        "math.exp": "tl.exp",
        "math.sqrt": "tl.sqrt",
        "math.absf": "tl.abs",
        "arith.extf": "float",   # placeholder
        "arith.truncf": "int",   # placeholder
    }

    def __init__(self):
        pass

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Strip % and replace invalid chars for Python identifiers."""
        name = name.lstrip("%")
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if name and name[0].isdigit():
            name = "_" + name
        return name

    @staticmethod
    def _parse_ptr_type(type_str: str) -> tuple:
        """
        Parse pointer types like '!tt.ptr<f32>'.
        Returns (is_ptr, element_dtype).
        """
        match = re.match(r"!tt\.ptr<(.+)>", type_str)
        if match:
            return True, match.group(1)
        return False, type_str

    @staticmethod
    def _parse_tensor_shape(type_str: str) -> tuple:
        """
        Parse tensor types like 'tensor<256xf32>' or 'tensor<64x128xf32>'.
        Returns (shape_list, element_dtype).
        """
        match = re.match(r"tensor<(.+)x(.+)>", type_str)
        if match:
            dims_str = match.group(1)
            dtype = match.group(2)
            dims = [int(d) if d.isdigit() else -1 for d in dims_str.split("x")]
            return dims, dtype
        return [], type_str

    @staticmethod
    def _infer_block_size_from_type(type_str: str) -> int:
        """Extract the first dimension from a tensor type for tl.arange."""
        dims, _ = TritonPythonGenerator._parse_tensor_shape(type_str)
        if dims:
            return dims[0]
        return 1024  # default fallback

    def _build_symbol_table(self, function_body: MLIRFunctionBody) -> Dict[str, str]:
        """
        Create a mapping from SSA register names to Python variable names.
        Also track whether a register holds a pointer vs a value.
        """
        symbols = {}
        for arg in function_body.arguments:
            py_name = self._sanitize_name(arg.name)
            is_ptr, _ = self._parse_ptr_type(arg.type)
            symbols[arg.name] = py_name
            symbols[f"__ptr__{arg.name}"] = is_ptr
        return symbols

    def generate(self, function_body: MLIRFunctionBody) -> str:
        """
        Generate a complete Triton Python module string.
        """
        func_name = function_body.function_name
        args = function_body.arguments
        ops = function_body.operations
        returns = function_body.returns

        # --- Build kernel signature ---
        arg_names = [self._sanitize_name(arg.name) for arg in args]
        sig = f"def {func_name}({', '.join(arg_names)}):"

        # --- Detect block size from tt.make_range ---
        block_size = self._detect_block_size(ops)

        # --- Generate body ---
        lines = []
        symbols = self._build_symbol_table(function_body)

        # Add block indexing
        lines.append("    pid = tl.program_id(0)")
        lines.append(f"    block_start = pid * {block_size}")

        for op in ops:
            line = self._emit_op(op, symbols, block_size)
            if line:
                lines.append(line)

        # Triton kernels don't return values; they operate via tt.store
        if returns:
            lines.append(f"    # WARNING: kernel has returns {returns} — Triton kernels should not return values.")

        # --- Assemble final Python module ---
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

    def _detect_block_size(self, ops: list) -> int:
        """Find BLOCK_SIZE from the first tt.make_range operation."""
        for op in ops:
            if getattr(op, "opcode", None) == "tt.make_range":
                attrs = getattr(op, "attributes", {}) or {}
                start = attrs.get("start", 0)
                end = attrs.get("end", 1024)
                return end - start
        return 1024

    def _emit_op(self, op: Any, symbols: Dict[str, Any], block_size: int) -> str:
        """
        Emit a single line of Python for a given JSON operation.
        Returns the line string or '' if nothing to emit.
        """
        opcode = getattr(op, "opcode", None)
        result = getattr(op, "result", "none")
        operands = list(getattr(op, "operands", []))
        out_type = getattr(op, "out_type", None)
        attrs = getattr(op, "attributes", None) or {}

        # --- arith.constant ---
        if opcode == "arith.constant":
            val = attrs.get("value", 0)
            py_name = self._sanitize_name(result)
            symbols[result] = py_name
            return f"    {py_name} = {val}"

        # --- tt.get_program_id ---
        if opcode == "tt.get_program_id":
            axis = attrs.get("axis", 0)
            py_name = self._sanitize_name(result)
            symbols[result] = py_name
            return f"    {py_name} = tl.program_id({axis})"

        # --- tt.make_range ---
        if opcode == "tt.make_range":
            start = attrs.get("start", 0)
            end = attrs.get("end", 1024)
            py_name = self._sanitize_name(result)
            symbols[result] = py_name
            return f"    {py_name} = block_start + tl.arange({start}, {end})"

        # --- tt.splat ---
        if opcode == "tt.splat":
            if operands:
                src = operands[0]
                src_name = symbols.get(src, self._sanitize_name(src))
                # Mark this register as a pointer expression (base ptr)
                symbols[result] = src_name
                symbols[f"__expr__{result}"] = src_name
            return ""

        # --- tt.addptr ---
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
                # No line emitted; tt.load / tt.store will inline the expression directly
                return ""
            return "    # ERROR: tt.addptr needs 2 operands"

        # --- tt.load ---
        if opcode == "tt.load":
            if operands:
                ptr_reg = operands[0]
                # Inline pointer expression if available
                ptr_expr = symbols.get(f"__expr__{ptr_reg}", None)
                if ptr_expr is None:
                    ptr_expr = symbols.get(ptr_reg, self._sanitize_name(ptr_reg))
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = tl.load({ptr_expr})"
            return "    # ERROR: tt.load needs a pointer operand"

        # --- tt.store ---
        if opcode == "tt.store":
            if len(operands) >= 2:
                ptr_reg = operands[0]
                val_reg = operands[1]
                ptr_expr = symbols.get(f"__expr__{ptr_reg}", None)
                if ptr_expr is None:
                    ptr_expr = symbols.get(ptr_reg, self._sanitize_name(ptr_reg))
                val_expr = symbols.get(val_reg, self._sanitize_name(val_reg))
                return f"    tl.store({ptr_expr}, {val_expr})"
            return "    # ERROR: tt.store needs ptr and value operands"

        # --- Binary arithmetic ---
        if opcode in self.BINARY_OPS:
            if len(operands) >= 2:
                left = symbols.get(operands[0], self._sanitize_name(operands[0]))
                right = symbols.get(operands[1], self._sanitize_name(operands[1]))
                op_symbol = self.BINARY_OPS[opcode]
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = {left} {op_symbol} {right}"
            return f"    # ERROR: {opcode} needs 2 operands"

        # --- Unary math ---
        if opcode in self.UNARY_OPS:
            if operands:
                arg = symbols.get(operands[0], self._sanitize_name(operands[0]))
                func = self.UNARY_OPS[opcode]
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = {func}({arg})"
            return f"    # ERROR: {opcode} needs 1 operand"

        # --- tt.broadcast ---
        if opcode == "tt.broadcast":
            if operands:
                src = symbols.get(operands[0], self._sanitize_name(operands[0]))
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = tl.broadcast_to({src}, ...)  # shape inferred"
            return "    # ERROR: tt.broadcast needs operand"

        # --- tt.expand_dims ---
        if opcode == "tt.expand_dims":
            if operands:
                src = symbols.get(operands[0], self._sanitize_name(operands[0]))
                dim = attrs.get("dimension_to_expand", 0)
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = tl.expand_dims({src}, {dim})"
            return "    # ERROR: tt.expand_dims needs operand"

        # --- tt.dot ---
        if opcode == "tt.dot":
            if len(operands) >= 2:
                a = symbols.get(operands[0], self._sanitize_name(operands[0]))
                b = symbols.get(operands[1], self._sanitize_name(operands[1]))
                py_name = self._sanitize_name(result)
                symbols[result] = py_name
                return f"    {py_name} = tl.dot({a}, {b})"
            return "    # ERROR: tt.dot needs 2 operands"

        # --- Unsupported / Phase 2 ---
        if opcode in ("scf.for", "scf.if", "scf.yield"):
            return f"    # PHASE 2 NOT IMPLEMENTED: {opcode}"

        return f"    # UNSUPPORTED OP: {opcode}"

    def generate_callable(self, function_body: MLIRFunctionBody) -> str:
        """
        Generate both the kernel and a host-side launcher function.
        Returns a complete Python script string.
        """
        kernel_code = self.generate(function_body)

        func_name = function_body.function_name
        arg_names = [self._sanitize_name(arg.name) for arg in function_body.arguments]
        arg_types = [arg.type for arg in function_body.arguments]

        # Determine block size from first tensor operation
        block_size = 1024
        for op in function_body.operations:
            ot = getattr(op, "out_type", None)
            if ot and "tensor<" in str(ot):
                dims, _ = self._parse_tensor_shape(str(ot))
                if dims:
                    block_size = dims[0]
                    break

        # Infer number of elements from arguments (heuristic: first pointer arg)
        n_elements = block_size  # default

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
            f"    return {launch_args[0] if launch_args else 'None'}  # return first output for validation",
        ])

        return kernel_code + "\n".join(launcher_lines)
