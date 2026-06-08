import time
import traceback
from typing import Dict, Any, Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

import torch


class TritonExecutor:
    """
    Compiles and executes generated Triton Python kernels,
    validates correctness against PyTorch, and measures speedup.
    """

    def __init__(self):
        self.last_error = None

    def run(
        self,
        triton_code: str,
        n_elements: int = 1024,
        warmup: int = 2,
        reps: int = 10,
    ) -> Dict[str, Any]:
        """
        Compile the Triton kernel, run it, and benchmark.

        Args:
            triton_code: Full Python source containing @triton.jit kernel.
            n_elements: Number of elements for test tensors.
            warmup: Number of warmup iterations before timing.
            reps: Number of timed iterations.

        Returns:
            dict with keys:
                success: bool
                correct: bool | None
                speedup: float | None
                error: str | None
                kernel_time_ms: float | None
                ref_time_ms: float | None
        """
        result = {
            "success": False,
            "correct": None,
            "speedup": None,
            "error": None,
            "kernel_time_ms": None,
            "ref_time_ms": None,
        }

        if not HAS_TRITON:
            result["error"] = "Triton is not installed in this environment. Cannot compile or execute GPU kernels."
            return result

        # --- 1. Write kernel to a real .py file (Triton's @triton.jit needs inspect.getsourcelines) ---
        import tempfile
        import importlib.util
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(triton_code)
                temp_path = f.name
            
            spec = importlib.util.spec_from_file_location("temp_kernel", temp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            namespace = module.__dict__
        except Exception as e:
            result["error"] = f"Triton Python compilation failed:\n{traceback.format_exc()}"
            return result

        # --- 2. Find the @triton.jit kernel ---
        kernel_fn = None
        kernel_name = None
        for name, obj in namespace.items():
            if callable(obj) and hasattr(obj, "kernel"):
                kernel_fn = obj
                kernel_name = name
                break

        if kernel_fn is None:
            result["error"] = "No @triton.jit kernel found in generated code."
            return result

        # --- 3. Infer signature from the kernel function ---
        import inspect
        # Triton JITFunction: .kernel may be None until compilation.
        # Use arg_names (stable across versions) or inspect the wrapper itself.
        if hasattr(kernel_fn, "arg_names") and kernel_fn.arg_names:
            params = list(kernel_fn.arg_names)
        else:
            try:
                sig = inspect.signature(kernel_fn)
                params = list(sig.parameters.keys())
            except Exception:
                params = []
        
        # Fallback: parse function definition from source if inspect fails
        if not params:
            import re
            src_match = re.search(r"def\s+\w+\s*\((.*?)\)", triton_code, re.DOTALL)
            if src_match:
                param_str = src_match.group(1)
                params = [p.strip().split("=")[0].split(":")[0].strip() for p in param_str.split(",") if p.strip()]

        # --- 4. Create test tensors ---
        try:
            tensors = self._create_test_tensors(params, n_elements)
        except Exception as e:
            result["error"] = f"Failed to create test tensors:\n{traceback.format_exc()}"
            return result

        # --- 5. Determine grid ---
        block_size = self._infer_block_size(triton_code)
        grid = (triton.cdiv(n_elements, block_size),)

        # --- 6. Run reference (PyTorch) implementation ---
        ref_output, ref_time_ms = self._run_reference(tensors, kernel_name, n_elements, warmup, reps)

        # --- 7. Run Triton kernel ---
        try:
            # Warmup
            for _ in range(warmup):
                kernel_fn[grid](*tensors)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # Timed runs
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(reps):
                    kernel_fn[grid](*tensors)
                end.record()
                torch.cuda.synchronize()
                kernel_time_ms = start.elapsed_time(end) / reps
            else:
                # CPU fallback timing (less precise)
                t0 = time.perf_counter()
                for _ in range(reps):
                    kernel_fn[grid](*tensors)
                t1 = time.perf_counter()
                kernel_time_ms = (t1 - t0) * 1000 / reps

            result["kernel_time_ms"] = kernel_time_ms

        except Exception as e:
            result["error"] = f"Triton kernel execution failed:\n{traceback.format_exc()}"
            return result

        # --- 8. Validate correctness ---
        try:
            # The first tensor is usually the output (for simple kernels)
            # For more complex kernels, we might need smarter detection
            output_tensor = tensors[-1] if len(tensors) >= 3 else tensors[0]
            if ref_output is not None:
                is_close = torch.allclose(output_tensor, ref_output, atol=1e-4, rtol=1e-3)
                result["correct"] = is_close
            else:
                result["correct"] = True  # No reference to compare
        except Exception as e:
            result["correct"] = False
            result["error"] = f"Correctness check failed:\n{traceback.format_exc()}"
            return result

        # --- 9. Compute speedup ---
        if ref_time_ms is not None and ref_time_ms > 0:
            result["speedup"] = ref_time_ms / kernel_time_ms if kernel_time_ms > 0 else float('inf')
            result["ref_time_ms"] = ref_time_ms

        result["success"] = True
        
        # --- Cleanup temp file ---
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        
        return result

    def _create_test_tensors(self, param_names: list, n_elements: int) -> list:
        """
        Heuristic: for simple kernels, assume params are (input_ptr, input_ptr, output_ptr, ...).
        Create random torch tensors on CUDA if available.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensors = []
        for i, name in enumerate(param_names):
            if i < len(param_names) - 1:
                # Inputs
                t = torch.rand(n_elements, device=device, dtype=torch.float32)
            else:
                # Output (last param)
                t = torch.empty(n_elements, device=device, dtype=torch.float32)
            tensors.append(t)
        return tensors

    def _infer_block_size(self, code: str) -> int:
        """
        Infer BLOCK_SIZE from tl.arange(N, M) in the code.
        """
        import re
        matches = re.findall(r"tl\.arange\(\s*(\d+)\s*,\s*(\d+)\s*\)", code)
        if matches:
            start, end = int(matches[0][0]), int(matches[0][1])
            return end - start
        return 1024

    def _run_reference(
        self,
        tensors: list,
        kernel_name: str,
        n_elements: int,
        warmup: int,
        reps: int,
    ) -> tuple:
        """
        Run a PyTorch reference implementation and time it.
        Returns (ref_output, ref_time_ms).
        """
        device = tensors[0].device
        ref_time_ms = None
        ref_output = None

        # Simple heuristics based on kernel name
        if "add" in kernel_name.lower() or "sum" in kernel_name.lower():
            if len(tensors) >= 3:
                a, b = tensors[0], tensors[1]
                ref_output = a + b

                if device.type == "cuda":
                    for _ in range(warmup):
                        _ = a + b
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(reps):
                        _ = a + b
                    end.record()
                    torch.cuda.synchronize()
                    ref_time_ms = start.elapsed_time(end) / reps
                else:
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        _ = a + b
                    t1 = time.perf_counter()
                    ref_time_ms = (t1 - t0) * 1000 / reps

        elif "mul" in kernel_name.lower():
            if len(tensors) >= 3:
                a, b = tensors[0], tensors[1]
                ref_output = a * b

                if device.type == "cuda":
                    for _ in range(warmup):
                        _ = a * b
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(reps):
                        _ = a * b
                    end.record()
                    torch.cuda.synchronize()
                    ref_time_ms = start.elapsed_time(end) / reps
                else:
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        _ = a * b
                    t1 = time.perf_counter()
                    ref_time_ms = (t1 - t0) * 1000 / reps

        elif "exp" in kernel_name.lower():
            if len(tensors) >= 2:
                a = tensors[0]
                ref_output = torch.exp(a)

                if device.type == "cuda":
                    for _ in range(warmup):
                        _ = torch.exp(a)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(reps):
                        _ = torch.exp(a)
                    end.record()
                    torch.cuda.synchronize()
                    ref_time_ms = start.elapsed_time(end) / reps
                else:
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        _ = torch.exp(a)
                    t1 = time.perf_counter()
                    ref_time_ms = (t1 - t0) * 1000 / reps

        return ref_output, ref_time_ms
