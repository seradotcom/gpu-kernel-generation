import os
import sys

# If triton is installed globally or in the venv, import it.
try:
    import triton
    import triton.compiler as tc
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

class TritonBackend:
    """
    Handles the lowering of TTIR (Triton Textual IR) to native NVIDIA code (PTX).
    Injects the code directly into the compiler internals, bypassing the standard @triton.jit frontend.
    """
    def __init__(self, target_architecture: str = "cuda"):
        """
        Initializes the Triton compiler backend.
        
        Args:
            target_architecture (str): Target hardware architecture, defaults to "cuda".
        """
        self.target = target_architecture
        if not HAS_TRITON:
            print("[Warning] Triton is not installed. Compilation of TTIR to physical PTX will fail.")
            print("Try: pip install triton")
    def _fix_ttir_syntax(self, ttir_str: str) -> str:
        import re
        m = re.search(r'sym_name = "([^"]+)"', ttir_str)
        if not m: return ttir_str
        sym_name = m.group(1)
        
        m_bb0 = re.search(r'\^bb0\((.*?)\):', ttir_str)
        args_str = m_bb0.group(1) if m_bb0 else ""
        
        if m_bb0:
            ttir_str = re.sub(r'^\s*\^bb0\(.*?\):\n', '', ttir_str, count=1, flags=re.MULTILINE)
            
        pattern = r'"tt\.func"\(\) \(\{\n(.*?)^\s*\}\) \{function_type = \(.*?\) -> \(.*?\), sym_name = "[^"]+"\} : \(\) -> \(\)'
        replacement = f"tt.func public @{sym_name}({args_str}) {{\n\\1}}"
        return re.sub(pattern, replacement, ttir_str, flags=re.MULTILINE | re.DOTALL)

    def compile_ttir_to_ptx(self, ttir_string: str, num_warps: int = 4, num_stages: int = 3) -> str:
        """
        Executes the pipeline: TTIR -> TTGIR -> LLVM IR -> PTX.
        
        Args:
            ttir_string (str): The giant string containing the MLIR code in `tt` dialect.
            num_warps (int): Number of warps per block (usually 4 or 8).
            num_stages (int): Number of software pipelining stages (for AsyncCopy).

        Returns:
            str: The pure PTX assembly code.

        Raises:
            ImportError: If Triton is not installed.
            RuntimeError: If the Triton compiler rejects the TTIR.
        """
        if not HAS_TRITON:
            raise ImportError("Triton is not installed in this environment.")
            
        # Target hardware parameters.
        # In production, this is dynamically extracted via `torch.cuda.get_device_capability()`.
        # GTX 1650 uses Turing architecture (Compute Capability 7.5 -> 75)
        compute_capability = 75  
        
        # Triton internal compilation options.
        import tempfile
        from triton.backends.compiler import GPUTarget
        import multiprocessing

        def _compile_worker(src, target, options, return_dict):
            try:
                import triton.compiler as tc
                compiled_kernel = tc.compile(src=src, target=target, options=options)
                return_dict["ptx"] = compiled_kernel.asm["ptx"]
            except Exception as e:
                return_dict["error"] = str(e)

        try:
            ttir_string = self._fix_ttir_syntax(ttir_string)
            target = GPUTarget("cuda", compute_capability, 32)
            with tempfile.NamedTemporaryFile(suffix=".ttir", delete=False) as f:
                f.write(ttir_string.encode("utf-8"))
                temp_filename = f.name
                
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            p = multiprocessing.Process(
                target=_compile_worker, 
                args=(temp_filename, target, {"num_warps": num_warps, "num_stages": num_stages}, return_dict)
            )
            p.start()
            p.join()
            
            # Remove temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
            if p.exitcode != 0:
                raise RuntimeError(f"Triton compiler crashed with exit code {p.exitcode} (Segmentation Fault / core dumped).")
                
            if "error" in return_dict:
                raise RuntimeError(return_dict["error"])
                
            if "ptx" not in return_dict:
                raise RuntimeError("Unknown compilation failure: no PTX generated.")
                
            return return_dict["ptx"]
            
        except Exception as e:
            raise RuntimeError(f"Triton compiler rejected the TTIR. Lowering failed:\n{e}")
