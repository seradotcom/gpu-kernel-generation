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
        # In production, extract dynamically with `torch.cuda.get_device_capability()`.
        # Assuming Ampere (sm_80) or Hopper for research purposes.
        compute_capability = 80  
        
        # Triton internal compilation options.
        # Mimicking what triton.jit does under the hood.
        options = tc.ASTSource(
            fn=None, # No Python function exists
            signature={}, # Inferred from TTIR
            constants={}
        )
        
        # triton.compiler.compile receives a .ttir, .llir, or .ptx string.
        # We explicitly tell it to compile towards the 'ptx' target.
        try:
            # In recent Triton versions (3.0+), `compile` signature varies.
            # We usually use 'src' as the string or make_backend.
            compiled_kernel = tc.compile(
                src=ttir_string,
                target=("cuda", compute_capability),
                options={"num_warps": num_warps, "num_stages": num_stages}
            )
            
            # compiled_kernel contains the final ASM code (PTX) and shared memory used
            return compiled_kernel.asm["ptx"]
            
        except Exception as e:
            raise RuntimeError(f"Triton compiler rejected the TTIR. Lowering failed:\n{e}")
