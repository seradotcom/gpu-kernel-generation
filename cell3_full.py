import torch, triton, tempfile, json
from triton.backends.compiler import GPUTarget

_cc = torch.cuda.get_device_capability()      # L4 -> (8, 9); T4 -> (7, 5)
TARGET = GPUTarget("cuda", _cc[0]*10 + _cc[1], 32)
print("compute capability:", _cc, "->", _cc[0]*10 + _cc[1])

def compile_ttir(ttir):
    import os
    with tempfile.NamedTemporaryFile(suffix=".ttir", delete=False, mode="w") as f:
        f.write(ttir); path = f.name
    errf = tempfile.NamedTemporaryFile(suffix=".err", delete=False, mode="w+")
    saved = os.dup(2)
    os.dup2(errf.fileno(), 2)
    try:
        k = triton.compile(path, target=TARGET)
    except Exception as e:
        os.dup2(saved, 2); os.close(saved)
        errf.flush(); errf.seek(0); diag = errf.read(); errf.close()
        raise RuntimeError(f"{repr(e)} | MLIR: {diag.strip()[:800]}")
    os.dup2(saved, 2); os.close(saved); errf.close()
    return k

def benchmark_kernel(name, ttir, make_inputs, ref_fn, grid, atol=1e-5):
    rec = {"name": name, "status": None, "correct": None, "max_err": None,
           "latency_ms": None, "torch_ms": None, "speedup": None,
           "shared_mem": None, "num_warps": None, "error": None}
    try:
        k = compile_ttir(ttir)
    except Exception as e:
        rec["status"] = "compile_failed"; rec["error"] = repr(e); return rec
    md_ = getattr(k, "metadata", None)
    rec["shared_mem"] = getattr(md_, "shared", None)
    rec["num_warps"] = getattr(md_, "num_warps", None)
    args, out, ref_inputs = make_inputs()
    expected = ref_fn(*ref_inputs)
    try:
        k[grid](*args); torch.cuda.synchronize()
    except Exception as e:
        rec["status"] = "launch_failed"; rec["error"] = repr(e); return rec
    rec["correct"] = bool(torch.allclose(out, expected, atol=atol))
    rec["max_err"] = (out - expected).abs().max().item()
    if not rec["correct"]:
        rec["status"] = "exec_incorrect"; return rec
    rec["latency_ms"] = triton.testing.do_bench(lambda: k[grid](*args))
    rec["torch_ms"]   = triton.testing.do_bench(lambda: ref_fn(*ref_inputs))
    rec["speedup"]    = rec["torch_ms"] / rec["latency_ms"] if rec["latency_ms"] else None
    rec["status"] = "success"
    return rec

# ---------- input builders (fixed golden references, never generated) ----------
N = 256
def two_in():     A=torch.rand(N,device="cuda"); B=torch.rand(N,device="cuda"); C=torch.empty_like(A); return [A,B,C], C, (A,B)
def div_in():     A=torch.rand(N,device="cuda"); B=torch.rand(N,device="cuda")+0.5; C=torch.empty_like(A); return [A,B,C], C, (A,B)
def one_in():     A=torch.rand(N,device="cuda"); C=torch.empty_like(A); return [A,C], C, (A,)
def signed_in():  A=torch.rand(N,device="cuda")*2-1; C=torch.empty_like(A); return [A,C], C, (A,)
def pos_in():     A=torch.rand(N,device="cuda")+1.0; C=torch.empty_like(A); return [A,C], C, (A,)
def three_in():   A=torch.rand(N,device="cuda"); B=torch.rand(N,device="cuda"); Cc=torch.rand(N,device="cuda"); D=torch.empty_like(A); return [A,B,Cc,D], D, (A,B,Cc)

KERNEL_TESTS = {
  # --- easy: single-op elementwise ---
  "vec_add_kernel":      {"make_inputs": two_in,    "ref_fn": lambda a,b: a+b,                 "grid": (1,1,1), "atol": 1e-5},
  "vec_sub_kernel":      {"make_inputs": two_in,    "ref_fn": lambda a,b: a-b,                 "grid": (1,1,1), "atol": 1e-5},
  "vec_mul_kernel":      {"make_inputs": two_in,    "ref_fn": lambda a,b: a*b,                 "grid": (1,1,1), "atol": 1e-5},
  "vec_div_kernel":      {"make_inputs": div_in,    "ref_fn": lambda a,b: a/b,                 "grid": (1,1,1), "atol": 1e-4},
  "vec_max_kernel":      {"make_inputs": two_in,    "ref_fn": lambda a,b: torch.maximum(a,b),  "grid": (1,1,1), "atol": 1e-5},
  "vec_min_kernel":      {"make_inputs": two_in,    "ref_fn": lambda a,b: torch.minimum(a,b),  "grid": (1,1,1), "atol": 1e-5},
  "vec_exp_kernel":      {"make_inputs": one_in,    "ref_fn": lambda a: torch.exp(a),          "grid": (1,1,1), "atol": 1e-4},
  "vec_sqrt_kernel":     {"make_inputs": one_in,    "ref_fn": lambda a: torch.sqrt(a),         "grid": (1,1,1), "atol": 1e-4},
  "vec_log_kernel":      {"make_inputs": pos_in,    "ref_fn": lambda a: torch.log(a),          "grid": (1,1,1), "atol": 1e-4},
  "vec_abs_kernel":      {"make_inputs": signed_in, "ref_fn": lambda a: torch.abs(a),          "grid": (1,1,1), "atol": 1e-5},
  "vec_neg_kernel":      {"make_inputs": signed_in, "ref_fn": lambda a: -a,                    "grid": (1,1,1), "atol": 1e-5},
  "vec_square_kernel":   {"make_inputs": signed_in, "ref_fn": lambda a: a*a,                   "grid": (1,1,1), "atol": 1e-5},
  # --- medium: constants / fused ---
  "vec_relu_kernel":     {"make_inputs": signed_in, "ref_fn": lambda a: torch.relu(a),         "grid": (1,1,1), "atol": 1e-5},
  "vec_scale2_kernel":   {"make_inputs": one_in,    "ref_fn": lambda a: a*2.0,                 "grid": (1,1,1), "atol": 1e-5},
  "vec_addscalar_kernel":{"make_inputs": one_in,    "ref_fn": lambda a: a+1.0,                 "grid": (1,1,1), "atol": 1e-5},
  "vec_fma_kernel":      {"make_inputs": three_in,  "ref_fn": lambda a,b,c: a*b+c,             "grid": (1,1,1), "atol": 1e-4},
  "vec_sqdiff_kernel":   {"make_inputs": two_in,    "ref_fn": lambda a,b: (a-b)*(a-b),         "grid": (1,1,1), "atol": 1e-5},
}

def run_benchmark_request(name, ttir):
    spec = KERNEL_TESTS.get(name)
    if spec is None:
        return {"name": name, "status": "unknown_kernel", "error": f"no test registered for '{name}'"}
    return benchmark_kernel(name, ttir, spec["make_inputs"], spec["ref_fn"], spec["grid"], spec.get("atol", 1e-5))

# known-good reference TTIR (smoke test)
VEC_TTIR = r"""module {
  "tt.func"() ({
  ^bb0(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>):
    %0 = "tt.make_range"() {end = 256 : i32, start = 0 : i32} : () -> tensor<256xi32>
    %1 = "tt.splat"(%arg0) : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %2 = "tt.addptr"(%1, %0) : (tensor<256x!tt.ptr<f32>>, tensor<256xi32>) -> tensor<256x!tt.ptr<f32>>
    %3 = "tt.load"(%2) {operandSegmentSizes = array<i32: 1, 0, 0>} : (tensor<256x!tt.ptr<f32>>) -> tensor<256xf32>
    %4 = "tt.splat"(%arg1) : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %5 = "tt.addptr"(%4, %0) : (tensor<256x!tt.ptr<f32>>, tensor<256xi32>) -> tensor<256x!tt.ptr<f32>>
    %6 = "tt.load"(%5) {operandSegmentSizes = array<i32: 1, 0, 0>} : (tensor<256x!tt.ptr<f32>>) -> tensor<256xf32>
    %7 = arith.addf %3, %6 : tensor<256xf32>
    %8 = "tt.splat"(%arg2) : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %9 = "tt.addptr"(%8, %0) : (tensor<256x!tt.ptr<f32>>, tensor<256xi32>) -> tensor<256x!tt.ptr<f32>>
    "tt.store"(%9, %7) {operandSegmentSizes = array<i32: 1, 1, 0>} : (tensor<256x!tt.ptr<f32>>, tensor<256xf32>) -> ()
    "tt.return"() : () -> ()
  }) {function_type = (!tt.ptr<f32>, !tt.ptr<f32>, !tt.ptr<f32>) -> (), sym_name = "vec_sum_kernel"} : () -> ()
}"""

print("benchmark core + registry ready:", len(KERNEL_TESTS), "kernels")
