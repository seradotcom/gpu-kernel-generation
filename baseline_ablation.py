import os
import json
import random
import time
import re
import argparse
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:
    print("WARNING: torch/triton not found. This script should be run where the GPU is available.")

# Pytorch references
unary_ops = {
    "vec_abs": torch.abs, "vec_sin": torch.sin, "vec_cos": torch.cos,
    "vec_exp": torch.exp, "vec_log": torch.log, "vec_sqrt": torch.sqrt,
    "vec_rsqrt": torch.rsqrt, "vec_negf": torch.neg, "vec_ceil": torch.ceil,
    "vec_floor": torch.floor, "vec_trunc": torch.trunc,
    "vec_addc": lambda a: a + 3.14, "vec_subc": lambda a: a - 2.71,
    "vec_mulc": lambda a: a * 1.618, "vec_divc": lambda a: a / 1.414,
    "vec_relu": lambda a: torch.clamp(a, min=0.0),
    "vec_sigmoid": torch.sigmoid,
    "vec_silu": lambda a: a * torch.sigmoid(a),
    "vec_gelu": torch.nn.functional.gelu,
    "vec_tanh": torch.tanh,
    "vec_clamp": lambda a: torch.clamp(a, min=-1.0, max=1.0),
    "vec_leaky_relu": lambda a: torch.nn.functional.leaky_relu(a, 0.01),
    "vec_elu": torch.nn.functional.elu,
    "vec_softplus": torch.nn.functional.softplus,
    "vec_sum": torch.sum,
    "vec_max": torch.max,
    "vec_min": torch.min,
    "vec_mean": torch.mean,
    "vec_var": torch.var,
    "vec_softmax": lambda a: torch.nn.functional.softmax(a, dim=-1),
    "vec_rmsnorm": lambda a: a / torch.sqrt(torch.mean(a*a) + 1e-5),
    "vec_layernorm": lambda a: (a - torch.mean(a)) / torch.sqrt(torch.var(a, unbiased=False) + 1e-5),
}

binary_ops = {
    "vec_add": lambda a, b: a + b,
    "vec_sub": lambda a, b: a - b,
    "vec_mul": lambda a, b: a * b,
    "vec_div": lambda a, b: a / b,
    "vec_maximum": torch.maximum,
    "vec_minimum": torch.minimum,
    "vec_madd": lambda a, b: a * b + 0.5,
    "vec_submul": lambda a, b: (a - b) * 0.5,
    "vec_lerp": lambda a, b: a + (b - a) * 0.5,
    "vec_dot": lambda a, b: torch.sum(a * b),
    "vec_cosine_sim": lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=0),
    "vec_euclidean": lambda a, b: torch.norm(a - b),
}

def get_torch_op(basename):
    # Dynamic affine
    match = re.search(r'aff(?:ine|2)_m([0-9p]+)_c([0-9p]+)', basename)
    if match:
        m = float(match.group(1).replace('p', '.'))
        c = float(match.group(2).replace('p', '.'))
        return lambda a, m=m, c=c: a * m + c, "unary"
    
    if "quad" in basename: return lambda a: a * a * a * a, "unary"
    if "halfsq" in basename: return lambda a: (a * a) * 0.5, "unary"
    if "minm1" in basename: return lambda a: torch.minimum(a, torch.tensor(-1.0, device='cuda')), "unary"
    if "relu6" in basename: return lambda a: torch.clamp(a, min=0.0, max=6.0), "unary"
    if "hardswish" in basename: return lambda a: a * torch.clamp(a + 3.0, 0.0, 6.0) / 6.0, "unary"

    # Scale modifiers
    scale = 1.0
    match = re.search(r'(?:scale|scaled|t)([0-9p]+)_', basename)
    if match:
        scale = float(match.group(1).replace('p', '.'))

    match = re.search(r'vec_mulc([0-9p]+)_', basename)
    if match: return lambda a: a * float(match.group(1).replace('p', '.')), "unary"
    match = re.search(r'vec_addc([0-9p]+)_', basename)
    if match: return lambda a: a + float(match.group(1).replace('p', '.')), "unary"
    match = re.search(r'vec_subc([0-9p]+)_', basename)
    if match: return lambda a: a - float(match.group(1).replace('p', '.')), "unary"
    match = re.search(r'vec_divc([0-9p]+)_', basename)
    if match: return lambda a: a / float(match.group(1).replace('p', '.')), "unary"
        
    for op_name, fn in unary_ops.items():
        if basename.startswith(op_name): 
            if scale != 1.0:
                return lambda a, f=fn, s=scale: f(a) * s, "unary"
            return fn, "unary"
    for op_name, fn in binary_ops.items():
        if basename.startswith(op_name): return fn, "binary"
    return None, None

def evaluate_python_kernel(kernel_code, kernel_name, basename):
    torch_op, op_type = get_torch_op(basename)
    if not torch_op:
        if "mulc" in basename or "addc" in basename or "subc" in basename or "divc" in basename:
            torch_op, op_type = get_torch_op(basename) 
        if not torch_op: return None, "Op not supported in test"

    import tempfile
    import importlib.util
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import triton\nimport triton.language as tl\nimport torch\n" + kernel_code)
        tmp_filename = f.name

    try:
        spec = importlib.util.spec_from_file_location("dynamic_kernel", tmp_filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        os.remove(tmp_filename)
    except Exception as e:
        os.remove(tmp_filename)
        return False, f"Compilation Error: {e}"

    kernel_func = getattr(module, kernel_name, None)
    if not kernel_func:
        for k in dir(module):
            v = getattr(module, k)
            if hasattr(v, "cache"):
                kernel_func = v
                break
        if not kernel_func: return False, "Could not find @triton.jit function"

    N = 256
    a = torch.rand(N, dtype=torch.float32, device='cuda')
    b = torch.rand(N, dtype=torch.float32, device='cuda')
    out = torch.zeros(N, dtype=torch.float32, device='cuda')

    try:
        import inspect
        sig = inspect.signature(kernel_func.fn)
        num_args = len(sig.parameters)
        grid = (1,)
        
        if num_args == 4:
            kernel_func[grid](a, out, N, BLOCK_SIZE=N)
            ref = torch_op(a)
        elif num_args == 5:
            kernel_func[grid](a, b, out, N, BLOCK_SIZE=N)
            ref = torch_op(a, b)
        else:
            return False, f"Unexpected arity: {num_args} args"
            
        torch.cuda.synchronize()
        correct = torch.allclose(ref, out, atol=1e-3, rtol=1e-3)
        return correct, "Correct" if correct else "Wrong Calculation"
    except Exception as e:
        err_str = str(e)
        if "CUDA" in err_str or "memory" in err_str: return False, "Memory Violation"
        return False, f"Runtime Error: {err_str}"

def ask_vllm_local(prompt):
    url = "http://localhost:8000/generate"
    payload = {
        "prompt": f"<|im_start|>system\nYou are a GPU kernel engineer. Write ONLY python code.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "max_tokens": 1024,
        "temperature": 0.2
    }
    
    res = requests.post(url, json=payload, timeout=300)
    res.raise_for_status()
    job_id = res.json()["job_id"]
    
    while True:
        time.sleep(2)
        status_res = requests.get(f"http://localhost:8000/status/{job_id}").json()
        if status_res["status"] == "done":
            return status_res["response"]
        elif status_res["status"] == "error":
            raise Exception(status_res["error"])

def process_tier(tier_file, num_samples):
    if "EASY" in tier_file.upper() or "easy" in tier_file.lower(): tier = "EASY"
    elif "MED" in tier_file.upper() or "med" in tier_file.lower(): tier = "MED"
    else: tier = "HARD"
    
    print(f"\n[Worker] Empezando a evaluar nivel: {tier}")
    
    try:
        with open(tier_file, "r") as fp:
            prompts = json.load(fp)
    except FileNotFoundError:
        print(f"[{tier}] ERROR: No se encontró el archivo {tier_file}. Debes estar en la carpeta donde están los JSON.")
        return tier, 0.0
        
    keys = list(prompts.keys())
    # random.shuffle(keys)  # Quitamos el shuffle para que los procese ordenadamente
    sampled = keys[:num_samples]
    passed = 0
    
    for key in sampled:
        instruction = prompts[key]["prompt"]
        user_p = f"""Write a Python Triton kernel for the following operation: {instruction}
Requirements:
1. Define a `@triton.jit` kernel named `{key}`.
2. The kernel should expect tensor pointers (e.g., `in_ptr0`, `out_ptr`), `n_elements`, and a `BLOCK_SIZE: tl.constexpr`.
3. Process a single block of exactly 256 elements.
4. Return ONLY valid Python code wrapped in ```python ... ```. Do NOT include tests or PyTorch execution logic."""

        try:
            response = ask_vllm_local(user_p)
            match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
            code = match.group(1) if match else response.replace('```python', '').replace('```', '')
            
            correct, msg = evaluate_python_kernel(code, key, key)
            if correct:
                print(f"[{tier}] {key} -> PASS")
                passed += 1
            else:
                print(f"[{tier}] {key} -> FAIL ({msg})")
        except Exception as e:
            print(f"[{tier}] {key} -> LLM/Connection Error: {e}")
            
    rate = (passed / num_samples) * 100 if num_samples > 0 else 0
    print(f"\n[{tier}] Finalizado. Tasa de éxito: {passed}/{num_samples} ({rate:.1f}%)")
    return tier, rate

def run_baseline(num_samples=100, parallel=False, target_tier="all"):
    all_files = ["benchmark_prompts_EASY100.json", "benchmark_prompts_MED100.json", "benchmark_prompts_HARD100.json"]
    
    if target_tier.lower() == "easy":
        files = ["benchmark_prompts_EASY100.json"]
    elif target_tier.lower() == "med":
        files = ["benchmark_prompts_MED100.json"]
    elif target_tier.lower() == "hard":
        files = ["benchmark_prompts_HARD100.json"]
    else:
        files = all_files
        
    final_results = {}
    
    if parallel:
        print(f"🚀 Iniciando evaluación en PARALELO ({target_tier})...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_tier, f, num_samples) for f in files]
            for future in as_completed(futures):
                tier, rate = future.result()
                final_results[tier] = rate
    else:
        print(f"Iniciando evaluación secuencial ({target_tier})...")
        for f in files:
            tier, rate = process_tier(f, num_samples)
            final_results[tier] = rate

    print("\n" + "="*40)
    print("=== BASELINE FINAL (Raw LLM to Triton) ===")
    print("="*40)
    for tier in ["EASY", "MED", "HARD"]:
        if tier in final_results:
            print(f" - {tier}: {final_results[tier]:.1f}%")
    print("="*40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--tier", type=str, default="all", choices=["all", "easy", "med", "hard"])
    args = parser.parse_args()
    run_baseline(args.samples, args.parallel, args.tier)
