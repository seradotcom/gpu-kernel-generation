import os
import glob
import json
import re
import argparse

def build_cupy_wrapper(ptx_code: str, instruction: str, mlir_args: list) -> str:
    wrapper_info = instruction.split("Wrapper Entry Information:")[-1].strip()
    func_name = wrapper_info.split("(")[0].strip()
    
    # We parse the arguments from the wrapper info
    m = re.search(r"\((.*?)\)", wrapper_info)
    arg_list = []
    if m:
        arg_str = m.group(1)
        for part in arg_str.split(','):
            part = part.strip()
            if not part: continue
            arg_name = part.split('=')[0].strip().split(':')[0].strip()
            arg_list.append(arg_name)
    
    input_args = [a for a in arg_list if a != 'out' and a != 'alpha']
    
    wrapper_code = f'''
import cupy as cp
import torch
import math
import numpy as np

PTX_CODE = """
{ptx_code}
"""

def {wrapper_info.split("->")[0].strip()}:
    local_vars = locals()
    
    tensors = []
    # Identify input tensors
    for arg_name in {input_args}:
        val = local_vars.get(arg_name)
        if isinstance(val, torch.Tensor):
            tensors.append(val)
            
    if not tensors:
        raise ValueError("No tensor inputs found for GPU kernel execution")
        
    n_elements = tensors[0].numel()
    
    out = local_vars.get('out')
    if out is None:
        out = torch.empty_like(tensors[0])
        
    module = cp.RawModule(code=PTX_CODE)
    ker = module.get_function("{func_name}")
    
    # Pack pointers for CuPy
    # Assuming MLIR expected inputs followed by output, and optionally n_elements
    cupy_args = []
    for t in tensors:
        cupy_args.append(t.data_ptr())
    cupy_args.append(out.data_ptr())
    
    # If MLIR expects more arguments (like n_elements), append them as scalars
    mlir_arg_count = {len(mlir_args)}
    if len(cupy_args) < mlir_arg_count:
        cupy_args.append(np.int32(n_elements))
    
    BLOCK_SIZE = 1024
    grid = (math.ceil(n_elements / BLOCK_SIZE), 1, 1)
    ker(grid, (BLOCK_SIZE, 1, 1), tuple(cupy_args))
    
    return out
'''
    return wrapper_code.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptx_dir", default="benchmark_ptx_output")
    ap.add_argument("--out", default="predictions.jsonl")
    args = ap.parse_args()

    ptx_files = glob.glob(os.path.join(args.ptx_dir, "*.ptx"))
    
    n_ok = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for ptx_path in ptx_files:
            func_name = os.path.basename(ptx_path).replace(".ptx", "")
            json_path = ptx_path.replace(".ptx", ".json")
            
            if not os.path.exists(json_path):
                continue
                
            with open(ptx_path, "r") as f:
                ptx_code = f.read()
                
            with open(json_path, "r") as f:
                meta = json.load(f)
                
            instruction = meta["instruction"]
            mlir_args = meta.get("mlir_args", [])
            
            predict_code = build_cupy_wrapper(ptx_code, instruction, mlir_args)
            
            fout.write(json.dumps({"instruction": instruction, "predict": predict_code},
                                  ensure_ascii=False) + "\n")
            n_ok += 1
            
    print(f"Done. Wrote {n_ok} CuPy wrappers to {args.out}.")

if __name__ == "__main__":
    main()
