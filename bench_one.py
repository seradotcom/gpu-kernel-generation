


import requests, json
import core.config          # puts your MLIR bindings on the path
from core.schemas import MlirResponse
from core.mlir_translator import MLIRTranslator

# === paste your current Kaggle PUBLIC URL here (no trailing slash) ===
NGROK_URL = "https://handled-baked-copied.ngrok-free.dev/"


vec = {"reasoning":"vector add","code":{
  "function_name":"vec_sum_kernel",
  "arguments":[{"name":"%arg0_A","type":"!tt.ptr<f32>"},
               {"name":"%arg1_B","type":"!tt.ptr<f32>"},
               {"name":"%arg2_C","type":"!tt.ptr<f32>"}],
  "operations":[
    {"opcode":"tt.make_range","operands":[],"attributes":{"start":0,"end":256},"result":"%offsets","out_type":"tensor<256xi32>"},
    {"opcode":"tt.splat","operands":["%arg0_A"],"result":"%pA","out_type":"tensor<256x!tt.ptr<f32>>"},
    {"opcode":"tt.addptr","operands":["%pA","%offsets"],"result":"%pAo","out_type":"tensor<256x!tt.ptr<f32>>"},
    {"opcode":"tt.load","operands":["%pAo"],"result":"%vA","out_type":"tensor<256xf32>"},
    {"opcode":"tt.splat","operands":["%arg1_B"],"result":"%pB","out_type":"tensor<256x!tt.ptr<f32>>"},
    {"opcode":"tt.addptr","operands":["%pB","%offsets"],"result":"%pBo","out_type":"tensor<256x!tt.ptr<f32>>"},
    {"opcode":"tt.load","operands":["%pBo"],"result":"%vB","out_type":"tensor<256xf32>"},
    {"opcode":"arith.addf","operands":["%vA","%vB"],"result":"%vC","out_type":"tensor<256xf32>"},
    {"opcode":"tt.splat","operands":["%arg2_C"],"result":"%pC","out_type":"tensor<256x!tt.ptr<f32>>"},
    {"opcode":"tt.addptr","operands":["%pC","%offsets"],"result":"%pCo","out_type":"tensor<256x!tt.ptr<f32>>"},
    {"opcode":"tt.store","operands":["%pCo","%vC"],"result":"none"}],
  "returns":[]}}

# 1. translate JSON -> TTIR locally (uses your Mac's MLIR bindings)
ttir = MLIRTranslator().translate_to_module(MlirResponse(**vec).code)
print("translated TTIR locally, length:", len(ttir))

# 2. send TTIR to the GPU endpoint, get metrics back
resp = requests.post(
    f"{NGROK_URL.rstrip('/')}/benchmark",
    json={"name": "vec_sum_kernel", "ttir": ttir},
    headers={"ngrok-skip-browser-warning": "true"},
    timeout=120,
)
print("HTTP", resp.status_code)
row = resp.json()
print(json.dumps(row, indent=2))

# 3. record locally
with open("results.json", "w") as f:
    json.dump([row], f, indent=2)
print("wrote results.json")