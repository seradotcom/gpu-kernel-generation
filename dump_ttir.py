import core.config          # adds your local MLIR bindings to the path
import mlir.ir              # if THIS line errors, your LLVM/MLIR build isn't finished
from core.schemas import MlirResponse
from core.mlir_translator import MLIRTranslator

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

ttir = MLIRTranslator().translate_to_module(MlirResponse(**vec).code)
print(ttir)
