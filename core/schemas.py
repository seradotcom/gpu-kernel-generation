from enum import Enum
from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# SCF & Triton Schema (Structured Outputs)
# ---------------------------------------------------------

class MLIRType(str, Enum):
    """
    Tipos MLIR estrictos. Se usa un Enum para evitar que el LLM alucine
    tamaños de bloque subóptimos (ej. 137) obligándolo a usar potencias de 2.
    """
    # === Escalares base ===
    F32   = "f32"
    F16   = "f16"
    I32   = "i32"
    I1    = "i1"
    INDEX = "index"

    # === Punteros base ===
    PTR_F32 = "!tt.ptr<f32>"
    PTR_F16 = "!tt.ptr<f16>"
    PTR_I32 = "!tt.ptr<i32>"

    # === 1D Tensors: Datos Matemáticos (f32 / f16) ===
    T_16_F32   = "tensor<16xf32>"
    T_32_F32   = "tensor<32xf32>"
    T_64_F32   = "tensor<64xf32>"
    T_128_F32  = "tensor<128xf32>"
    T_256_F32  = "tensor<256xf32>"
    T_512_F32  = "tensor<512xf32>"
    T_1024_F32 = "tensor<1024xf32>"
    T_2048_F32 = "tensor<2048xf32>"

    T_16_F16   = "tensor<16xf16>"
    T_32_F16   = "tensor<32xf16>"
    T_64_F16   = "tensor<64xf16>"
    T_128_F16  = "tensor<128xf16>"
    T_256_F16  = "tensor<256xf16>"
    T_512_F16  = "tensor<512xf16>"
    T_1024_F16 = "tensor<1024xf16>"
    T_2048_F16 = "tensor<2048xf16>"

    # === 1D Tensors: Índices/Offsets (Para tt.make_range) ===
    T_16_I32   = "tensor<16xi32>"
    T_32_I32   = "tensor<32xi32>"
    T_64_I32   = "tensor<64xi32>"
    T_128_I32  = "tensor<128xi32>"
    T_256_I32  = "tensor<256xi32>"
    T_512_I32  = "tensor<512xi32>"
    T_1024_I32 = "tensor<1024xi32>"
    T_2048_I32 = "tensor<2048xi32>"

    # === 1D Tensors: Máscaras Booleanas (Para arith.cmpf y tt.load) ===
    T_16_I1   = "tensor<16xi1>"
    T_32_I1   = "tensor<32xi1>"
    T_64_I1   = "tensor<64xi1>"
    T_128_I1  = "tensor<128xi1>"
    T_256_I1  = "tensor<256xi1>"
    T_512_I1  = "tensor<512xi1>"
    T_1024_I1 = "tensor<1024xi1>"
    T_2048_I1 = "tensor<2048xi1>"

    # === 1D Tensors: Tensores de Punteros (Para tt.splat y tt.addptr) ===
    T_16_PTR_F32   = "tensor<16x!tt.ptr<f32>>"
    T_32_PTR_F32   = "tensor<32x!tt.ptr<f32>>"
    T_64_PTR_F32   = "tensor<64x!tt.ptr<f32>>"
    T_128_PTR_F32  = "tensor<128x!tt.ptr<f32>>"
    T_256_PTR_F32  = "tensor<256x!tt.ptr<f32>>"
    T_512_PTR_F32  = "tensor<512x!tt.ptr<f32>>"
    T_1024_PTR_F32 = "tensor<1024x!tt.ptr<f32>>"
    T_2048_PTR_F32 = "tensor<2048x!tt.ptr<f32>>"

    T_16_PTR_F16   = "tensor<16x!tt.ptr<f16>>"
    T_32_PTR_F16   = "tensor<32x!tt.ptr<f16>>"
    T_64_PTR_F16   = "tensor<64x!tt.ptr<f16>>"
    T_128_PTR_F16  = "tensor<128x!tt.ptr<f16>>"
    T_256_PTR_F16  = "tensor<256x!tt.ptr<f16>>"
    T_512_PTR_F16  = "tensor<512x!tt.ptr<f16>>"
    T_1024_PTR_F16 = "tensor<1024x!tt.ptr<f16>>"
    T_2048_PTR_F16 = "tensor<2048x!tt.ptr<f16>>"

    # === 2D Tensors: Bloques para Matmul, Softmax 2D, LayerNorm ===
    # (Tamaños simétricos)
    T_16x16_F32   = "tensor<16x16xf32>"
    T_32x32_F32   = "tensor<32x32xf32>"
    T_64x64_F32   = "tensor<64x64xf32>"
    T_128x128_F32 = "tensor<128x128xf32>"
    T_256x256_F32 = "tensor<256x256xf32>"

    # (Tamaños asimétricos típicos)
    T_32x64_F32   = "tensor<32x64xf32>"
    T_64x32_F32   = "tensor<64x32xf32>"
    T_64x128_F32  = "tensor<64x128xf32>"
    T_128x64_F32  = "tensor<128x64xf32>"
    
    # (Lo mismo para i32 (offsets), i1 (máscaras) y Punteros en 2D)
    T_16x16_I32   = "tensor<16x16xi32>"
    T_32x32_I32   = "tensor<32x32xi32>"
    T_64x64_I32   = "tensor<64x64xi32>"
    T_128x128_I32 = "tensor<128x128xi32>"

    T_16x16_I1   = "tensor<16x16xi1>"
    T_32x32_I1   = "tensor<32x32xi1>"
    T_64x64_I1   = "tensor<64x64xi1>"
    T_128x128_I1 = "tensor<128x128xi1>"

    T_16x16_PTR_F32   = "tensor<16x16x!tt.ptr<f32>>"
    T_32x32_PTR_F32   = "tensor<32x32x!tt.ptr<f32>>"
    T_64x64_PTR_F32   = "tensor<64x64x!tt.ptr<f32>>"
    T_128x128_PTR_F32 = "tensor<128x128x!tt.ptr<f32>>"
    
class MlirOpcode(str, Enum):
    # Arith Dialect
    ARITH_ADDF = "arith.addf"
    ARITH_SUBF = "arith.subf"
    ARITH_MULF = "arith.mulf"
    ARITH_DIVF = "arith.divf"
    ARITH_MAXIMUMF = "arith.maximumf"
    ARITH_MINIMUMF = "arith.minimumf"
    ARITH_CMPF = "arith.cmpf"
    ARITH_CONSTANT = "arith.constant"
    ARITH_EXTF = "arith.extf"
    ARITH_TRUNCF = "arith.truncf"
    ARITH_SELECT = "arith.select"
    
    # Math Dialect
    MATH_EXP = "math.exp"
    MATH_LOG = "math.log"
    MATH_SQRT = "math.sqrt"
    MATH_COS = "math.cos"
    MATH_SIN = "math.sin"
    MATH_ABS = "math.absf"

    # Tensor Dialect
    # Removed: tensor.empty, tensor.extract, tensor.insert
    # Triton MLIR requires block-based pointer operations instead of scalar tensor indexing.

    # Triton (tt) Dialect
    TT_LOAD = "tt.load"
    TT_STORE = "tt.store"
    TT_SPLAT = "tt.splat"
    TT_BROADCAST = "tt.broadcast"
    TT_EXPAND_DIMS = "tt.expand_dims"
    TT_DOT = "tt.dot"
    TT_REDUCE = "tt.reduce"
    TT_RESHAPE = "tt.reshape"
    TT_TRANS = "tt.trans"
    TT_MAKE_RANGE = "tt.make_range"
    TT_ADVANCE = "tt.advance"
    TT_ADD_PTR = "tt.addptr"
    TT_PTR_TO_INT = "tt.ptr_to_int"
    TT_INT_TO_PTR = "tt.int_to_ptr"
    TT_REDUCE_RETURN = "tt.reduce.return"
    TT_GET_PROGRAM_ID = "tt.get_program_id"
    TT_GET_NUM_PROGRAMS = "tt.get_num_programs"
    TT_ATOMIC_RMW = "tt.atomic_rmw"
    TT_ATOMIC_CAS = "tt.atomic_cas"
    TT_RAND = "tt.rand"
    
    # Triton GPU (ttg) Dialect
    TTG_LOCAL_ALLOC = "ttg.local_alloc"
    TTG_LOCAL_LOAD = "ttg.local_load"
    TTG_LOCAL_STORE = "ttg.local_store"
    TTG_CONVERT_LAYOUT = "ttg.convert_layout"
    TTG_ASYNC_WAIT = "ttg.async_wait"
    TTG_ASYNC_COMMIT_GROUP = "ttg.async_commit_group"
    TTG_ASYNC_COPY_GLOBAL_TO_LOCAL = "ttg.async_copy_global_to_local"

class InputArgument(BaseModel):
    """
    Represents an input argument for the function.
    The LLM MUST provide the explicit type here.
    """
    name: str = Field(..., pattern=r"^%[a-zA-Z0-9_]{1,30}$", description="Register name, e.g., '%arg0'")
    type: MLIRType = Field(..., description="Exact MLIR type. Choose from the available enum.")

class UnaryOperation(BaseModel):
    opcode: Literal["math.exp", "math.log", "math.sqrt", "math.cos", "math.sin", "math.absf", "arith.extf", "arith.truncf"] = Field(..., description="Ops with exactly 1 operand")
    operands: List[Union[str, float, int]] = Field(..., min_length=1, max_length=1, description="Input registers or literal numbers")
    result: str = Field(..., pattern=r"^(?:%[a-zA-Z0-9_]{1,30}|none)$", description="Output register")
    out_type: Optional[MLIRType] = Field(None, description="Specify ONLY for explicit casting or when type cannot be inferred.")
    attributes: Optional[Dict[str, Any]] = Field(None)

class BinaryOperation(BaseModel):
    opcode: Literal["arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.maximumf", "arith.minimumf", "arith.cmpf"] = Field(..., description="Ops with exactly 2 operands")
    operands: List[Union[str, float, int]] = Field(..., min_length=2, max_length=2, description="Input registers or literal numbers")
    result: str = Field(..., pattern=r"^(?:%[a-zA-Z0-9_]{1,30}|none)$", description="Output register")
    out_type: Optional[MLIRType] = Field(None, description="Specify ONLY for explicit casting or when type cannot be inferred.")
    attributes: Optional[Dict[str, Any]] = Field(None)

class GenericOperation(BaseModel):
    opcode: MlirOpcode = Field(..., description="Allowed MLIR/Triton operation")
    operands: List[Union[str, float, int]] = Field(..., description="Input registers or literal numbers")
    result: str = Field(..., pattern=r"^(?:%[a-zA-Z0-9_]{1,30}|none)$", description="Output register")
    out_type: Optional[MLIRType] = Field(None, description="Specify ONLY for explicit casting or when type cannot be inferred.")
    attributes: Optional[Dict[str, Any]] = Field(None)
    region_combiner: Optional[str] = Field(None)

AnyOperation = Union[UnaryOperation, BinaryOperation, GenericOperation]

class ScfYield(BaseModel):
    """
    Return operation within an SCF block.
    """
    opcode: Literal["scf.yield"] = Field("scf.yield", description="Must always be 'scf.yield'")
    operands: List[Union[str, float, int]] = Field(default_factory=list, description="Values yielded to the next loop iter_arg or out of the if-block")

class ScfIf(BaseModel):
    """
    If-then-else block in SCF.
    """
    opcode: Literal["scf.if"] = Field("scf.if", description="Must always be 'scf.if'")
    condition: str = Field(..., description="Register containing the boolean (i1) condition")
    results: List[str] = Field(default_factory=list, description="Names of the registers where the scf.if result is stored")
    then_body: List[Union[AnyOperation, 'ScfForLoop', 'ScfIf', 'ScfYield']] = Field(...)
    else_body: Optional[List[Union[AnyOperation, 'ScfForLoop', 'ScfIf', 'ScfYield']]] = Field(None)

class ScfForLoop(BaseModel):
    """
    For loop (scf.for) with support for loop-carried variables (iter_args).
    """
    opcode: Literal["scf.for"] = Field("scf.for", description="Must always be 'scf.for'")
    lower_bound: Union[int, str] = Field(..., description="Lower bound (constant integer or register)")
    upper_bound: Union[int, str] = Field(..., description="Upper bound")
    step: Union[int, str] = Field(..., description="Step size")
    loop_var: str = Field(..., description="Iterator register, e.g., '%i'")
    iter_args: Dict[str, Union[str, float, int]] = Field(..., description="Dictionary {Iter_Arg_Name: Initial_Value}. E.g. {'%sum': 0.0}")
    results: List[str] = Field(..., description="MUST have the EXACT same length as iter_args. These are the final values of the iter_args after the loop finishes.")
    body: List[Union[AnyOperation, 'ScfForLoop', 'ScfIf', 'ScfYield']] = Field(...)

# Rebuild recursive references (required in Pydantic for recursive Unions)
ScfIf.model_rebuild()
ScfForLoop.model_rebuild()

class MLIRFunctionBody(BaseModel):
    """
    Complete function structure.
    """
    function_name: str
    arguments: List[InputArgument] = Field(..., description="Input variables WITH their types")
    operations: List[Union[AnyOperation, ScfForLoop, ScfIf]] = Field(..., description="Body of operations")
    returns: List[str] = Field(..., description="Registers returned by the function")

class MlirResponse(BaseModel):
    """
    Final contract for the LLM response (Structured Output).
    """
    reasoning: str = Field(..., description="Keep this extremely brief (max 3-4 sentences). State your plan and register mapping concisely.")
    code: MLIRFunctionBody = Field(..., description="The program compiled to JSON")
