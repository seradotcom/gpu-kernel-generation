from enum import Enum
from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# SCF & Triton Schema (Structured Outputs)
# ---------------------------------------------------------

class MLIRType(str, Enum):
    """
    Tipos MLIR permitidos. Enumerados explícitamente para que XGrammar
    pueda enforcarlos correctamente (los regex con {n,m} no funcionan bien).
    """
    # Escalares
    F32   = "f32"
    F16   = "f16"
    I32   = "i32"
    I1    = "i1"
    INDEX = "index"

    # Tensores 1D — tamaños comunes en kernels GPU
    T_32_F32   = "tensor<32xf32>"
    T_64_F32   = "tensor<64xf32>"
    T_128_F32  = "tensor<128xf32>"
    T_256_F32  = "tensor<256xf32>"
    T_512_F32  = "tensor<512xf32>"
    T_1024_F32 = "tensor<1024xf32>"
    T_32_F16   = "tensor<32xf16>"
    T_64_F16   = "tensor<64xf16>"
    T_128_F16  = "tensor<128xf16>"
    T_256_F16  = "tensor<256xf16>"
    T_512_F16  = "tensor<512xf16>"
    T_1024_F16 = "tensor<1024xf16>"

    # Tensores 2D
    T_64x64_F32   = "tensor<64x64xf32>"
    T_64x128_F32  = "tensor<64x128xf32>"
    T_128x64_F32  = "tensor<128x64xf32>"
    T_128x128_F32 = "tensor<128x128xf32>"
    T_64x64_F16   = "tensor<64x64xf16>"
    T_64x128_F16  = "tensor<64x128xf16>"
    T_128x64_F16  = "tensor<128x64xf16>"
    T_128x128_F16 = "tensor<128x128xf16>"

    # Punteros Triton
    PTR_F32 = "!tt.ptr<f32>"
    PTR_F16 = "!tt.ptr<f16>"

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
    TENSOR_EMPTY = "tensor.empty"
    TENSOR_EXTRACT = "tensor.extract"
    TENSOR_INSERT = "tensor.insert"

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
    
    # Triton GPU (ttg) Dialect
    TTG_LOCAL_ALLOC = "ttg.local_alloc"
    TTG_LOCAL_LOAD = "ttg.local_load"
    TTG_LOCAL_STORE = "ttg.local_store"
    TTG_CONVERT_LAYOUT = "ttg.convert_layout"

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
    region_combiner: Optional[str] = Field(None)

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
    reasoning: str = Field(..., description="Step 1: Write the mathematical pseudocode. Step 2: Map exact MLIR registers. Step 3: Explain the scoping for scf.for loops BEFORE generating the JSON.", max_length=800)
    code: MLIRFunctionBody = Field(..., description="The program compiled to JSON")
