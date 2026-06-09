from enum import Enum
from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# SCF & Triton Schema (Structured Outputs)
# ---------------------------------------------------------

# Dynamic MLIR Types Generation
base_types = ["f32", "f16", "bf16", "i32", "i64", "i16", "i8", "i1", "index"]
ptr_types = [f"!tt.ptr<{t}>" for t in base_types if t != "index"]
all_scalar_types = base_types + ptr_types

sizes_1d = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
tensors_1d = [f"tensor<{s}x{t}>" for s in sizes_1d for t in all_scalar_types]

sizes_2d = [16, 32, 64, 128, 256]
tensors_2d = [f"tensor<{s1}x{s2}x{t}>" for s1 in sizes_2d for s2 in sizes_2d for t in all_scalar_types]

asym_2d = [(1, 128), (1, 256), (1, 1024), (32, 64), (64, 32), (64, 128), (128, 64)]
tensors_2d_asym = [f"tensor<{s1}x{s2}x{t}>" for (s1, s2) in asym_2d for t in all_scalar_types]

ALL_MLIR_TYPES = all_scalar_types + tensors_1d + tensors_2d + tensors_2d_asym
_enum_dict = {t.replace('<', '_').replace('>', '_').replace('!', '').replace('.', '_').replace('x', '_'): t for t in ALL_MLIR_TYPES}
MLIRType = Enum('MLIRType', _enum_dict, type=str)

class MlirOpcode(str, Enum):
    # Arith Dialect
    ARITH_ADDF = "arith.addf"
    ARITH_SUBF = "arith.subf"
    ARITH_MULF = "arith.mulf"
    ARITH_DIVF = "arith.divf"
    ARITH_MAXIMUMF = "arith.maximumf"
    ARITH_MINIMUMF = "arith.minimumf"
    ARITH_MAXF = "arith.maxf"
    ARITH_MINF = "arith.minf"
    ARITH_MAXSI = "arith.maxsi"
    ARITH_MAXUI = "arith.maxui"
    ARITH_MINSI = "arith.minsi"
    ARITH_MINUI = "arith.minui"
    ARITH_CMPF = "arith.cmpf"
    ARITH_CMPI = "arith.cmpi"
    ARITH_ADDI = "arith.addi"
    ARITH_SUBI = "arith.subi"
    ARITH_MULI = "arith.muli"
    ARITH_ANDI = "arith.andi"
    ARITH_ORI = "arith.ori"
    ARITH_XORI = "arith.xori"
    ARITH_CONSTANT = "arith.constant"
    ARITH_EXTF = "arith.extf"
    ARITH_TRUNCF = "arith.truncf"
    ARITH_SITOFP = "arith.sitofp"
    ARITH_FPTOSI = "arith.fptosi"
    ARITH_EXTSI = "arith.extsi"
    ARITH_EXTUI = "arith.extui"
    ARITH_TRUNCI = "arith.trunci"
    MATH_RSQRT = "math.rsqrt"
    MATH_ERF = "math.erf"
    TT_MAKE_BLOCK_PTR = "tt.make_block_ptr"

    ARITH_SELECT = "arith.select"
    
    # Math Dialect
    MATH_EXP = "math.exp"
    MATH_LOG = "math.log"
    MATH_SQRT = "math.sqrt"
    MATH_COS = "math.cos"
    MATH_SIN = "math.sin"
    MATH_ABS = "math.absf"
    MATH_FLOOR = "math.floor"
    MATH_CEIL = "math.ceil"

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
    opcode: Literal["math.exp", "math.log", "math.sqrt", "math.rsqrt", "math.erf", "math.cos", "math.sin", "math.absf", "math.floor", "math.ceil", "arith.extf", "arith.truncf", "arith.sitofp", "arith.fptosi", "arith.extsi", "arith.extui", "arith.trunci", "tt.ptr_to_int", "tt.int_to_ptr"] = Field(..., description="Ops with exactly 1 operand")
    operands: List[Union[str, float, int]] = Field(..., min_length=1, max_length=1, description="Input registers or literal numbers")
    result: str = Field(..., pattern=r"^(?:%[a-zA-Z0-9_]{1,30}|none)$", description="Output register")
    out_type: Optional[MLIRType] = Field(None, description="Specify ONLY for explicit casting or when type cannot be inferred.")
    attributes: Optional[Dict[str, Union[int, float, str, bool, List[int], List[float]]]] = Field(None, description="Attributes like predicate=2 for cmpi, or value=256 for constant")

class BinaryOperation(BaseModel):
    opcode: Literal["arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.maximumf", "arith.minimumf", "arith.maxf", "arith.minf", "arith.maxsi", "arith.maxui", "arith.minsi", "arith.minui", "arith.cmpf", "arith.cmpi", "arith.addi", "arith.subi", "arith.muli", "arith.divsi", "arith.divui", "arith.remsi", "arith.remui", "arith.shli", "arith.shrsi", "arith.shrui", "arith.andi", "arith.ori", "arith.xori", "tt.addptr"] = Field(..., description="Ops with exactly 2 operands")
    operands: List[Union[str, float, int]] = Field(..., min_length=2, max_length=2, description="Input registers or literal numbers")
    result: str = Field(..., pattern=r"^(?:%[a-zA-Z0-9_]{1,30}|none)$", description="Output register")
    out_type: Optional[MLIRType] = Field(None, description="Specify ONLY for explicit casting or when type cannot be inferred.")
    attributes: Optional[Dict[str, Union[int, float, str, bool, List[int], List[float]]]] = Field(None, description="Attributes like predicate=2 for cmpi, or value=256 for constant")

class GenericOperation(BaseModel):
    opcode: MlirOpcode = Field(..., description="Allowed MLIR/Triton operation")
    operands: List[Union[str, float, int]] = Field(default_factory=list, description="Input registers or literal numbers")
    result: str = Field(..., pattern=r"^(?:%[a-zA-Z0-9_]{1,30}|none)$", description="Output register")
    out_type: Optional[MLIRType] = Field(None, description="Specify ONLY for explicit casting or when type cannot be inferred.")
    attributes: Optional[Dict[str, Union[int, float, str, bool, List[int], List[float]]]] = Field(None, description="Attributes like predicate=2 for cmpi, or value=256 for constant")
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
    reasoning: str = Field(..., description="Use this field to think step-by-step (Chain of Thought). Detail your register allocation, loop invariants, and type coercions. A thorough breakdown prevents logical errors in the JSON.")
    code: MLIRFunctionBody = Field(..., description="The program compiled to JSON")
