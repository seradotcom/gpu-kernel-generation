# MLIR Core Python Bindings: Comprehensive Multi-Platform Setup Guide

This document provides definitive, production-grade instructions to install and configure LLVM/MLIR with native Python bindings across multiple platforms and environment managers. Visit the [official documentation](https://mlir.llvm.org/) for additional reference.

---

## 1. Prerequisites & System Architecture

Building or linking LLVM/MLIR requires an isolated runtime ecosystem. Due to strict system-package management overheads in modern operating systems (e.g., PEP 668 in Ubuntu 24.04+), a virtual environment or containerized setup is **mandatory**.

### Target Environment Specification

- **Python Runtime:** Python 3.10 (Highly recommended for downstream compatibility with tools like Triton, `xgrammar`, and Torch-MLIR)
- **Build System:** CMake (>= 3.20) & Ninja
- **Compiler Infrastructure:** Clang & LLD (Linker) are strongly recommended to maximize compilation speed and minimize memory thrashing

---

## 2. Installation Options

Choose the method that best fits your use case:

- **Option A: Build from Source** — For custom dialect generation, local research extensions, or explicit hardware target matching (e.g., NVIDIA PTX optimization).
- **Option B: Docker Container** — For sandboxed execution or consistent deployment targets across Linux and macOS hosts.

---

## 2.1 Option A: Monolithic Compilation From Source

### Step 1: OS Setup & Toolchains

#### 🐧 Linux (Ubuntu 24.04 LTS / Debian-based)

Modern Ubuntu ships with Python 3.12+. We register the `deadsnakes` PPA to securely provision Python 3.10 without disrupting OS stability.

```bash
# Update system package definitions
sudo apt update && sudo apt upgrade -y

# Install core build toolchains, Clang, LLD, and ccache
sudo apt install -y cmake ninja-build clang lld ccache software-properties-common

# Register Deadsnakes PPA for Python 3.10
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Install Python 3.10 with headers and venv support
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

#### 🍎 macOS (Intel & Apple Silicon)

macOS users leverage `Homebrew` to manage compiler tools and Python runtimes securely.

```bash
# Install Homebrew if not present
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install package dependencies
brew install cmake ninja llvm ccache python@3.10

# Ensure the system knows where LLVM tools are located
# For Apple Silicon:
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
# For Intel:
# export PATH="/usr/local/opt/llvm/bin:$PATH"
```

#### 🪟 Windows (Windows 10/11 via x64 Native Tools Command Prompt)

1. **Install Visual Studio Build Tools:**
   - Download and install Visual Studio.
   - Under workloads, check **Desktop development with C++**.
   - Under Individual components, ensure **C++ CMake tools for Windows** is checked.
2. **Install Python 3.10:** Download from [python.org](https://www.python.org/downloads/) and tick **"Add python.exe to PATH"** during setup.
3. **Install Ninja:** Download from [GitHub](https://github.com/ninja-build/ninja/releases) and add its binary path to your System Environment variables.

---

### Step 2: Python Virtual Environment Configuration

```bash
# Navigate to your workspace directory
cd ~/your-workspace/llm-mlir-compiler

# Provision the virtual environment targeting Python 3.10
python3.10 -m venv .venv
# On Windows CMD/PowerShell: python -m venv .venv

# Activate the environment
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate.bat       # Windows CMD
# .venv\Scripts\Activate.ps1       # Windows PowerShell

# Bootstrap pip
python -m pip install --upgrade pip
```

---

### Step 3: Source Tree & Binding Dependencies

```bash
# Clone the unified LLVM Project Monorepo
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Install tablegen wrapper requirements in your activated Python space
pip install -r mlir/python/requirements.txt
```

---

### Step 4: Build Orchestration (CMake & Ninja)

Pay close attention to the `-DLLVM_TARGETS_TO_BUILD` flag — it varies by platform.

#### For Linux (with NVIDIA GPU support):

```bash
mkdir build && cd build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_ENABLE_LLD=ON \
   -DLLVM_CCACHE_BUILD=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=$(which python)
```

#### For macOS (Apple Silicon / Intel):

> ⚠️ **Important:** Omit `NVPTX` on macOS to prevent compilation errors.

```bash
mkdir build && cd build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_ENABLE_LLD=ON \
   -DLLVM_CCACHE_BUILD=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=$(which python)
```

> ⚠️ **Windows Note:** Replace `$(which python)` with the absolute path to your `.venv` executable (e.g., `-DPython3_EXECUTABLE="C:\your-workspace\llm-mlir-compiler\.venv\Scripts\python.exe"`). Omit the Clang/LLD flags unless explicit LLVM-MinGW paths are declared.

---

### Step 5: Compilation and Test Suite

Kick off the compilation. This process can take significant time depending on your hardware.

```bash
cmake --build . --target check-mlir-python
```

Expected output:

```text
Total Discovered Tests: 107
Unsupported: 7
Passed: 100
```

> The ~7 unsupported tests reflect missing optional system components such as architecture-specific target SDKs — this is normal and safe.

---

## 2.2 Option B: Isolated Containerization via Docker

For sandboxed execution or consistent deployment targets across Linux and macOS hosts, leverage pre-compiled Docker images optimized for MLIR development.

```bash
# Pull CUDA-optimized stack (Ubuntu 24.04 base)
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir20-cuda-ubuntu24.04

# Pull CPU-only stack (Ubuntu 24.04 base)
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir20-ubuntu24.04

# Run interactively
docker run -it ghcr.io/sdiehl/docker-mlir-cuda:mlir20-ubuntu24.04 bash
```

---

## 3. Runtime Linkage & System Validation

If you built from source, register the compiled packages with your environment paths before executing any scripts.

### Register Environmental Pathing

Run this in your shell, or add it permanently to `~/.bashrc` / `~/.zshrc`:

```bash
export PYTHONPATH=$(pwd)/tools/mlir/python_packages/mlir_core:$PYTHONPATH
```

### Smoke Test Script (`sanity_check.py`)

```python
import sys
from mlir.ir import Context, Module, InsertionPoint, Location, FunctionType, IntegerType
from mlir.dialects import func

def verify_toolchain():
    print("[+] Checking MLIR Core Binding Infrastructure...")
    try:
        with Context() as ctx, Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                i32 = IntegerType.get_signless(32)
                func_type = FunctionType.get(inputs=[i32, i32], results=[i32])
                add_op = func.FuncOp(name="llvm_pipeline_test", type=func_type)
                add_op.add_entry_block()

            print("[+] System Diagnostics Normal. Output AST:\n")
            print(module)
    except Exception as error:
        print(f"[-] Execution Failure: {str(error)}")

if __name__ == "__main__":
    verify_toolchain()
```

Run the validation script:

```bash
python sanity_check.py
```

### Expected Output

```mlir
[+] Checking MLIR Core Binding Infrastructure...
[+] System Diagnostics Normal. Output AST:

"builtin.module"() ({
  "func.func"() <{function_type = (i32, i32) -> i32, sym_name = "llvm_pipeline_test"}> ({
  ^bb0(%arg0: i32, %arg1: i32):
  }) : () -> ()
}) : () -> ()
```