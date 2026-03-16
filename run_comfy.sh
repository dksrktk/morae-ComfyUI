#!/bin/bash
#===============================================================================
# run_comfy.sh - Optimized ComfyUI Launch Script for RTX 5060 Ti
# Uses SDPA (Scaled Dot Product Attention) instead of flash-attn
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

#===============================================================================
# Environment Variables for Optimization
#===============================================================================

# Force PyTorch to use SDPA (built-in scaled dot product attention)
# This avoids flash-attn installation issues
export TORCH_SDPA_ENABLE=1

# Enable TF32 for better performance on Ampere+ GPUs
export TORCH_ALLOW_TF32=1
export CUDA_ALLOW_TF32=1

# torch.compile optimizations
export TORCH_COMPILE_BACKEND=inductor
export TORCHINDUCTOR_CACHE_DIR="${SCRIPT_DIR}/.torch_compile_cache"

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable triton cache warnings
export TRITON_CACHE_DIR="${SCRIPT_DIR}/.triton_cache"

# For WSL2: Use host's CUDA libraries
if grep -qi microsoft /proc/version 2>/dev/null; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
fi

#===============================================================================
# ComfyUI Launch Options
#===============================================================================

# Default arguments
ARGS=(
    "--highvram"           # Use high VRAM mode (16GB is plenty)
    "--listen"             # Listen on all interfaces (0.0.0.0)
    "--preview-method"     # Enable previews
    "auto"
)

# Optional: Enable torch.compile for extra performance
# Uncomment if you want torch.compile (may increase first-run time)
# ARGS+=("--use-pytorch-cross-attention")

# Optional: Specific port (default is 8188)
# ARGS+=("--port" "8188")

# Optional: Disable CUDA malloc for debugging
# ARGS+=("--disable-cuda-malloc")

# Parse command line arguments and append
ARGS+=("$@")

#===============================================================================
# Pre-flight Check
#===============================================================================

echo "========================================"
echo "ComfyUI Launcher - RTX 5060 Ti Optimized"
echo "========================================"

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'VRAM: {mem:.1f} GB')
print(f'SDPA: Available')
"

echo "========================================"
echo "Starting ComfyUI with args: ${ARGS[*]}"
echo "========================================"
echo ""

#===============================================================================
# Launch ComfyUI
#===============================================================================

cd "${SCRIPT_DIR}"
exec python main.py "${ARGS[@]}"
