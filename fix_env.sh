#!/bin/bash
#===============================================================================
# fix_env.sh - RTX 5060 Ti (sm_120) PyTorch Environment Setup
# Installs PyTorch nightly with CUDA 13.0+ support for Blackwell/sm_120 GPUs
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

echo "========================================"
echo "RTX 50 Series (sm_120) Environment Setup"
echo "========================================"

# Check if running in WSL
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "[INFO] WSL2 environment detected"
fi

# Check CUDA version from nvidia-smi
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "unknown")
echo "[INFO] Driver CUDA Version: ${CUDA_VERSION}"

# Activate virtual environment
if [ -d "${VENV_DIR}" ]; then
    echo "[INFO] Activating existing venv..."
    source "${VENV_DIR}/bin/activate"
else
    echo "[INFO] Creating new venv..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
fi

echo "[INFO] Python: $(which python)"
echo "[INFO] Pip: $(which pip)"

# Upgrade pip
echo ""
echo "[STEP 1/5] Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Uninstall existing torch packages to avoid conflicts
echo ""
echo "[STEP 2/5] Removing existing torch packages..."
pip uninstall -y torch torchvision torchaudio triton 2>/dev/null || true

# Install PyTorch nightly with CUDA 13.0 (cu130) support
# sm_120 support requires PyTorch nightly builds
echo ""
echo "[STEP 3/5] Installing PyTorch nightly (cu130) for sm_120 support..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Verify PyTorch installation
echo ""
echo "[STEP 4/5] Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Capability: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}')
    # Test SDPA
    print(f'SDPA Available: {hasattr(torch.nn.functional, \"scaled_dot_product_attention\")}')
"

# Install ComfyUI requirements
echo ""
echo "[STEP 5/5] Installing ComfyUI requirements..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# Install triton for torch.compile optimization
echo ""
echo "[BONUS] Installing triton for torch.compile..."
pip install --pre triton --index-url https://download.pytorch.org/whl/nightly/cu130 || \
    pip install triton || echo "[WARN] Triton installation failed, torch.compile may be limited"

# Install additional optimization packages
pip install accelerate xformers 2>/dev/null || echo "[INFO] xformers not available for this config"

echo ""
echo "========================================"
echo "Environment setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run: ./run_comfy.sh"
echo "  2. Or manually: source venv/bin/activate && python main.py --highvram --listen"
echo ""
