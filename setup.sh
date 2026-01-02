#!/bin/bash
# MASt3R Visual Servo Tools - Setup Script

set -e

echo "========================================"
echo "MASt3R Visual Servo Tools Setup"
echo "========================================"

# 檢查 Python 版本
echo "Checking Python version..."
python3 --version

# 初始化 submodule
echo ""
echo "Step 1: Initializing MASt3R submodule..."
if [ ! -d "mast3r-research/.git" ]; then
    git submodule update --init --recursive
else
    echo "  MASt3R already initialized"
fi

# 創建虛擬環境
echo ""
echo "Step 2: Creating virtual environment..."
cd mast3r-research

if [ ! -d "mast3r-env" ]; then
    python3 -m venv mast3r-env
    echo "  Created mast3r-env"
else
    echo "  mast3r-env already exists"
fi

# 啟動虛擬環境
echo ""
echo "Step 3: Installing dependencies..."
source mast3r-env/bin/activate

# 升級 pip
pip install --upgrade pip

# 安裝 PyTorch (自動檢測 MPS/CUDA)
echo "  Installing PyTorch..."
pip install torch torchvision torchaudio

# 安裝 MASt3R 依賴
echo "  Installing MASt3R requirements..."
pip install -r requirements.txt
pip install -r dust3r/requirements.txt

# 可選依賴 (視覺化等)
echo "  Installing optional requirements..."
pip install -r dust3r/requirements_optional.txt 2>/dev/null || true

cd ..

# 檢查安裝
echo ""
echo "Step 4: Verifying installation..."
cd mast3r-research
./mast3r-env/bin/python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
cd ..

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  cd mast3r-research && source mast3r-env/bin/activate"
echo ""
echo "To run the demo:"
echo "  python demo.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --device mps"
echo ""
echo "To use the polygon transfer tool:"
echo "  cd .. && python tools/polygon_target_transfer.py --help"
