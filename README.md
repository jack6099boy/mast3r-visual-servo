# MASt3R Visual Servo Tools

基於 MASt3R (Matching And Stereo 3D Reconstruction) 的視覺伺服工具集，用於影像特徵匹配與目標點轉換。

## 功能

1. **影像特徵匹配** - 使用 MASt3R 進行 dense feature matching
2. **多邊形 ROI 目標點轉換** - 在指定區域內匹配並轉換目標點
3. **多種幾何變換支援** - Affine, Homography, Fundamental Matrix

## 系統需求

- macOS (M1/M2) 或 Linux (CUDA)
- Python 3.11+
- 8GB+ RAM

## 快速開始

### 1. 安裝

```bash
# Clone 專案
git clone https://github.com/your-username/project3.git
cd project3

# 初始化 MASt3R submodule
git submodule update --init --recursive

# 創建虛擬環境
cd mast3r-research
python3 -m venv mast3r-env
source mast3r-env/bin/activate

# 安裝依賴
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
```

### 2. 使用工具

#### 基本特徵匹配測試
```bash
cd mast3r-research
source mast3r-env/bin/activate
cd ..
python tools/test_pairwise_mast3r.py
```

#### 多邊形目標點轉換
```bash
python tools/polygon_target_transfer.py \
    --template mast3r-research/assets/NLE_tower/image1.jpg \
    --target mast3r-research/assets/NLE_tower/image2.jpg \
    --polygon '[[100,50],[200,50],[200,400],[100,400]]' \
    --target_point '[150,200]' \
    --method affine \
    --output result.png
```

## 專案結構

```
project3/
├── README.md                      # 本文件
├── setup.sh                       # 安裝腳本
├── .gitignore                     # Git 忽略規則
├── .gitmodules                    # Git submodule 配置
│
├── tools/                         # 自製工具
│   ├── polygon_target_transfer.py # 多邊形目標點轉換工具
│   └── test_pairwise_mast3r.py   # 特徵匹配測試
│
├── docs/                          # 文檔
│   └── ARCHITECTURE.md           # 架構說明
│
├── examples/                      # 範例 (測試結果)
│
├── plans/                         # 規劃文檔
│   └── dual-visual-servo-architecture.md
│
└── mast3r-research/               # MASt3R 核心 (submodule)
    ├── requirements.txt          # 依賴
    ├── assets/                   # 測試影像
    ├── mast3r/                   # MASt3R 模型
    └── dust3r/                   # DUSt3R 基礎 (submodule)
```

## 變換方法

| 方法 | DOF | 說明 | 適用場景 |
|------|-----|------|----------|
| `partial` | 4 | 旋轉 + 均勻縮放 + 平移 | 小範圍調整 |
| `affine` | 6 | 完整仿射變換 | 一般用途 |
| `homography` | 8 | 透視變換 | 平面目標、大視角 |
| `fundamental` | 7 | 極線幾何 | 3D 位姿估計 |

## 架構流程

```
Template + Target → MASt3R 匹配 → 過濾多邊形內 → RANSAC → 計算變換 → 轉換點
                       651對          55對          25個       M矩陣    (x',y')
```

詳細架構請參考 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## API 使用

```python
# 從 tools 目錄導入
import sys
sys.path.insert(0, 'tools')
from polygon_target_transfer import PolygonTargetTransfer

# 初始化
transfer = PolygonTargetTransfer(device='mps')  # or 'cuda', 'cpu'

# 執行轉換
result = transfer.transfer_target_point(
    template_path='template.jpg',
    target_path='target.jpg',
    polygon=[[100,50],[200,50],[200,400],[100,400]],
    target_point=[150, 200],
    transform_method='affine',  # partial, affine, homography, fundamental
    ransac_thresh=5.0
)

# 結果
print(result['transformed_point'])  # (128.51, 209.02)
print(result['num_inliers'])        # 25
print(result['transform_matrix'])   # 2x3 矩陣
```

## 測試結果

使用 NLE_tower 測試影像：

| 方法 | 轉換點 | Inliers | 誤差 |
|------|--------|---------|------|
| partial | (100.52, 220.19) | 18/55 | 2.34 px |
| affine | (128.51, 209.02) | 25/55 | 1.83 px |
| homography | (130.08, 206.40) | 25/55 | 2.02 px |

## 相關資源

- [MASt3R 官方 Repo](https://github.com/naver/MASt3R)
- [DUSt3R 論文](https://arxiv.org/abs/2312.14132)
- [MASt3R 論文](https://arxiv.org/abs/2406.09756)

## License

本專案工具部分為 MIT License。  
MASt3R 核心為 CC BY-NC-SA 4.0 (非商業用途)。
