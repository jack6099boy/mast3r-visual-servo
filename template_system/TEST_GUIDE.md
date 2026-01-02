# Template System 測試指南

## 概述

Template System 提供模板匹配和座標轉換功能，主要用於將預定義模板中的關鍵點 (keypoints) 轉換到目標影像的對應位置。本測試指南說明如何運行和驗證系統功能。

## 系統架構

- **TemplateManager**: 管理模板資料夾結構 (`templates/rule/key/`)
- **TemplateMatcher**: 使用 MASt3R 進行特徵匹配和座標轉換
- **測試腳本**: `test_match.py` 和 `test_transform.py`

## 環境設置

### 依賴項

- Python 3.7+
- PyTorch (支援 CUDA/MPS/CPU)
- OpenCV
- NumPy
- Matplotlib (用於視覺化)
- MASt3R 模型 (從 HuggingFace 下載)

### 安裝步驟

1. 確保在專案根目錄
2. 安裝依賴項：
   ```bash
   pip install torch torchvision opencv-python numpy matplotlib
   ```

3. 確保 `tools/` 模組可用（包含 MASt3R 相關函數）

## 測試資料結構

模板檔案組織在 `templates/` 目錄下：

```
templates/
├── rule_name/           # 規則名稱 (e.g., a_lab)
│   └── key_name/        # 鍵名稱 (e.g., dog)
│       ├── template.jpg # 模板影像
│       └── template_roi.json # ROI 和 keypoints 定義
```

### ROI JSON 格式

```json
{
  "polygons": [
    {
      "name": "target_roi",
      "points": [[x1,y1], [x2,y2], ...]
    }
  ],
  "keypoints": {
    "point_name": [x, y],
    ...
  }
}
```

## 測試腳本

### 1. 模板匹配測試 (`test_match.py`)

測試模板匹配功能，驗證系統能正確識別和匹配模板。

#### 使用方式

```bash
python3 -m template_system.test_match --rule RULE --key KEY --target TARGET_IMAGE [--output OUTPUT_FILE]
```

#### 參數說明

- `--rule`: 規則名稱 (e.g., `a_lab`)
- `--key`: 鍵名稱 (e.g., `dog`)
- `--target`: 目標影像路徑
- `--output`: 輸出視覺化檔案路徑 (預設: `match_result.png`)

#### 範例

```bash
# 使用 dog 模板匹配測試影像
python3 -m template_system.test_match --rule a_lab --key dog --target test_dog.jpeg --output dog_match.png
```

#### 輸出說明

- **控制台輸出**:
  - 使用的匹配器 (MASt3R 或 ORB fallback)
  - 最佳模板名稱和匹配分數
  - 總匹配點數和 ROI 內匹配點數
  - 前 5 個匹配結果

- **視覺化輸出**:
  - 目標影像上的匹配點 (藍色圓點)
  - ROI 多邊形轉移 (綠色多邊形)
  - 匹配資訊文字疊加

### 2. 座標轉換測試 (`test_transform.py`)

測試 `match_and_transform` 功能，將模板中的 keypoints 轉換到目標影像座標。

#### 使用方式

目前為硬編碼測試，可直接運行：

```bash
python3 template_system/test_transform.py
```

#### 測試內容

- 使用 `a_lab/room` 模板
- 測試 keypoints: `{'center': [376, 1065], 'corner': [100, 100]}`
- 轉換方法: `homography`

#### 輸出說明

- **控制台輸出**:
  - 最佳模板名稱
  - 匹配分數
  - 轉換後座標
  - 重投影誤差

- **視覺化輸出** (`transform_test_result.png`):
  - 左圖: 模板影像 + 原始 keypoints (綠色)
  - 右圖: 目標影像 + 轉換後 keypoints (紅色)

## 匹配器說明

### MASt3R 匹配器 (主要)

- 使用深度學習模型進行特徵匹配
- 支援 GPU 加速 (CUDA/MPS)
- 需要網路下載模型 (首次運行)

### ORB 匹配器 (備用)

- 當 MASt3R 不可用時自動切換
- 使用傳統特徵點匹配
- 不需要額外依賴

## 故障排除

### 常見問題

1. **ImportError: No module named 'tools'**
   - 確保從專案根目錄運行
   - 檢查 `tools/` 目錄是否存在

2. **MASt3R model loading failed**
   - 檢查網路連線 (首次下載模型)
   - 確認 PyTorch 和 CUDA 版本相容

3. **No matches found**
   - 檢查目標影像是否存在
   - 確認模板和目標影像相似度足夠
   - 嘗試不同的 rule/key 組合

4. **Template not found**
   - 確認 `templates/RULE/KEY/` 目錄存在
   - 檢查模板檔案命名 (`.jpg`/`.jpeg` + `_roi.json`)

### 除錯提示

- 運行時會顯示使用的匹配器類型
- 檢查控制台輸出中的匹配統計資訊
- 視覺化輸出有助於理解匹配品質

## 效能考量

- MASt3R 匹配較慢但準確
- ORB 匹配快速但準確度較低
- GPU 加速可大幅提升 MASt3R 效能
- 影像解析度影響處理時間 (建議 512x512 以下)

## 延伸測試

- 測試不同照明條件下的匹配
- 測試不同角度/尺度的影像
- 驗證 keypoints 轉換準確度
- 測試多個模板的排名邏輯