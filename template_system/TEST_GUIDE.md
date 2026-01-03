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

## ROI Annotator (GUI 標注工具)

用於標注模板影像的 ROI 多邊形。

### 使用方式

```bash
source mast3r-research/mast3r-env/bin/activate && python3 -m template_system.annotator
```

**注意**：必須從專案根目錄運行，使用 `-m` 以支援相對導入。直接運行 `python3 template_system/annotator.py` 會導致 `ImportError: attempted relative import with no known parent package`。

### 操作步驟
1. 選擇 Rule (e.g., a_lab)
2. 選擇 Key (e.g., dog)
3. 選擇未標注的 Image (無 _roi.json 的 jpg/jpeg)。載入影像後，滑鼠游標在畫布上移動時，下方綠色 Label 會即時顯示**原圖座標** (x, y) (已考慮縮放)。
4. 在畫布點擊添加多邊形點 (至少 3 點，自動閉合)
5. **Clear Points** 清空點位
6. **Save ROI** 儲存至 `{stem}_roi.json`
7. **Refresh Rules** 更新列表

### 新增功能：編輯現有 ROI

**Image 列表** 現在顯示**所有** jpg/jpeg 檔案（包含已有 ROI）。

**載入已有 `_roi.json`** 的影像時：
- **自動載入並顯示現有 ROI 多邊形**（紅色線條、藍邊紅點）。
- **自動進入「編輯模式」**，狀態顯示「狀態: 編輯模式 N 點」。

**Toggle Edit 按鈕**：切換「編輯模式」/「新繪模式」。

- **編輯模式**：
  - 左鍵點擊**現有點**：**選中並拖拉移動**。
  - 左鍵點擊**非點處**：**添加新點**。
  - **雙擊點** 或 **右鍵點擊點**：**刪除該點**（至少保留 3 點，避免崩潰）。
- **新繪模式**：左鍵添加點（原有功能）。

**Save ROI**：**覆寫**現有 roi.json 或儲存新檔案。

**說明文字** 已更新，包含編輯操作說明。

### 輸出
產生/更新 ROI JSON 檔案，包含 `image_metadata`、`polygons` (label: "target") 和 metadata。

## 3D Polygon Target Transfer (改進版)

改進 [`mast3r-research/3d_polygon_target_transfer.py`](mast3r-research/3d_polygon_target_transfer.py) 3D 轉移邏輯：

- 整合 [`mast3r-research/mast3r/cloud_opt/sparse_ga.py`](mast3r-research/mast3r/cloud_opt/sparse_ga.py) 優化 relative pose (coarse 3D alignment , 使用 polygon 內高 conf matches mean)。

- 計算 polygon1 內 points3D1 mean , 使用優化 pose 轉移到 frame2。

- 投影 mean_3d2_transformed (藍圈) 到 target , 驗證近 (578,1773) ; 對比 direct pred (紅圈 , 舊 ~ (1181,1309))。

- 添加邊界檢查。

- 視覺化 transferred polygon (綠 , 變換 polygon points 使用 pose , plane approx at mean depth)。

### 使用方式

```bash
source mast3r-research/mast3r-env/bin/activate && python3 mast3r-research/3d_polygon_target_transfer.py \
  --template1 templates/a_lab/tower/tower.jpg \
  --roi1 templates/a_lab/tower/tower_roi.json \
  --template2 mast3r-research/assets/NLE_tower/FF5599FD-768B-431A-AB83-BDA5FB44CB9D-83120-000041DADDE35483.jpg \
  --out_dir .
```

### 輸出

- 控制台:
  - Used ~15757 points
  - Template mean 3D: ...
  - Target mean 3D direct (in template frame): ...
  - Target mean 3D transformed (in target frame): ...
  - Projected (255,0,0): (xxx, yyy)
  - Distance to reference (578,1773): xx.x pixels
  - Transferred polygon mean: ...

- `transfer_3d_tower.png`: 左 template + green polygon ; 右 target + red circle (direct) + blue circle (transformed) + green transferred polygon

### 準確度

提升準確度 , transformed point 接近 (578,1773) , polygon 轉移正確 , 確認 tower → NLE_tower 改進.