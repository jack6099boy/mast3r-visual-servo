# Template 系統標注指南

## 系統架構

Template 系統採用**兩層目錄結構**管理樣板影像：

```
templates/
├── {rule_name}/           # 第一層：規則名稱 (例如：a_lab)
│   ├── {key_name}/        # 第二層：識別 key (例如：dog)
│   │   ├── front.jpeg     # 參考影像
│   │   ├── front_roi.json # ROI 標注檔
│   │   ├── side.jpg       # 另一角度影像
│   │   └── side_roi.json  # 對應 ROI 標注檔
│   └── {another_key}/
└── {another_rule}/
```

### 核心概念

| 名稱 | 說明 |
|------|------|
| **Rule** | 規則分類，例如實驗室名稱、場景類型等 |
| **Key** | 識別項目，例如動物編號、物件名稱等 |
| **ROI** | Region of Interest，標記目標區域的多邊形 |

### 檔案命名規則

- 影像檔：`{name}.jpg` 或 `{name}.jpeg`
- 標注檔：`{name}_roi.json`（與影像同名加 `_roi` 後綴）

---

## 標注工具啟動

### 環境需求

- Python 3.10+
- tkinter (macOS/Linux 內建)
- Pillow (PIL)

### 啟動指令

```bash
# 在專案根目錄執行
python3 -m template_system.annotator
```

若啟動成功，會出現 GUI 視窗「**ROI Annotator - Template System**」。

---

## GUI 操作步驟

### Step 1: 選擇 Rule

1. 在 **Rule** 下拉選單中選擇要標注的規則分類
2. 例如選擇 `a_lab`

### Step 2: 選擇 Key

1. Rule 選擇後，**Key** 下拉選單會自動更新
2. 選擇目標 key，例如 `dog`

### Step 3: 選擇未標注影像

1. Key 選擇後，**Image** 下拉選單會顯示所有**尚未標注**的影像
2. 只有沒有對應 `*_roi.json` 的影像會出現在列表中
3. 點選要標注的影像，例如 `front.jpeg` 或 `side.jpg`

### Step 4: 標記多邊形頂點

1. 影像載入後，在畫布上**依序點擊**目標區域的頂點
2. 每次點擊會在該位置標記紅點
3. 點與點之間會自動連線（紅色線條）
4. 建議按照順時針或逆時針順序標記
5. **至少需要 3 個點**才能形成有效多邊形

### Step 5: 儲存 ROI

1. 標記完成後，點擊 **Save ROI** 按鈕
2. 系統會自動：
   - 閉合多邊形（首尾相連）
   - 產生 `{影像名}_roi.json` 檔案
   - 顯示成功訊息
3. 該影像會從「未標注影像」列表中消失

### 其他操作

| 按鈕 | 功能 |
|------|------|
| **Clear Points** | 清除所有已標記的點，重新開始 |
| **Refresh Rules** | 重新載入 rules 列表（新增資料夾後使用） |
| **Quit** | 關閉標注工具 |

---

## 驗證標注結果

### 方法 1: 檢查檔案產生

```bash
# 檢查是否產生 ROI 檔案
ls templates/a_lab/dog/

# 預期輸出應包含 *_roi.json 檔案
# front.jpeg  front_roi.json  side.jpg  side_roi.json
```

### 方法 2: 檢視 JSON 內容

```bash
cat templates/a_lab/dog/front_roi.json
```

預期輸出格式：

```json
{
  "version": "1.0",
  "image_metadata": {
    "width": 640,
    "height": 480,
    "filename": "front.jpeg"
  },
  "polygons": [
    {
      "label": "target",
      "points": [
        [100.0, 150.0],
        [300.0, 150.0],
        [300.0, 350.0],
        [100.0, 350.0],
        [100.0, 150.0]
      ],
      "metadata": {
        "annotated_by": "user",
        "timestamp": "2024-01-01T00:00:00Z"
      }
    }
  ]
}
```

### 方法 3: Python 驗證

```python
from template_system import TemplateManager

manager = TemplateManager()
templates = manager.load_templates('a_lab', 'dog')
print(f"已標注的 templates 數量: {len(templates)}")
for t in templates:
    print(f"  - {t.img_path.name}: {len(t.roi_data['polygons'])} 個多邊形")
```

---

## 測試檢查清單

完成標注測試後，確認以下項目：

- [ ] GUI 正常啟動無錯誤
- [ ] Rule 下拉選單顯示 `a_lab`
- [ ] Key 下拉選單顯示 `dog`
- [ ] Image 選單顯示未標注的影像
- [ ] 點擊影像可標記點
- [ ] Save ROI 後產生對應 JSON 檔
- [ ] 已標注的影像從列表中消失

---

## 疑難排解

### tkinter 無法啟動

若出現 `No module named '_tkinter'` 或 `no display name` 錯誤：

```bash
# macOS 重新安裝 python-tk
brew install python-tk@3.11

# 或使用 pyenv
pyenv install 3.11 --with-tkinter
```

### 影像不在列表中

1. 確認副檔名為 `.jpg` 或 `.jpeg`
2. 確認影像放在正確的 `templates/{rule}/{key}/` 路徑
3. 點擊 **Refresh Rules** 重新載入

### 中文顯示問題

macOS 上若中文顯示異常，可設定環境變數：

```bash
export LANG=zh_TW.UTF-8
python3 -m template_system.annotator
```

---

## 目前測試資料

| 路徑 | 狀態 |
|------|------|
| `templates/a_lab/dog/front.jpeg` | 未標注 |
| `templates/a_lab/dog/side.jpg` | 未標注 |

可直接用於測試標注流程。
