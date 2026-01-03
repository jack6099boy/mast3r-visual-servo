# 3D Matcher 實作 TODO 清單

## 概覽

將 3D 點轉換功能從 `mast3r-research/3d_polygon_target_transfer.py` 重構到 `template_system/matcher_3d.py`，
並修正座標系混用的問題。

---

## 修改清單

### ✅ Phase 1: 建立新的 3D Matcher 模組

#### 1.1 建立 `template_system/matcher_3d.py`

```python
# 核心類別
class TemplateMatcher3D:
    """使用 MASt3R sparse_global_alignment 進行 3D 點轉換"""
    
    def __init__(self, model_name: str, device: str)
    def match_and_transform_3d(self, template: Template, target_path: str, 
                                target_point: Tuple[float, float]) -> dict
```

**關鍵修改：**
- [ ] 移除 `inference()` 調用
- [ ] 統一使用 `sparse_global_alignment` 獲取 poses 和 pts3d
- [ ] 使用 `scene.get_sparse_pts3d()` 代替 `inference()` 的 pts3d
- [ ] 支援自定義標記點（不只是 ROI 質心）
- [ ] 返回相對姿態 T_rel

---

### ✅ Phase 2: 實作正確的 3D 流程

#### 2.1 `get_aligned_scene()` 方法

```python
def get_aligned_scene(self, template_path: str, target_path: str) -> SparseGA:
    """
    使用 sparse_global_alignment 獲取一致的場景數據
    
    Returns:
        scene: SparseGA 物件，包含：
            - get_im_poses() → 相機姿態 (cam2w)
            - get_sparse_pts3d() → 稀疏 3D 點（世界座標系）
            - get_focals() → 焦距
            - get_principal_points() → 主點
    """
```

**實作步驟：**
- [ ] 載入圖片 `load_images([template_path, target_path], size=518)`
- [ ] 建立 pairs_in 格式
- [ ] 調用 `sparse_global_alignment()`
- [ ] 返回 scene 物件

---

#### 2.2 `transform_point_3d()` 方法

```python
def transform_point_3d(self, scene, template_pixel: Tuple[float, float], 
                       orig_shape: Tuple[int, int], res_shape: Tuple[int, int]) -> dict:
    """
    將 template 上的像素點轉換到 target 圖片座標
    
    Args:
        scene: sparse_ga 返回的 scene 物件
        template_pixel: template 上的像素座標 (x, y)
        orig_shape: 原始圖片形狀 (H, W)
        res_shape: 縮放後形狀 (H, W)
    
    Returns:
        dict: {
            'target_pixel': (x, y),      # 在 target 圖片上的像素位置
            'point_3d_world': (x, y, z), # 3D 世界座標
            'point_3d_cam2': (x, y, z),  # 在 Camera2 座標系的位置
            'T_rel': np.ndarray,         # 相對姿態矩陣 4x4
            'focals': (f1, f2),          # 估計的焦距
        }
    """
```

**實作步驟：**
- [ ] 從 scene 獲取所有數據（poses, pts3d, focals）
- [ ] 縮放像素座標
- [ ] 從 pts3d 找出標記點的 3D 位置（使用插值或最近鄰）
- [ ] 計算相對姿態 `T_rel = inv(cam2w_2) @ cam2w_1`
- [ ] 將 3D 點轉換到 Camera2 座標系
- [ ] 投影到 target 圖片座標

---

#### 2.3 `get_3d_at_pixel()` 輔助方法

```python
def get_3d_at_pixel(self, pts3d: np.ndarray, pixel: Tuple[float, float], 
                    method: str = 'nearest') -> np.ndarray:
    """
    從稀疏 pts3d 獲取指定像素位置的 3D 座標
    
    Args:
        pts3d: sparse_ga 的 pts3d 輸出（稀疏點）
        pixel: 像素座標 (x, y)
        method: 'nearest' 或 'interpolate'
    
    Returns:
        np.ndarray: 3D 座標 [x, y, z]
    """
```

**實作步驟：**
- [ ] 處理稀疏 pts3d 的索引
- [ ] 實作最近鄰方法
- [ ] (可選) 實作插值方法

---

### ✅ Phase 3: 整合到 template_system

#### 3.1 更新 `template_system/__init__.py`

```python
from .manager import TemplateManager, Template
from .matcher import TemplateMatcher
from .matcher_3d import TemplateMatcher3D  # 新增

__all__ = ['TemplateManager', 'Template', 'TemplateMatcher', 'TemplateMatcher3D']
```

---

#### 3.2 建立 `template_system/visual_servo.py`（可選）

```python
class VisualServoController:
    """整合 2D/3D 匹配的視覺伺服控制器"""
    
    def __init__(self, matcher_2d: TemplateMatcher, matcher_3d: TemplateMatcher3D)
    def compute_correction(self, template: Template, target_path: str, 
                           target_point: Tuple, desired_pixel: Tuple) -> dict
```

---

### ✅ Phase 4: 測試和驗證

#### 4.1 建立測試腳本 `template_system/test_3d_matcher.py`

- [ ] 測試 sparse_ga 座標系一致性
- [ ] 測試點轉換準確度
- [ ] 測試相對姿態計算
- [ ] 與舊版 3d_polygon_target_transfer.py 比較結果

---

## 參考：正確的座標系關係

```
sparse_global_alignment 輸出：
├── poses[0] (cam2w_1) → Camera 1 到 World 的變換（通常是單位矩陣）
├── poses[1] (cam2w_2) → Camera 2 到 World 的變換
├── pts3d[0] → Template 的 3D 點（世界座標系）
├── pts3d[1] → Target 的 3D 點（世界座標系）
└── focals[0], focals[1] → 估計的焦距

座標轉換：
├── 世界座標 → Camera2 座標：P_cam2 = inv(cam2w_2) @ P_world
├── Camera2 座標 → 像素：p = K2 @ P_cam2[:3] / P_cam2[2]
└── 相對姿態：T_rel = inv(cam2w_2) @ cam2w_1 (Camera1 到 Camera2)
```

---

## 不要做的事

1. ❌ 不要混用 `inference()` 和 `sparse_global_alignment()` 的數據
2. ❌ 不要假設 `inference()` 的 pts3d 和 `sparse_ga` 的 poses 在同一座標系
3. ❌ 不要硬編碼假設 pose_cam1 是單位矩陣（雖然通常是）
4. ❌ 不要修改 mast3r-research 子模組的代碼

---

## 估計完成時間

- Phase 1: 建立基本結構
- Phase 2: 實作核心邏輯
- Phase 3: 整合測試
- Phase 4: 驗證修正
