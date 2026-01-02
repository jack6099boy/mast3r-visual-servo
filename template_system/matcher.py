"""
Template Matcher 模組
===================

使用 MASt3R 進行模板匹配，複用 tools/polygon_target_transfer.py 的核心函數。
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from .manager import TemplateManager, Template

# 設置 path 以便 import tools 模組
import sys
import os
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 從 tools 模組導入核心函數
try:
    from tools import (
        load_mast3r_model,
        compute_matches,
        filter_matches_in_polygon,
        compute_transform,
        transform_point,
        compute_reprojection_error,
        MAST3R_AVAILABLE,
    )
except ImportError as e:
    print(f"Tools import error: {e}. Matching will use dummy scores.")
    MAST3R_AVAILABLE = False
    load_mast3r_model = None
    compute_matches = None
    filter_matches_in_polygon = None
    compute_transform = None
    transform_point = None
    compute_reprojection_error = None


class TemplateMatcher:
    """模板匹配器，使用 MASt3R 進行特徵匹配"""
    
    def __init__(self, manager: TemplateManager, 
                 model_name: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", 
                 device: str = None):
        """
        初始化模板匹配器
        
        Args:
            manager: TemplateManager 實例
            model_name: MASt3R 模型名稱 (HuggingFace)
            device: 計算設備 ('cuda', 'mps', 'cpu')，若為 None 則自動選擇
        """
        self.manager = manager
        self.model_name = model_name
        
        # 設備選擇邏輯
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        
        # 延遲載入模型
        self._model = None
    
    @property
    def model(self):
        """延遲載入 MASt3R 模型"""
        if self._model is None and MAST3R_AVAILABLE:
            self._model = load_mast3r_model(device=self.device, model_name=self.model_name)
        return self._model

    def match(self, rule: str, key: str, target_image: Path) -> Dict[str, Any]:
        """
        匹配 rule/key 下的所有 templates 與 target_image，返回最佳匹配與所有結果
        
        Args:
            rule: 規則名稱 (e.g., 'a_lab')
            key: 鍵名稱 (e.g., 'dog')
            target_image: 目標影像路徑
            
        Returns:
            dict: {
                'best': Dict (最佳匹配結果) or None,
                'all': List[Dict] (所有匹配結果)
            }
        """
        templates = self.manager.load_templates(rule, key)
        if not templates:
            return {'best': None, 'all': []}

        results = []
        for temp in templates:
            matches = self._get_matches(temp.img_path, target_image)
            if matches is None:
                continue
            
            # 假設第一個 polygon 是 'target' ROI
            if 'polygons' in temp.roi_data and temp.roi_data['polygons']:
                polygon = temp.roi_data['polygons'][0]['points']
                # 使用 tools 的 filter_matches_in_polygon
                filtered_im0, filtered_im1 = filter_matches_in_polygon(
                    matches[:, :2], matches[:, 2:4], polygon
                )
                num_in_roi = len(filtered_im0)
            else:
                num_in_roi = 0
            
            score = num_in_roi / max(1, len(matches))
            results.append({
                'template': temp.img_path.name,
                'score': float(score),
                'num_matches': len(matches),
                'num_in_roi': num_in_roi
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        best = results[0] if results else None
        return {'best': best, 'all': results}

    def _get_matches(self, template_img: Path, target_img: Path) -> Optional[np.ndarray]:
        """
        使用 MASt3R 取得 matches
        
        Args:
            template_img: 模板影像路徑
            target_img: 目標影像路徑
            
        Returns:
            np.ndarray: [N, 4] 格式 (x1, y1, x2, y2) 或 None
        """
        if not MAST3R_AVAILABLE or self.model is None:
            return None
        
        try:
            # 使用 tools.compute_matches 函數
            result = compute_matches(
                model=self.model,
                img1_path=str(template_img),
                img2_path=str(target_img),
                device=self.device,
                size=512,
                use_original_coords=True  # 返回原圖座標
            )
            
            matches_im0 = result['matches_im0']
            matches_im1 = result['matches_im1']
            
            # 組合為 [N, 4] 格式
            return np.column_stack([matches_im0, matches_im1])
            
        except Exception as e:
            print(f"Matching error: {e}")
            return None

    def _filter_in_polygon(self, pts: np.ndarray, polygon: List[List[float]]) -> np.ndarray:
        """
        過濾在多邊形內的點
        
        此方法保留以維持向後相容，內部使用 tools.filter_matches_in_polygon
        
        Args:
            pts: [N, 2] 點座標
            polygon: 多邊形頂點 [[x1, y1], ...]
            
        Returns:
            np.ndarray: 在多邊形內的點
        """
        if filter_matches_in_polygon is not None:
            filtered, _ = filter_matches_in_polygon(pts, pts, polygon)
            return filtered
        else:
            # Fallback
            polygon_np = np.array(polygon, dtype=np.float32)
            mask = [cv2.pointPolygonTest(polygon_np, (float(pt[0]), float(pt[1])), False) >= 0 for pt in pts]
            return pts[np.array(mask)]

    def match_and_transform(
        self,
        rule: str,
        key: str,
        target_image_path: str,
        keypoints: Dict[str, List[float]] = None,
        transform_method: str = 'homography'
    ) -> dict:
        """
        匹配 template 並轉換 keypoints 到目標圖像座標
        
        Args:
            rule: 規則名稱 (e.g., 'a_lab')
            key: 鍵名稱 (e.g., 'dog')
            target_image_path: 目標影像路徑
            keypoints: keypoint 字典 {name: [x, y], ...}，若 None 則從 ROI JSON 讀取
            transform_method: 變換方法 ('homography', 'affine', 'rigid', 'translation')
        
        Returns:
            dict: {
                'best_template': str,
                'match_score': float,
                'transform_matrix': np.ndarray,
                'transformed_keypoints': {name: (x, y), ...},
                'reprojection_error': (mean, std)
            }
        """
        # 1. 複用現有 match() 找最佳 template
        match_result = self.match(rule, key, Path(target_image_path))
        
        if match_result['best'] is None:
            return {
                'best_template': None,
                'match_score': 0.0,
                'transform_matrix': None,
                'transformed_keypoints': {},
                'reprojection_error': (float('inf'), float('inf'))
            }
        
        best_template_name = match_result['best']['template']
        
        # 找到對應的 Template 物件
        templates = self.manager.load_templates(rule, key)
        template = next((t for t in templates if t.img_path.name == best_template_name), None)
        
        if template is None:
            return {
                'best_template': best_template_name,
                'match_score': match_result['best']['score'],
                'transform_matrix': None,
                'transformed_keypoints': {},
                'reprojection_error': (float('inf'), float('inf'))
            }
        
        # 2. 用 tools.compute_matches() 取得匹配點
        matches = self._get_matches(template.img_path, Path(target_image_path))
        
        if matches is None or len(matches) < 4:
            return {
                'best_template': best_template_name,
                'match_score': match_result['best']['score'],
                'transform_matrix': None,
                'transformed_keypoints': {},
                'reprojection_error': (float('inf'), float('inf'))
            }
        
        matches_im0 = matches[:, :2]
        matches_im1 = matches[:, 2:4]
        
        # 3. 用 tools.filter_matches_in_polygon() 過濾 ROI 內匹配點
        if 'polygons' in template.roi_data and template.roi_data['polygons']:
            polygon = template.roi_data['polygons'][0]['points']
            filtered_im0, filtered_im1 = filter_matches_in_polygon(
                matches_im0, matches_im1, polygon
            )
        else:
            filtered_im0, filtered_im1 = matches_im0, matches_im1
        
        if len(filtered_im0) < 4:
            return {
                'best_template': best_template_name,
                'match_score': match_result['best']['score'],
                'transform_matrix': None,
                'transformed_keypoints': {},
                'reprojection_error': (float('inf'), float('inf'))
            }
        
        # 4. 用 tools.compute_transform() 估算變換矩陣
        transform_result = compute_transform(
            filtered_im0, filtered_im1, method=transform_method
        )
        transform_matrix = transform_result[0]  # (M, inliers, info)
        
        if transform_matrix is None:
            return {
                'best_template': best_template_name,
                'match_score': match_result['best']['score'],
                'transform_matrix': None,
                'transformed_keypoints': {},
                'reprojection_error': (float('inf'), float('inf'))
            }
        
        # 5. 若 keypoints 為 None，從 ROI JSON 讀取 keypoints
        if keypoints is None:
            keypoints = template.roi_data.get('keypoints', {})
        
        # 6. 用 tools.transform_point() 轉換每個 keypoint
        transformed_keypoints = {}
        for name, coords in keypoints.items():
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                transformed = transform_point(transform_matrix, coords, method=transform_method)
                if transformed is not None and not isinstance(transformed, dict):
                    transformed_keypoints[name] = tuple(transformed)
        
        # 7. 計算重投影誤差
        reprojection_error = compute_reprojection_error(
            transform_matrix, filtered_im0, filtered_im1, method=transform_method
        )
        
        return {
            'best_template': best_template_name,
            'match_score': match_result['best']['score'],
            'transform_matrix': transform_matrix,
            'transformed_keypoints': transformed_keypoints,
            'reprojection_error': reprojection_error
        }
