"""
Tools 模組
=========

提供影像匹配和多邊形 ROI 轉換的核心功能。

主要功能：
    - load_mast3r_model: 載入 MASt3R 模型
    - compute_matches: 計算兩張圖片的特徵匹配點
    - filter_matches_in_polygon: 過濾在多邊形內的匹配點
    - visualize_transfer: 視覺化轉移結果
    - compute_transform: 計算幾何變換矩陣
    - transform_point: 使用變換矩陣轉換點

使用範例：
    from tools import load_mast3r_model, compute_matches, filter_matches_in_polygon
    
    model = load_mast3r_model(device='mps')
    result = compute_matches(model, 'template.jpg', 'target.jpg', device='mps')
    filtered = filter_matches_in_polygon(result['matches_im0'], result['matches_im1'], polygon)
"""

from .polygon_target_transfer import (
    # 核心匹配函數
    load_mast3r_model,
    compute_matches,
    filter_matches_in_polygon,
    point_in_polygon,
    
    # 視覺化
    visualize_transfer,
    
    # 幾何變換
    compute_transform,
    transform_point,
    compute_reprojection_error,
    
    # 類別
    PolygonTargetTransfer,
    
    # 常數
    MAST3R_AVAILABLE,
)

__all__ = [
    # 核心匹配函數
    'load_mast3r_model',
    'compute_matches',
    'filter_matches_in_polygon',
    'point_in_polygon',
    
    # 視覺化
    'visualize_transfer',
    
    # 幾何變換
    'compute_transform',
    'transform_point',
    'compute_reprojection_error',
    
    # 類別
    'PolygonTargetTransfer',
    
    # 常數
    'MAST3R_AVAILABLE',
]
