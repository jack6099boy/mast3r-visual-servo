#!/usr/bin/env python
"""測試 3D Matcher - Tower 圖片"""

import sys
import json
from pathlib import Path

# 確保可以 import template_system
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from template_system.matcher_3d import TemplateMatcher3D

def main():
    print("初始化 3D Matcher...")
    matcher = TemplateMatcher3D()
    
    template_path = str(PROJECT_ROOT / "templates/a_lab/tower/tower.jpg")
    target_path = str(PROJECT_ROOT / "mast3r-research/assets/NLE_tower/FF5599FD-768B-431A-AB83-BDA5FB44CB9D-83120-000041DADDE35483.jpg")
    roi_path = PROJECT_ROOT / "templates/a_lab/tower/tower_roi.json"
    
    # 載入 ROI 資料
    polygon = None
    if roi_path.exists():
        with open(roi_path, 'r') as f:
            roi_data = json.load(f)
        if roi_data.get('polygons'):
            polygon = roi_data['polygons'][0]['points']
            print(f"已載入 ROI: {len(polygon)} 個頂點")
    
    # 使用 ROI 的質心作為 template_point（如果有 ROI）
    if polygon:
        import numpy as np
        poly_array = np.array(polygon)
        template_point = (float(np.mean(poly_array[:, 0])), float(np.mean(poly_array[:, 1])))
        print(f"使用 ROI 質心作為 Template Point: {template_point}")
    else:
        template_point = (500, 1770)
        print(f"Template Point: {template_point}")
    
    print(f"Template: {template_path}")
    print(f"Target: {target_path}")
    
    print("\n執行 3D 轉換...")
    result = matcher.transform_point_3d(
        template_path=template_path,
        target_path=target_path,
        template_point=template_point,
        polygon=polygon,
    )
    
    print("\n=== 結果 ===")
    print(f"Success: {result['success']}")
    print(f"Message: {result.get('message', '')}")
    
    if result['success']:
        print(f"\nTarget Pixel: {result['target_pixel']}")
        print(f"Point 3D (World): {result.get('point_3d_world')}")
        print(f"Point 3D (Cam2): {result.get('point_3d_cam2')}")
        print(f"Focals: {result.get('focals')}")
        print(f"\nT_rel (4x4):\n{result.get('T_rel')}")
        
        # 生成視覺化
        output_path = str(PROJECT_ROOT / "output_test/test_3d_matcher_tower.png")
        print(f"\n生成視覺化到: {output_path}")
        matcher.visualize_transform(
            template_path=template_path,
            target_path=target_path,
            template_point=template_point,
            polygon=polygon,
            output_path=output_path
        )

if __name__ == "__main__":
    main()
