"""
3D Matcher 測試腳本
==================

驗證 TemplateMatcher3D 的核心功能。

使用方式:
    cd template_system
    python test_matcher_3d.py
    
或指定自定義圖片:
    python test_matcher_3d.py --template path/to/template.jpg --target path/to/target.jpg
"""

import sys
from pathlib import Path

# 設置路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from template_system.matcher_3d import TemplateMatcher3D, MAST3R_AVAILABLE


def test_basic_initialization():
    """測試基本初始化"""
    print("\n" + "="*60)
    print("Test 1: Basic Initialization")
    print("="*60)
    
    if not MAST3R_AVAILABLE:
        print("⚠️  MASt3R not available, skipping test")
        return False
    
    try:
        matcher = TemplateMatcher3D()
        print(f"✓ Device: {matcher.device}")
        print(f"✓ Model name: {matcher.model_name}")
        print(f"✓ Image size: {matcher.image_size}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_model_loading():
    """測試模型載入（延遲載入）"""
    print("\n" + "="*60)
    print("Test 2: Model Loading (Lazy)")
    print("="*60)
    
    if not MAST3R_AVAILABLE:
        print("⚠️  MASt3R not available, skipping test")
        return False
    
    try:
        matcher = TemplateMatcher3D()
        print("✓ Matcher created (model not loaded yet)")
        
        # 觸發模型載入
        model = matcher.model
        if model is not None:
            print("✓ Model loaded successfully")
            return True
        else:
            print("✗ Model is None")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transform_with_default_images():
    """使用預設圖片測試轉換"""
    print("\n" + "="*60)
    print("Test 3: Transform with Default Images")
    print("="*60)
    
    if not MAST3R_AVAILABLE:
        print("⚠️  MASt3R not available, skipping test")
        return False
    
    # 使用 tower 模板作為測試
    project_root = Path(__file__).parent.parent
    template_path = project_root / "templates/a_lab/tower/tower.jpg"
    target_path = project_root / "mast3r-research/test_target.jpg"
    roi_path = project_root / "templates/a_lab/tower/tower_roi.json"
    
    if not template_path.exists():
        print(f"⚠️  Template not found: {template_path}")
        return False
    
    if not target_path.exists():
        print(f"⚠️  Target not found: {target_path}")
        return False
    
    # 讀取 ROI 來獲取多邊形和關鍵點
    polygon = None
    template_point = None
    if roi_path.exists():
        with open(roi_path, 'r') as f:
            roi_data = json.load(f)
        
        if 'polygons' in roi_data and roi_data['polygons']:
            polygon = roi_data['polygons'][0]['points']
            print(f"✓ Loaded polygon with {len(polygon)} points")
            
            # 使用多邊形的質心作為測試點
            poly_np = np.array(polygon[:-1])  # 移除重複的最後一點
            template_point = tuple(poly_np.mean(axis=0))
            print(f"✓ Template point (polygon centroid): {template_point}")
        
        if 'keypoints' in roi_data and roi_data['keypoints']:
            # 使用第一個 keypoint
            for name, coords in roi_data['keypoints'].items():
                template_point = tuple(coords[:2])
                print(f"✓ Template keypoint '{name}': {template_point}")
                break
    
    if template_point is None:
        # 使用圖片中心
        import cv2
        img = cv2.imread(str(template_path))
        h, w = img.shape[:2]
        template_point = (w/2, h/2)
        print(f"✓ Using image center as template point: {template_point}")
    
    try:
        matcher = TemplateMatcher3D()
        print("\nRunning transform_point_3d...")
        
        result = matcher.transform_point_3d(
            str(template_path),
            str(target_path),
            template_point,
            polygon=polygon,
            niter1=200,  # 減少迭代以加速測試
            niter2=0
        )
        
        print(f"\n--- Result ---")
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        
        if result['success']:
            print(f"Target pixel: {result['target_pixel']}")
            print(f"Target pixel (res): {result['target_pixel_res']}")
            print(f"3D world: {result['point_3d_world']}")
            print(f"3D cam2: {result['point_3d_cam2']}")
            print(f"Focals: {result['focals']}")
            
            # 簡單驗證
            if result['T_rel'] is not None:
                print(f"T_rel shape: {result['T_rel'].shape}")
                assert result['T_rel'].shape == (4, 4), "T_rel should be 4x4"
                print("✓ T_rel shape is correct")
            
            # 保存視覺化
            output_path = project_root / "output_test/test_matcher_3d_result.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            matcher.visualize_transform(
                str(template_path),
                str(target_path),
                template_point,
                polygon=polygon,
                output_path=str(output_path)
            )
            print(f"\n✓ Visualization saved to {output_path}")
            
            return True
        else:
            print(f"✗ Transform failed: {result['message']}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_relative_pose_computation():
    """測試相對姿態計算"""
    print("\n" + "="*60)
    print("Test 4: Relative Pose Computation")
    print("="*60)
    
    from template_system.matcher_3d import compute_relative_pose
    
    # 建立測試用的姿態矩陣
    poses = np.zeros((2, 4, 4))
    
    # Camera 1: 單位矩陣（原點，無旋轉）
    poses[0] = np.eye(4)
    
    # Camera 2: 沿 Z 軸平移 1 單位
    poses[1] = np.eye(4)
    poses[1, 2, 3] = 1.0  # z translation
    
    T_rel = compute_relative_pose(poses)
    
    print(f"Camera 1 pose:\n{poses[0]}")
    print(f"\nCamera 2 pose:\n{poses[1]}")
    print(f"\nT_rel (cam1 to cam2):\n{T_rel}")
    
    # 驗證：T_rel 應該將 camera 1 座標系的點轉換到 camera 2 座標系
    # 由於 camera 2 在 z=1，camera 1 在 z=0
    # 一個在 camera 1 原點 (0,0,0) 的點，在 camera 2 應該是 (0,0,-1)
    p1 = np.array([0, 0, 0, 1])
    p2 = T_rel @ p1
    expected = np.array([0, 0, -1, 1])
    
    print(f"\nPoint in cam1: {p1[:3]}")
    print(f"Point in cam2: {p2[:3]}")
    print(f"Expected: {expected[:3]}")
    
    if np.allclose(p2, expected):
        print("✓ Relative pose computation is correct")
        return True
    else:
        print("✗ Relative pose computation is incorrect")
        return False


def test_projection():
    """測試投影函數"""
    print("\n" + "="*60)
    print("Test 5: Projection to Image")
    print("="*60)
    
    from template_system.matcher_3d import project_to_image
    
    # 測試：焦距 500，主點 (320, 240)
    focal = 500.0
    pp = (320.0, 240.0)
    
    # 一個在相機前方 1 單位的點 (0, 0, 1)
    point_3d = np.array([0, 0, 1])
    u, v = project_to_image(point_3d, focal, pp)
    
    print(f"3D point: {point_3d}")
    print(f"Focal: {focal}, PP: {pp}")
    print(f"Projected: ({u}, {v})")
    
    # 預期：(320, 240) - 正對相機中心
    if abs(u - 320) < 1e-6 and abs(v - 240) < 1e-6:
        print("✓ Projection is correct for center point")
    else:
        print("✗ Projection is incorrect for center point")
    
    # 測試：偏移的 3D 點
    point_3d = np.array([1, 0.5, 2])
    u, v = project_to_image(point_3d, focal, pp)
    
    expected_u = focal * 1 / 2 + 320  # = 570
    expected_v = focal * 0.5 / 2 + 240  # = 365
    
    print(f"\n3D point: {point_3d}")
    print(f"Projected: ({u}, {v})")
    print(f"Expected: ({expected_u}, {expected_v})")
    
    if abs(u - expected_u) < 1e-6 and abs(v - expected_v) < 1e-6:
        print("✓ Projection is correct for offset point")
        return True
    else:
        print("✗ Projection is incorrect for offset point")
        return False


def main():
    """執行所有測試"""
    print("\n" + "="*60)
    print("        TemplateMatcher3D Test Suite")
    print("="*60)
    
    results = []
    
    # 基本測試（不需要 GPU）
    results.append(("Basic Initialization", test_basic_initialization()))
    results.append(("Relative Pose Computation", test_relative_pose_computation()))
    results.append(("Projection to Image", test_projection()))
    
    # 需要模型的測試
    results.append(("Model Loading", test_model_loading()))
    results.append(("Transform with Images", test_transform_with_default_images()))
    
    # 總結
    print("\n" + "="*60)
    print("                   Summary")
    print("="*60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results:
        if result is True:
            status = "✓ PASSED"
            passed += 1
        elif result is False:
            status = "✗ FAILED"
            failed += 1
        else:
            status = "⚠ SKIPPED"
            skipped += 1
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    return failed == 0


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test TemplateMatcher3D')
    parser.add_argument('--template', help='Custom template image path')
    parser.add_argument('--target', help='Custom target image path')
    parser.add_argument('--point', type=float, nargs=2, help='Template point (x, y)')
    args = parser.parse_args()
    
    if args.template and args.target:
        # 自定義測試
        print("Running custom test...")
        if not MAST3R_AVAILABLE:
            print("Error: MASt3R not available")
            sys.exit(1)
        
        matcher = TemplateMatcher3D()
        point = tuple(args.point) if args.point else None
        
        if point is None:
            import cv2
            img = cv2.imread(args.template)
            h, w = img.shape[:2]
            point = (w/2, h/2)
        
        result = matcher.transform_point_3d(args.template, args.target, point)
        print(f"\nResult: {result}")
        
        if result['success']:
            output_path = "custom_test_result.png"
            matcher.visualize_transform(args.template, args.target, point, 
                                        output_path=output_path)
    else:
        # 執行測試套件
        success = main()
        sys.exit(0 if success else 1)
