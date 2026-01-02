"""
多邊形 ROI 目標點轉換工具
=========================
功能：
1. 在 template 影像上定義多邊形 ROI
2. 在多邊形內定義自定義目標點
3. 使用 MASt3R 匹配 template → target 影像
4. 計算 Affine Transform Matrix
5. 將目標點轉換到 target 影像上

用法：
    python polygon_target_transfer.py --template <template.jpg> --target <target.jpg> \
        --polygon "[[x1,y1],[x2,y2],...]" --target_point "[x,y]"
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import torch
import json
import argparse
import sys
from pathlib import Path

# Add mast3r-research to Python path
MAST3R_PATH = Path(__file__).parent.parent / 'mast3r-research'
sys.path.insert(0, str(MAST3R_PATH))

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images


def point_in_polygon(point, polygon):
    """檢查點是否在多邊形內"""
    polygon_np = np.array(polygon, dtype=np.float32)
    result = cv2.pointPolygonTest(polygon_np, (float(point[0]), float(point[1])), False)
    return result >= 0


def filter_matches_in_polygon(matches_im0, matches_im1, polygon):
    """過濾在多邊形內的匹配點"""
    polygon_np = np.array(polygon, dtype=np.float32)
    mask = []
    for pt in matches_im0:
        in_poly = cv2.pointPolygonTest(polygon_np, (float(pt[0]), float(pt[1])), False) >= 0
        mask.append(in_poly)
    mask = np.array(mask)
    return matches_im0[mask], matches_im1[mask]


def compute_transform(src_pts, dst_pts, method='partial', ransac_thresh=5.0):
    """
    計算幾何變換矩陣
    
    method:
        'partial'     - estimateAffinePartial2D (4 DOF: rotation, uniform scale, translation)
        'affine'      - estimateAffine2D (6 DOF: full affine)
        'homography'  - findHomography (8 DOF: perspective transform)
        'fundamental' - findFundamentalMat (7 DOF: epipolar geometry)
    
    Returns:
        M: 變換矩陣
        inliers: inlier mask
        info: 額外資訊 (如 fundamental matrix 的極線誤差等)
    """
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    
    min_pts = {'partial': 2, 'affine': 3, 'homography': 4, 'fundamental': 8}
    if len(src_pts) < min_pts.get(method, 3):
        raise ValueError(f"Method '{method}' 需要至少 {min_pts[method]} 個匹配點，目前只有 {len(src_pts)} 個")
    
    info = {'method': method}
    
    if method == 'partial':
        # 4 DOF: rotation, uniform scale, tx, ty
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh
        )
        info['dof'] = 4
        info['description'] = 'Rotation + Uniform Scale + Translation'
        
    elif method == 'affine':
        # 6 DOF: full affine (rotation, non-uniform scale, shear, translation)
        M, inliers = cv2.estimateAffine2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh
        )
        info['dof'] = 6
        info['description'] = 'Full Affine (including shear)'
        
    elif method == 'homography':
        # 8 DOF: perspective transform
        M, inliers = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh
        )
        info['dof'] = 8
        info['description'] = 'Perspective Transform (for planar objects)'
        
    elif method == 'fundamental':
        # 7 DOF: fundamental matrix (epipolar geometry)
        M, inliers = cv2.findFundamentalMat(
            src_pts, dst_pts,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=ransac_thresh
        )
        info['dof'] = 7
        info['description'] = 'Fundamental Matrix (epipolar geometry, for 3D pose)'
        
        # 計算平均極線誤差
        if M is not None and inliers is not None:
            inlier_mask = inliers.ravel().astype(bool)
            if np.any(inlier_mask):
                src_inliers = src_pts[inlier_mask]
                dst_inliers = dst_pts[inlier_mask]
                # 計算極線誤差: x2^T * F * x1 應該接近 0
                errors = []
                for s, d in zip(src_inliers, dst_inliers):
                    pt1 = np.array([s[0], s[1], 1])
                    pt2 = np.array([d[0], d[1], 1])
                    err = abs(pt2 @ M @ pt1)
                    errors.append(err)
                info['mean_epipolar_error'] = np.mean(errors)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return M, inliers, info


def transform_point(M, point, method='affine'):
    """
    使用變換矩陣轉換單一點
    
    對於 Affine (2x3): pt' = M @ [x, y, 1]^T
    對於 Homography (3x3): pt' = M @ [x, y, 1]^T, 然後除以 w
    對於 Fundamental: 不能直接轉換點，返回極線係數
    """
    pt = np.array([[point[0]], [point[1]], [1]], dtype=np.float32)
    
    if method == 'fundamental':
        # Fundamental matrix: 返回極線 l = F @ x
        # l = [a, b, c], 極線方程: ax + by + c = 0
        pt_flat = np.array([point[0], point[1], 1], dtype=np.float32)
        epiline = M @ pt_flat
        return {'epiline': epiline, 'type': 'fundamental'}
    
    if M.shape == (2, 3):  # Affine (partial or full)
        transformed = M @ pt
        return float(transformed[0, 0]), float(transformed[1, 0])
    elif M.shape == (3, 3):  # Homography
        transformed = M @ pt
        w = transformed[2, 0]
        if abs(w) < 1e-10:
            return None  # 無效轉換 (投影到無窮遠)
        return float(transformed[0, 0] / w), float(transformed[1, 0] / w)
    else:
        raise ValueError(f"Unexpected matrix shape: {M.shape}")


def compute_reprojection_error(M, src_pts, dst_pts, method='affine'):
    """計算重投影誤差 (用於評估變換質量)"""
    errors = []
    for src, dst in zip(src_pts, dst_pts):
        transformed = transform_point(M, src, method)
        if transformed is None or isinstance(transformed, dict):
            continue
        err = np.sqrt((transformed[0] - dst[0])**2 + (transformed[1] - dst[1])**2)
        errors.append(err)
    return np.mean(errors) if errors else float('inf'), np.std(errors) if errors else 0


class PolygonTargetTransfer:
    def __init__(self, device='mps'):
        self.device = device
        self.model = None
        
    def load_model(self):
        """載入 MASt3R 模型"""
        if self.model is None:
            print("載入 MASt3R 模型中...")
            model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
            self.model = AsymmetricMASt3R.from_pretrained(model_name).to(self.device)
        return self.model
    
    def get_matches(self, template_path, target_path, size=512):
        """獲取 template 和 target 之間的匹配點"""
        model = self.load_model()
        
        images = load_images([template_path, target_path], size=size)
        output = inference([tuple(images)], model, self.device, batch_size=1, verbose=True)
        
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        
        desc1 = pred1['desc'].squeeze(0).detach()
        desc2 = pred2['desc'].squeeze(0).detach()
        
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=8,
            device=self.device, dist='dot', block_size=2**13
        )
        
        # 轉為 numpy
        if hasattr(matches_im0, 'cpu'):
            matches_im0 = matches_im0.cpu().numpy()
        if hasattr(matches_im1, 'cpu'):
            matches_im1 = matches_im1.cpu().numpy()
        
        matches_im0 = np.asarray(matches_im0, dtype=np.float32)
        matches_im1 = np.asarray(matches_im1, dtype=np.float32)
        
        # 獲取縮放比例 (原圖 → 處理尺寸)
        H0, W0 = int(view1['true_shape'][0][0]), int(view1['true_shape'][0][1])
        H1, W1 = int(view2['true_shape'][0][0]), int(view2['true_shape'][0][1])
        
        return {
            'matches_template': matches_im0,
            'matches_target': matches_im1,
            'template_shape': (H0, W0),
            'target_shape': (H1, W1),
            'view1': view1,
            'view2': view2
        }
    
    def transfer_target_point(self, template_path, target_path, polygon, target_point,
                               transform_method='partial', ransac_thresh=5.0,
                               visualize=True, output_path='transfer_result.png'):
        """
        主要功能：將 template 上的目標點轉換到 target 影像上
        
        Args:
            template_path: template 影像路徑
            target_path: target 影像路徑
            polygon: 多邊形頂點 [[x1,y1], [x2,y2], ...]
            target_point: 目標點 [x, y]
            transform_method: 變換方法
                - 'partial': Affine Partial (4 DOF) - 旋轉+均勻縮放+平移
                - 'affine': Full Affine (6 DOF) - 包含剪切
                - 'homography': Homography (8 DOF) - 透視變換
                - 'fundamental': Fundamental Matrix (7 DOF) - 極線幾何
            ransac_thresh: RANSAC 閾值 (像素)
            visualize: 是否產生視覺化
            output_path: 輸出圖片路徑
            
        Returns:
            dict: {
                'transformed_point': (x, y) or epiline dict,
                'transform_matrix': M,
                'num_matches_in_polygon': int,
                'num_inliers': int,
                'transform_info': dict,
                'reprojection_error': (mean, std)
            }
        """
        polygon = np.array(polygon, dtype=np.float32)
        target_point = np.array(target_point, dtype=np.float32)
        
        # 檢查目標點是否在多邊形內
        if not point_in_polygon(target_point, polygon):
            print("Warning: Target point is outside polygon!")
        
        # 獲取匹配點
        print("Running MASt3R feature matching...")
        match_result = self.get_matches(template_path, target_path)
        
        matches_template = match_result['matches_template']
        matches_target = match_result['matches_target']
        
        print(f"Total matches: {len(matches_template)}")
        
        # 過濾多邊形內的匹配點
        matches_in_poly_template, matches_in_poly_target = filter_matches_in_polygon(
            matches_template, matches_target, polygon
        )
        
        print(f"Matches in polygon: {len(matches_in_poly_template)}")
        
        min_pts = {'partial': 2, 'affine': 3, 'homography': 4, 'fundamental': 8}
        if len(matches_in_poly_template) < min_pts.get(transform_method, 3):
            raise ValueError(f"Not enough matches in polygon (need {min_pts[transform_method]}, got {len(matches_in_poly_template)})")
        
        # 計算變換
        M, inliers, info = compute_transform(
            matches_in_poly_template, matches_in_poly_target,
            method=transform_method, ransac_thresh=ransac_thresh
        )
        
        if M is None:
            raise ValueError(f"Failed to compute {transform_method} transform")
        
        num_inliers = np.sum(inliers) if inliers is not None else len(matches_in_poly_template)
        inlier_ratio = num_inliers / len(matches_in_poly_template) * 100
        print(f"Transform method: {info['description']}")
        print(f"Inliers: {num_inliers}/{len(matches_in_poly_template)} ({inlier_ratio:.1f}%)")
        
        # 計算重投影誤差
        inlier_mask = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(matches_in_poly_template), dtype=bool)
        reproj_err_mean, reproj_err_std = compute_reprojection_error(
            M,
            matches_in_poly_template[inlier_mask],
            matches_in_poly_target[inlier_mask],
            method=transform_method
        )
        print(f"Reprojection error: {reproj_err_mean:.2f} ± {reproj_err_std:.2f} pixels")
        
        # 轉換目標點
        transformed_point = transform_point(M, target_point, method=transform_method)
        print(f"Original target point: {target_point}")
        
        if transform_method == 'fundamental':
            epiline = transformed_point['epiline']
            print(f"Epiline coefficients: [{epiline[0]:.4f}, {epiline[1]:.4f}, {epiline[2]:.4f}]")
            print("Note: Fundamental matrix gives epipolar line, not direct point mapping")
        else:
            print(f"Transformed point: ({transformed_point[0]:.2f}, {transformed_point[1]:.2f})")
        
        result = {
            'transformed_point': transformed_point,
            'transform_matrix': M,
            'num_matches_in_polygon': len(matches_in_poly_template),
            'num_inliers': int(num_inliers),
            'inlier_ratio': inlier_ratio,
            'transform_info': info,
            'reprojection_error': (reproj_err_mean, reproj_err_std),
            'all_matches': (matches_template, matches_target),
            'polygon_matches': (matches_in_poly_template, matches_in_poly_target)
        }
        
        if visualize:
            self._visualize(
                template_path, target_path, polygon, target_point,
                transformed_point, matches_in_poly_template, matches_in_poly_target,
                match_result, output_path, transform_method, info
            )
        
        return result
    
    def _visualize(self, template_path, target_path, polygon, target_point,
                   transformed_point, matches_template, matches_target, match_result,
                   output_path, transform_method='partial', info=None):
        """視覺化結果"""
        view1 = match_result['view1']
        view2 = match_result['view2']
        
        image_mean = torch.as_tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        image_std = torch.as_tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        
        img_template = (view1['img'] * image_std + image_mean).squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_target = (view2['img'] * image_std + image_mean).squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        img_template = np.clip(img_template, 0, 1)
        img_target = np.clip(img_target, 0, 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Left: Template + polygon + target point
        axes[0].imshow(img_template)
        poly_patch = patches.Polygon(polygon, linewidth=2, edgecolor='lime', facecolor='lime', alpha=0.2)
        axes[0].add_patch(poly_patch)
        axes[0].plot(target_point[0], target_point[1], 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
        axes[0].set_title(f'Template\nPolygon ROI + Target Point (red star)', fontsize=12)
        axes[0].axis('off')
        
        # Middle: Match lines
        h1, w1 = img_template.shape[:2]
        h2, w2 = img_target.shape[:2]
        
        max_h = max(h1, h2)
        combined = np.zeros((max_h, w1 + w2, 3))
        combined[:h1, :w1] = img_template
        combined[:h2, w1:] = img_target
        
        axes[1].imshow(combined)
        
        n_show = min(30, len(matches_template))
        if n_show > 0:
            indices = np.linspace(0, len(matches_template) - 1, n_show).astype(int)
            cmap = plt.get_cmap('jet')
            for i, idx in enumerate(indices):
                x0, y0 = matches_template[idx]
                x1, y1 = matches_target[idx]
                color = cmap(i / max(n_show - 1, 1))
                axes[1].plot([x0, x1 + w1], [y0, y1], '-', color=color, linewidth=1, alpha=0.7)
                axes[1].plot(x0, y0, '+', color=color, markersize=6)
                axes[1].plot(x1 + w1, y1, '+', color=color, markersize=6)
        
        method_name = info.get('description', transform_method) if info else transform_method
        axes[1].set_title(f'Matches ({len(matches_template)}, showing {n_show})\nMethod: {method_name}', fontsize=11)
        axes[1].axis('off')
        
        # Right: Target + transformed target point (or epiline for fundamental)
        axes[2].imshow(img_target)
        
        if transform_method == 'fundamental' and isinstance(transformed_point, dict):
            # Draw epiline
            epiline = transformed_point['epiline']
            a, b, c = epiline
            # y = (-a*x - c) / b
            x_vals = np.array([0, w2])
            if abs(b) > 1e-10:
                y_vals = (-a * x_vals - c) / b
                axes[2].plot(x_vals, y_vals, 'r-', linewidth=2, label='Epiline')
            axes[2].set_title(f'Target\nEpiline (red line)\nFundamental Matrix', fontsize=12)
        else:
            axes[2].plot(transformed_point[0], transformed_point[1], 'r*', markersize=20,
                         markeredgecolor='white', markeredgewidth=2)
            axes[2].set_title(f'Target\nTransformed Point (red star)\n({transformed_point[0]:.1f}, {transformed_point[1]:.1f})', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Polygon ROI Target Point Transfer Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Transform Methods:
  partial     - Affine Partial (4 DOF): rotation + uniform scale + translation
  affine      - Full Affine (6 DOF): + non-uniform scale + shear
  homography  - Homography (8 DOF): perspective transform (best for planar objects)
  fundamental - Fundamental Matrix (7 DOF): epipolar geometry (for 3D pose)

Examples:
  python polygon_target_transfer.py --template img1.jpg --target img2.jpg \\
      --polygon '[[100,50],[200,50],[200,400],[100,400]]' --target_point '[150,200]'
  
  python polygon_target_transfer.py --template img1.jpg --target img2.jpg \\
      --polygon '[[100,50],[200,50],[200,400],[100,400]]' --target_point '[150,200]' \\
      --method homography --ransac_thresh 3.0
        """
    )
    parser.add_argument('--template', type=str, required=True, help='Template image path')
    parser.add_argument('--target', type=str, required=True, help='Target image path')
    parser.add_argument('--polygon', type=str, required=True,
                        help='Polygon vertices JSON: [[x1,y1],[x2,y2],...]')
    parser.add_argument('--target_point', type=str, required=True,
                        help='Target point JSON: [x,y]')
    parser.add_argument('--method', type=str, default='partial',
                        choices=['partial', 'affine', 'homography', 'fundamental'],
                        help='Transform method (default: partial)')
    parser.add_argument('--ransac_thresh', type=float, default=5.0,
                        help='RANSAC threshold in pixels (default: 5.0)')
    parser.add_argument('--output', type=str, default='transfer_result.png',
                        help='Output image path')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device: mps, cuda, cpu')
    
    args = parser.parse_args()
    
    # Parse JSON arguments
    polygon = json.loads(args.polygon)
    target_point = json.loads(args.target_point)
    
    # Execute transfer
    transfer = PolygonTargetTransfer(device=args.device)
    result = transfer.transfer_target_point(
        template_path=args.template,
        target_path=args.target,
        polygon=polygon,
        target_point=target_point,
        transform_method=args.method,
        ransac_thresh=args.ransac_thresh,
        visualize=True,
        output_path=args.output
    )
    
    print("\n=== Results ===")
    info = result['transform_info']
    print(f"Method: {info['description']} ({info['dof']} DOF)")
    
    if args.method == 'fundamental':
        epiline = result['transformed_point']['epiline']
        print(f"Epiline: {epiline[0]:.4f}x + {epiline[1]:.4f}y + {epiline[2]:.4f} = 0")
    else:
        print(f"Transformed point: ({result['transformed_point'][0]:.2f}, {result['transformed_point'][1]:.2f})")
    
    print(f"Matches in polygon: {result['num_matches_in_polygon']}")
    print(f"Inliers: {result['num_inliers']} ({result['inlier_ratio']:.1f}%)")
    print(f"Reproj error: {result['reprojection_error'][0]:.2f} ± {result['reprojection_error'][1]:.2f} px")
    print(f"\nTransform Matrix:\n{result['transform_matrix']}")


if __name__ == '__main__':
    main()
