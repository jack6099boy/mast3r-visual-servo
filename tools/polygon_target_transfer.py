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

可複用模組函數：
    - load_mast3r_model(device) - 載入 MASt3R 模型
    - compute_matches(model, img1_path, img2_path, device, size) - 計算特徵匹配點
    - filter_matches_in_polygon(matches_im0, matches_im1, polygon) - 過濾多邊形內的匹配點
    - visualize_transfer(...) - 視覺化轉移結果
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
from typing import Dict, Any, Tuple, List, Optional

# Add mast3r-research to Python path
MAST3R_PATH = Path(__file__).parent.parent / 'mast3r-research'
sys.path.insert(0, str(MAST3R_PATH))

try:
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    MAST3R_AVAILABLE = True
except ImportError as e:
    print(f"MASt3R import error: {e}")
    MAST3R_AVAILABLE = False
    AsymmetricMASt3R = None
    fast_reciprocal_NNs = None
    inference = None
    load_images = None


# =============================================================================
# 獨立可複用函數 (模組化 API)
# =============================================================================

def load_mast3r_model(device: str = 'mps', model_name: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"):
    """
    載入 MASt3R 模型
    
    Args:
        device: 計算設備 ('mps', 'cuda', 'cpu')
        model_name: HuggingFace 模型名稱
        
    Returns:
        model: 載入的 MASt3R 模型
        
    Raises:
        ImportError: 若 MASt3R 不可用
    """
    if not MAST3R_AVAILABLE:
        raise ImportError("MASt3R is not available. Please install it first.")
    
    print(f"載入 MASt3R 模型中... (device={device})")
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device).eval()
    print("MASt3R 模型載入完成")
    return model


def compute_matches(model, img1_path: str, img2_path: str, device: str = 'mps', 
                    size: int = 512, use_original_coords: bool = True) -> Dict[str, Any]:
    """
    計算兩張圖片的特徵匹配點
    
    Args:
        model: MASt3R 模型
        img1_path: 第一張圖片路徑 (template)
        img2_path: 第二張圖片路徑 (target)
        device: 計算設備
        size: MASt3R 處理尺寸 (default: 512)
        use_original_coords: 是否轉換為原圖座標 (若 False，返回 512x512 座標)
        
    Returns:
        dict: {
            'matches_im0': np.ndarray [N, 2] - img1 上的匹配點座標
            'matches_im1': np.ndarray [N, 2] - img2 上的匹配點座標
            'img0_shape': (H, W) - img1 原始尺寸
            'img1_shape': (H, W) - img2 原始尺寸
            'view1': dict - MASt3R view1 資訊
            'view2': dict - MASt3R view2 資訊
        }
    """
    if not MAST3R_AVAILABLE:
        raise ImportError("MASt3R is not available. Please install it first.")
    
    # 載入影像並執行推論
    images = load_images([str(img1_path), str(img2_path)], size=size)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    # 提取描述符並進行匹配
    desc1 = pred1['desc'].squeeze(0).detach()
    desc2 = pred2['desc'].squeeze(0).detach()
    
    matches_im0, matches_im1 = fast_reciprocal_NNs(
        desc1, desc2, subsample_or_initxy1=8,
        device=device, dist='dot', block_size=2**13
    )
    
    # 轉為 numpy
    if hasattr(matches_im0, 'cpu'):
        matches_im0 = matches_im0.cpu().numpy()
    if hasattr(matches_im1, 'cpu'):
        matches_im1 = matches_im1.cpu().numpy()
    
    matches_im0 = np.asarray(matches_im0, dtype=np.float32)
    matches_im1 = np.asarray(matches_im1, dtype=np.float32)
    
    # 獲取原圖尺寸 (使用 cv2 直接讀取，避免 true_shape 的問題)
    # 注意: view['true_shape'] 返回的是處理後尺寸，不是原圖尺寸
    img0 = cv2.imread(str(img1_path))
    img1 = cv2.imread(str(img2_path))
    H0, W0 = img0.shape[:2]
    H1, W1 = img1.shape[:2]
    
    if use_original_coords:
        # 計算縮放比例和 padding，轉換為原圖座標
        size_f = float(size)
        
        # img0 (template)
        scale0 = size_f / max(H0, W0)
        resized_h0 = H0 * scale0
        resized_w0 = W0 * scale0
        pad_top0 = (size_f - resized_h0) / 2
        pad_left0 = (size_f - resized_w0) / 2
        
        # img1 (target)
        scale1 = size_f / max(H1, W1)
        resized_h1 = H1 * scale1
        resized_w1 = W1 * scale1
        pad_top1 = (size_f - resized_h1) / 2
        pad_left1 = (size_f - resized_w1) / 2
        
        # 邊緣過濾 (在處理後座標系統)
        valid0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < size_f - 3) & \
                 (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < size_f - 3)
        valid1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < size_f - 3) & \
                 (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < size_f - 3)
        valid = valid0 & valid1
        matches0_proc = matches_im0[valid]
        matches1_proc = matches_im1[valid]
        
        # 轉換為原圖座標 (去除 padding 並反向縮放)
        orig_x0 = (matches0_proc[:, 0] - pad_left0) / scale0
        orig_y0 = (matches0_proc[:, 1] - pad_top0) / scale0
        orig_x1 = (matches1_proc[:, 0] - pad_left1) / scale1
        orig_y1 = (matches1_proc[:, 1] - pad_top1) / scale1
        
        orig_pts0 = np.stack([orig_x0, orig_y0], axis=1)
        orig_pts1 = np.stack([orig_x1, orig_y1], axis=1)
        
        # 邊界檢查 (在原圖座標系統)
        valid_orig0 = (orig_pts0[:, 0] >= 0) & (orig_pts0[:, 0] < W0) & \
                      (orig_pts0[:, 1] >= 0) & (orig_pts0[:, 1] < H0)
        valid_orig1 = (orig_pts1[:, 0] >= 0) & (orig_pts1[:, 0] < W1) & \
                      (orig_pts1[:, 1] >= 0) & (orig_pts1[:, 1] < H1)
        valid_orig = valid_orig0 & valid_orig1
        
        matches_im0 = orig_pts0[valid_orig]
        matches_im1 = orig_pts1[valid_orig]
    
    return {
        'matches_im0': matches_im0,
        'matches_im1': matches_im1,
        'img0_shape': (H0, W0),
        'img1_shape': (H1, W1),
        'view1': view1,
        'view2': view2
    }


def point_in_polygon(point, polygon) -> bool:
    """檢查點是否在多邊形內"""
    polygon_np = np.array(polygon, dtype=np.float32)
    result = cv2.pointPolygonTest(polygon_np, (float(point[0]), float(point[1])), False)
    return result >= 0


def filter_matches_in_polygon(matches_im0: np.ndarray, matches_im1: np.ndarray, 
                               polygon: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    過濾在多邊形內的匹配點
    
    Args:
        matches_im0: [N, 2] img0 上的匹配點座標
        matches_im1: [N, 2] img1 上的匹配點座標
        polygon: 多邊形頂點 [[x1, y1], [x2, y2], ...]
        
    Returns:
        filtered_im0: 在多邊形內的 img0 匹配點
        filtered_im1: 對應的 img1 匹配點
    """
    polygon_np = np.array(polygon, dtype=np.float32)
    mask = []
    for pt in matches_im0:
        in_poly = cv2.pointPolygonTest(polygon_np, (float(pt[0]), float(pt[1])), False) >= 0
        mask.append(in_poly)
    mask = np.array(mask)
    return matches_im0[mask], matches_im1[mask]


def visualize_transfer(img0_path: str, img1_path: str, 
                        polygon_src: np.ndarray, polygon_dst: Optional[np.ndarray],
                        matches_in_roi: Tuple[np.ndarray, np.ndarray],
                        target_point_src: Optional[np.ndarray] = None,
                        target_point_dst: Optional[Tuple[float, float]] = None,
                        output_path: str = 'transfer_result.png',
                        title_info: str = '',
                        use_processed_images: bool = False,
                        view1: Optional[Dict] = None,
                        view2: Optional[Dict] = None):
    """
    視覺化轉移結果
    
    Args:
        img0_path: template 影像路徑
        img1_path: target 影像路徑
        polygon_src: 源多邊形頂點 [[x1, y1], ...]
        polygon_dst: 目標多邊形頂點 (可選)
        matches_in_roi: (matches_im0, matches_im1) 在 ROI 內的匹配點
        target_point_src: 源目標點 [x, y] (可選)
        target_point_dst: 轉換後目標點 (x, y) (可選)
        output_path: 輸出圖片路徑
        title_info: 額外標題資訊
        use_processed_images: 是否使用 MASt3R 處理後的影像 (需提供 view1, view2)
        view1, view2: MASt3R view 物件 (若 use_processed_images=True)
    """
    if use_processed_images and view1 is not None and view2 is not None:
        # 使用 MASt3R 處理後的影像
        image_mean = torch.as_tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        image_std = torch.as_tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        
        img_template = (view1['img'] * image_std + image_mean).squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_target = (view2['img'] * image_std + image_mean).squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        img_template = np.clip(img_template, 0, 1)
        img_target = np.clip(img_target, 0, 1)
    else:
        # 使用原圖
        img_template = cv2.imread(str(img0_path))
        img_target = cv2.imread(str(img1_path))
        img_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2RGB) / 255.0
        img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB) / 255.0
    
    matches_template, matches_target = matches_in_roi
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Left: Template + polygon + target point
    axes[0].imshow(img_template)
    poly_patch = patches.Polygon(polygon_src, linewidth=2, edgecolor='lime', facecolor='lime', alpha=0.2)
    axes[0].add_patch(poly_patch)
    if target_point_src is not None:
        axes[0].plot(target_point_src[0], target_point_src[1], 'r*', markersize=20, 
                     markeredgecolor='white', markeredgewidth=2)
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
    
    axes[1].set_title(f'Matches in ROI ({len(matches_template)}, showing {n_show})\n{title_info}', fontsize=11)
    axes[1].axis('off')
    
    # Right: Target + transformed target point or polygon
    axes[2].imshow(img_target)
    
    if polygon_dst is not None:
        poly_patch_dst = patches.Polygon(polygon_dst, linewidth=2, edgecolor='lime', facecolor='lime', alpha=0.2)
        axes[2].add_patch(poly_patch_dst)
    
    if target_point_dst is not None:
        if isinstance(target_point_dst, dict) and 'epiline' in target_point_dst:
            # Draw epiline for fundamental matrix
            epiline = target_point_dst['epiline']
            a, b, c = epiline
            x_vals = np.array([0, w2])
            if abs(b) > 1e-10:
                y_vals = (-a * x_vals - c) / b
                axes[2].plot(x_vals, y_vals, 'r-', linewidth=2, label='Epiline')
            axes[2].set_title(f'Target\nEpiline (red line)', fontsize=12)
        else:
            axes[2].plot(target_point_dst[0], target_point_dst[1], 'r*', markersize=20,
                         markeredgecolor='white', markeredgewidth=2)
            axes[2].set_title(f'Target\nTransformed Point (red star)\n({target_point_dst[0]:.1f}, {target_point_dst[1]:.1f})', fontsize=12)
    else:
        axes[2].set_title('Target\nTransferred ROI', fontsize=12)
    
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


# =============================================================================
# 幾何變換函數
# =============================================================================

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


# =============================================================================
# 類別封裝 (保持向後相容)
# =============================================================================

class PolygonTargetTransfer:
    """多邊形目標點轉換器 (使用獨立函數的封裝類別)"""
    
    def __init__(self, device='mps'):
        self.device = device
        self.model = None
        
    def load_model(self):
        """載入 MASt3R 模型"""
        if self.model is None:
            self.model = load_mast3r_model(self.device)
        return self.model
    
    def get_matches(self, template_path, target_path, size=512):
        """獲取 template 和 target 之間的匹配點"""
        model = self.load_model()
        result = compute_matches(model, template_path, target_path, self.device, size)
        
        # 保持向後相容的返回格式
        return {
            'matches_template': result['matches_im0'],
            'matches_target': result['matches_im1'],
            'template_shape': result['img0_shape'],
            'target_shape': result['img1_shape'],
            'view1': result['view1'],
            'view2': result['view2']
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
            visualize_transfer(
                img0_path=template_path,
                img1_path=target_path,
                polygon_src=polygon,
                polygon_dst=None,
                matches_in_roi=(matches_in_poly_template, matches_in_poly_target),
                target_point_src=target_point,
                target_point_dst=transformed_point,
                output_path=output_path,
                title_info=f"Method: {info.get('description', transform_method)}",
                use_processed_images=True,
                view1=match_result['view1'],
                view2=match_result['view2']
            )
        
        return result


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
