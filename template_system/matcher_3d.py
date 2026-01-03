"""
3D Template Matcher 模組
========================

使用 MASt3R sparse_global_alignment 進行 3D 點轉換。

核心功能：
1. 統一使用 sparse_global_alignment 獲取一致的座標系數據
2. 支援自定義標記點（不只是 ROI 質心）
3. 返回相對姿態 T_rel（用於視覺伺服）
4. 返回 target 圖片上的像素位置

座標系說明：
- sparse_global_alignment 的 poses[i] 是 camera-to-world 變換
- pts3d 在世界座標系中
- Camera 1 (template) 的 pose 通常接近單位矩陣
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import tempfile
import shutil

# 設置 path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
MAST3R_ROOT = PROJECT_ROOT / "mast3r-research"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MAST3R_ROOT))

# MASt3R 相關 imports
try:
    from mast3r.model import AsymmetricMASt3R
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    MAST3R_AVAILABLE = True
except ImportError as e:
    print(f"MASt3R import error: {e}. 3D matching will not be available.")
    MAST3R_AVAILABLE = False
    AsymmetricMASt3R = None
    load_images = None
    sparse_global_alignment = None


class TemplateMatcher3D:
    """
    使用 MASt3R sparse_global_alignment 進行 3D 點轉換
    
    主要解決原本 3d_polygon_target_transfer.py 中座標系混用的問題：
    - 原本：使用 sparse_global_alignment 獲取 poses，但用 inference() 獲取 pts3d
    - 現在：統一從 sparse_global_alignment 獲取所有數據
    """
    
    def __init__(self, 
                 model_name: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                 device: str = None,
                 image_size: int = 518):
        """
        初始化 3D 匹配器
        
        Args:
            model_name: MASt3R 模型名稱 (HuggingFace)
            device: 計算設備 ('cuda', 'mps', 'cpu')，若為 None 則自動選擇
            image_size: 圖片處理尺寸（518 是 MASt3R 預設）
        """
        self.model_name = model_name
        self.image_size = image_size
        
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
            print(f"[TemplateMatcher3D] Loading model to {self.device}...")
            self._model = AsymmetricMASt3R.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            print(f"[TemplateMatcher3D] Model loaded.")
        return self._model
    
    def get_aligned_scene(self, template_path: str, target_path: str,
                          matching_conf_thr: float = 0.1,
                          niter1: int = 300, niter2: int = 0) -> Dict[str, Any]:
        """
        使用 sparse_global_alignment 獲取一致的場景數據
        
        Args:
            template_path: template 圖片路徑
            target_path: target 圖片路徑
            matching_conf_thr: 匹配置信度閾值
            niter1: 第一階段（粗對齊）迭代次數
            niter2: 第二階段（精細化）迭代次數
        
        Returns:
            dict: {
                'scene': SparseGA 物件,
                'poses': np.ndarray, shape [2, 4, 4]，camera-to-world
                'focals': np.ndarray, shape [2]
                'pps': np.ndarray, shape [2, 2]，主點
                'pts3d': list of np.ndarray，稀疏 3D 點（世界座標系）
                'res_shapes': [(H1, W1), (H2, W2)]，縮放後形狀
                'orig_shapes': [(H1, W1), (H2, W2)]，原始形狀
                'images': [img1, img2]，原始圖片
            }
        """
        if not MAST3R_AVAILABLE:
            raise RuntimeError("MASt3R is not available. Please check installation.")
        
        cache_dir = tempfile.mkdtemp()
        try:
            # 載入圖片
            images = load_images([template_path, target_path], size=self.image_size)
            res_shape1 = images[0]['img'].shape[-2:]  # (H, W)
            res_shape2 = images[1]['img'].shape[-2:]
            
            # 讀取原始圖片
            orig_img1 = cv2.imread(template_path)
            orig_shape1 = orig_img1.shape[:2]  # (H, W)
            orig_img2 = cv2.imread(target_path)
            orig_shape2 = orig_img2.shape[:2]
            
            # 建立 pairs_in 格式
            path1, path2 = template_path, target_path
            pairs_in = [[
                {"idx": 0, "instance": path1, "img": images[0]["img"], 
                 "true_shape": np.array([res_shape1])},
                {"idx": 1, "instance": path2, "img": images[1]["img"], 
                 "true_shape": np.array([res_shape2])}
            ]]
            
            # 調用 sparse_global_alignment
            scene = sparse_global_alignment(
                [path1, path2],
                pairs_in,
                cache_dir,
                self.model,
                device=self.device,
                matching_conf_thr=matching_conf_thr,
                lr1=0.0005,
                niter1=niter1,
                lr2=0.00005,
                niter2=niter2,
            )
            
            # 從 scene 獲取所有數據
            poses = scene.get_im_poses().cpu().numpy()  # [2, 4, 4] cam-to-world
            focals = scene.get_focals().cpu().numpy()   # [2]
            pps = scene.get_principal_points().cpu().numpy()  # [2, 2]
            
            # 獲取稀疏 3D 點
            pts3d_tensors = scene.get_sparse_pts3d()  # list of tensors
            pts3d = [p.cpu().numpy() for p in pts3d_tensors]
            
            # 提取 2D 匹配點（使用 MASt3R inference 直接獲取）
            pts2d_1 = None
            pts2d_2 = None
            try:
                from dust3r.inference import inference
                from mast3r.fast_nn import fast_reciprocal_NNs
                
                output = inference([tuple(pairs_in[0])], self.model, self.device, batch_size=1)
                view1, pred1 = output['view1'], output['pred1']
                view2, pred2 = output['view2'], output['pred2']
                
                desc1 = pred1['desc'].squeeze(0).detach()
                desc2 = pred2['desc'].squeeze(0).detach()
                
                # 獲取匹配點
                matches_im0, matches_im1 = fast_reciprocal_NNs(
                    desc1, desc2,
                    subsample_or_initxy1=8,
                    device=self.device,
                    dist='dot',
                    block_size=2**13
                )
                
                # 將 grid 索引轉為像素座標
                H1, W1 = res_shape1
                H2, W2 = res_shape2
                
                # fast_reciprocal_NNs 直接返回 numpy array
                if hasattr(matches_im0, 'cpu'):
                    pts2d_1 = matches_im0.cpu().numpy().astype(np.float32)
                    pts2d_2 = matches_im1.cpu().numpy().astype(np.float32)
                else:
                    pts2d_1 = np.asarray(matches_im0, dtype=np.float32)
                    pts2d_2 = np.asarray(matches_im1, dtype=np.float32)
                
                print(f"[DEBUG] Extracted {len(pts2d_1)} matching points")
                
            except Exception as e:
                print(f"[DEBUG] Failed to extract pts2d: {e}")
            
            return {
                'scene': scene,
                'poses': poses,
                'focals': focals,
                'pps': pps,
                'pts3d': pts3d,
                'pts2d_1': pts2d_1,
                'pts2d_2': pts2d_2,
                'res_shapes': [tuple(res_shape1), tuple(res_shape2)],
                'orig_shapes': [orig_shape1, orig_shape2],
                'images': [orig_img1, orig_img2],
            }
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)
    
    def get_3d_at_pixel(self, pts3d: np.ndarray, pixel: Tuple[float, float],
                        res_shape: Tuple[int, int], 
                        method: str = 'nearest',
                        pixel_indices: np.ndarray = None) -> Optional[np.ndarray]:
        """
        從稀疏 pts3d 獲取指定像素位置的 3D 座標
        
        注意：sparse_global_alignment 的 pts3d 是稀疏的，
        形狀可能是 [N, 3] 而非 [H, W, 3]。
        
        Args:
            pts3d: sparse_ga 的 pts3d 輸出
            pixel: 像素座標 (x, y)，在縮放後圖片空間
            res_shape: 縮放後圖片形狀 (H, W)
            method: 'nearest' 或 'interpolate'
            pixel_indices: 稀疏點對應的像素索引 [N, 2] (x, y)
        
        Returns:
            np.ndarray: 3D 座標 [x, y, z] 或 None
        """
        x, y = pixel
        H, W = res_shape
        
        # 檢查 pts3d 形狀
        if pts3d.ndim == 2 and pts3d.shape[1] == 3:
            # 稀疏點 [N, 3]
            if pixel_indices is not None and len(pixel_indices) > 0:
                # 使用最近鄰匹配
                distances = np.sqrt((pixel_indices[:, 0] - x)**2 + (pixel_indices[:, 1] - y)**2)
                nearest_idx = np.argmin(distances)
                return pts3d[nearest_idx].copy()
            # fallback：返回平均值
            if len(pts3d) > 0:
                return pts3d.mean(axis=0)
            return None
        
        elif pts3d.ndim == 3:
            # 密集點 [H, W, 3] 
            H_pts, W_pts, _ = pts3d.shape
            
            # 座標映射
            y_idx = int(np.clip(y * H_pts / H, 0, H_pts - 1))
            x_idx = int(np.clip(x * W_pts / W, 0, W_pts - 1))
            
            if method == 'nearest':
                return pts3d[y_idx, x_idx].copy()
            
            elif method == 'interpolate':
                # 雙線性插值
                y_f = np.clip(y * H_pts / H, 0, H_pts - 1)
                x_f = np.clip(x * W_pts / W, 0, W_pts - 1)
                
                y0, x0 = int(y_f), int(x_f)
                y1 = min(y0 + 1, H_pts - 1)
                x1 = min(x0 + 1, W_pts - 1)
                
                dy, dx = y_f - y0, x_f - x0
                
                p00 = pts3d[y0, x0]
                p01 = pts3d[y0, x1]
                p10 = pts3d[y1, x0]
                p11 = pts3d[y1, x1]
                
                result = (p00 * (1 - dx) * (1 - dy) +
                         p01 * dx * (1 - dy) +
                         p10 * (1 - dx) * dy +
                         p11 * dx * dy)
                return result
        
        return None
    
    def transform_point_3d(self, template_path: str, target_path: str,
                           template_point: Tuple[float, float],
                           polygon: List[List[float]] = None,
                           matching_conf_thr: float = 0.1,
                           niter1: int = 300, niter2: int = 0) -> Dict[str, Any]:
        """
        將 template 上的像素點轉換到 target 圖片座標
        
        這是主要的 API 函數，整合了所有步驟。
        
        Args:
            template_path: template 影像路徑
            target_path: target 影像路徑
            template_point: template 上的標記點像素座標 (x, y)，原始圖片座標
            polygon: 可選的 ROI 多邊形（用於計算區域質心的 3D 位置）
            matching_conf_thr: 匹配置信度閾值
            niter1: 粗對齊迭代次數
            niter2: 精細化迭代次數
        
        Returns:
            dict: {
                'target_pixel': (x, y),       # 在 target 圖片上的像素位置（原始座標）
                'target_pixel_res': (x, y),   # 在 target 圖片上的像素位置（縮放後座標）
                'point_3d_world': (x, y, z),  # 3D 世界座標
                'point_3d_cam2': (x, y, z),   # 在 Camera2 座標系的位置
                'T_rel': np.ndarray,          # 相對姿態矩陣 4x4 (cam1 to cam2)
                'focals': (f1, f2),           # 估計的焦距
                'poses': np.ndarray,          # 相機姿態 [2, 4, 4]
                'success': bool,
                'message': str,
            }
        """
        result = {
            'target_pixel': None,
            'target_pixel_res': None,
            'point_3d_world': None,
            'point_3d_cam2': None,
            'T_rel': None,
            'focals': None,
            'poses': None,
            'success': False,
            'message': '',
        }
        
        if not MAST3R_AVAILABLE:
            result['message'] = "MASt3R is not available"
            return result
        
        try:
            # 1. 獲取對齊的場景數據
            scene_data = self.get_aligned_scene(
                template_path, target_path,
                matching_conf_thr=matching_conf_thr,
                niter1=niter1, niter2=niter2
            )
            
            poses = scene_data['poses']
            focals = scene_data['focals']
            pps = scene_data['pps']
            pts3d = scene_data['pts3d']
            res_shapes = scene_data['res_shapes']
            orig_shapes = scene_data['orig_shapes']
            
            result['poses'] = poses
            result['focals'] = tuple(focals)
            
            # 2. 計算相對姿態 T_rel = inv(cam2w_2) @ cam2w_1
            # T_rel 將 Camera 1 座標系的點轉換到 Camera 2 座標系
            cam2w_1 = poses[0]  # Template camera-to-world
            cam2w_2 = poses[1]  # Target camera-to-world
            w2cam_2 = np.linalg.inv(cam2w_2)  # world-to-camera2
            T_rel = w2cam_2 @ cam2w_1
            result['T_rel'] = T_rel
            
            # 3. 將原始像素座標縮放到 res 空間
            orig_H1, orig_W1 = orig_shapes[0]
            res_H1, res_W1 = res_shapes[0]
            scale_x1 = res_W1 / orig_W1
            scale_y1 = res_H1 / orig_H1
            
            template_point_res = (
                template_point[0] * scale_x1,
                template_point[1] * scale_y1
            )
            
            # 4. 獲取標記點的 3D 位置
            point_3d_world = None
            
            if polygon is not None:
                # 如果提供了多邊形，計算多邊形區域的 3D 質心
                point_3d_world = self._get_polygon_centroid_3d(
                    pts3d[0], polygon, orig_shapes[0], res_shapes[0]
                )
            
            if point_3d_world is None:
                # 使用標記點位置的 3D 座標
                point_3d_world = self.get_3d_at_pixel(
                    pts3d[0], template_point_res, res_shapes[0]
                )
            
            if point_3d_world is None:
                result['message'] = "Failed to get 3D position for template point"
                return result
            
            result['point_3d_world'] = tuple(point_3d_world)
            
            # 5. 將世界座標轉換到 Camera 2 座標系
            point_3d_world_hom = np.append(point_3d_world, 1.0)
            point_3d_cam2 = w2cam_2 @ point_3d_world_hom
            point_3d_cam2 = point_3d_cam2[:3]
            result['point_3d_cam2'] = tuple(point_3d_cam2)
            
            # 6. 投影到 target 圖片
            focal2 = focals[1]
            pp2 = pps[1]  # Principal point
            res_H2, res_W2 = res_shapes[1]
            orig_H2, orig_W2 = orig_shapes[1]
            
            z = point_3d_cam2[2]
            if z <= 0:
                result['message'] = f"Invalid depth: {z}"
                return result
            
            # 縮放座標系中的投影
            # 使用 principal point 作為中心（如果可用），否則用圖片中心
            if pp2 is not None and len(pp2) == 2:
                cx2, cy2 = pp2
            else:
                cx2 = res_W2 / 2.0
                cy2 = res_H2 / 2.0
            
            u_res = focal2 * (point_3d_cam2[0] / z) + cx2
            v_res = focal2 * (point_3d_cam2[1] / z) + cy2
            
            result['target_pixel_res'] = (float(u_res), float(v_res))
            
            # 7. 轉換到原始圖片座標
            scale_x2 = orig_W2 / res_W2
            scale_y2 = orig_H2 / res_H2
            
            u_orig = u_res * scale_x2
            v_orig = v_res * scale_y2
            
            result['target_pixel'] = (float(u_orig), float(v_orig))
            result['success'] = True
            result['message'] = "Transform successful"
            
            return result
            
        except Exception as e:
            result['message'] = f"Error: {str(e)}"
            import traceback
            traceback.print_exc()
            return result
    
    def _get_polygon_centroid_3d(self, pts3d: np.ndarray, 
                                  polygon: List[List[float]],
                                  orig_shape: Tuple[int, int],
                                  res_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        計算多邊形區域的 3D 質心
        
        Args:
            pts3d: 3D 點雲
            polygon: 多邊形頂點（原始圖片座標）
            orig_shape: 原始圖片形狀 (H, W)
            res_shape: 縮放後形狀 (H, W)
        
        Returns:
            np.ndarray: 3D 質心座標，或 None
        """
        try:
            from shapely.geometry import Polygon, Point
        except ImportError:
            print("Shapely not available, falling back to point method")
            return None
        
        # 縮放多邊形
        orig_H, orig_W = orig_shape
        res_H, res_W = res_shape
        scale_x = res_W / orig_W
        scale_y = res_H / orig_H
        
        poly_res = [(p[0] * scale_x, p[1] * scale_y) for p in polygon]
        shapely_poly = Polygon(poly_res)
        
        # 檢查 pts3d 形狀
        if pts3d.ndim == 2 and pts3d.shape[1] == 3:
            # 稀疏點，無法直接按像素位置過濾
            # 返回所有點的平均值（fallback）
            return pts3d.mean(axis=0) if len(pts3d) > 0 else None
        
        elif pts3d.ndim == 3:
            # 密集點 [H, W, 3]
            H_pts, W_pts, _ = pts3d.shape
            
            # 建立像素網格
            yy, xx = np.mgrid[:H_pts, :W_pts]
            points_2d = np.stack([xx.ravel(), yy.ravel()], axis=-1)
            
            # 縮放到 res 空間
            points_2d_scaled = points_2d.astype(float)
            points_2d_scaled[:, 0] *= res_W / W_pts
            points_2d_scaled[:, 1] *= res_H / H_pts
            
            # 找出多邊形內的點
            mask = np.array([shapely_poly.contains(Point(p)) for p in points_2d_scaled])
            
            if mask.sum() == 0:
                return None
            
            pts3d_flat = pts3d.reshape(-1, 3)
            pts3d_in_poly = pts3d_flat[mask]
            
            return pts3d_in_poly.mean(axis=0)
        
        return None
    
    def _transform_point_with_scene(self, scene_data: Dict[str, Any], 
                                    template_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        使用已計算的場景數據轉換單個點（使用單應性矩陣）
        
        Args:
            scene_data: 由 get_aligned_scene 返回的場景數據
            template_point: template 上的像素座標 (x, y)，原始圖片座標
        
        Returns:
            (u_orig, v_orig): target 上的像素座標，或 None
        """
        scene = scene_data['scene']
        res_shapes = scene_data['res_shapes']
        orig_shapes = scene_data['orig_shapes']
        
        # 將原始像素座標縮放到 res 空間
        orig_H1, orig_W1 = orig_shapes[0]
        res_H1, res_W1 = res_shapes[0]
        scale_x1 = res_W1 / orig_W1
        scale_y1 = res_H1 / orig_H1
        
        template_point_res = np.array([
            template_point[0] * scale_x1,
            template_point[1] * scale_y1
        ])
        
        try:
            # 使用快取的單應性矩陣（如果存在）
            if '_homography' not in scene_data:
                # 使用預先提取的 pts2d
                pts2d_1 = scene_data.get('pts2d_1')
                pts2d_2 = scene_data.get('pts2d_2')
                
                if pts2d_1 is None or pts2d_2 is None or len(pts2d_1) < 4:
                    scene_data['_homography'] = None
                else:
                    # 計算單應性矩陣
                    H, mask = cv2.findHomography(pts2d_1, pts2d_2, cv2.RANSAC, 5.0)
                    if H is not None:
                        scene_data['_homography'] = H
                        print(f"[DEBUG] Homography computed with {len(pts2d_1)} points")
                    else:
                        scene_data['_homography'] = None
            
            H = scene_data.get('_homography')
            if H is None:
                return self._transform_point_with_3d_projection(scene_data, template_point)
            
            # 使用單應性矩陣轉換點
            pt_src = np.array([[template_point_res]], dtype=np.float32)
            pt_dst = cv2.perspectiveTransform(pt_src, H)
            target_point_res = pt_dst[0, 0]
            
            # 轉換到原始圖片座標
            orig_H2, orig_W2 = orig_shapes[1]
            res_H2, res_W2 = res_shapes[1]
            scale_x2 = orig_W2 / res_W2
            scale_y2 = orig_H2 / res_H2
            
            u_orig = target_point_res[0] * scale_x2
            v_orig = target_point_res[1] * scale_y2
            
            return (float(u_orig), float(v_orig))
            
        except Exception as e:
            print(f"[DEBUG] _transform_point_with_scene error: {e}")
            # Fallback: 使用 3D 投影方法
            return self._transform_point_with_3d_projection(scene_data, template_point)
    
    def _transform_point_with_3d_projection(self, scene_data: Dict[str, Any], 
                                            template_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        使用 3D 投影方法轉換點（fallback）
        """
        poses = scene_data['poses']
        focals = scene_data['focals']
        pps = scene_data['pps']
        pts3d = scene_data['pts3d']
        res_shapes = scene_data['res_shapes']
        orig_shapes = scene_data['orig_shapes']
        
        # 計算相對姿態
        cam2w_2 = poses[1]
        w2cam_2 = np.linalg.inv(cam2w_2)
        
        # 將原始像素座標縮放到 res 空間
        orig_H1, orig_W1 = orig_shapes[0]
        res_H1, res_W1 = res_shapes[0]
        scale_x1 = res_W1 / orig_W1
        scale_y1 = res_H1 / orig_H1
        
        template_point_res = (
            template_point[0] * scale_x1,
            template_point[1] * scale_y1
        )
        
        # 獲取 3D 位置
        point_3d_world = self.get_3d_at_pixel(pts3d[0], template_point_res, res_shapes[0])
        if point_3d_world is None:
            return None
        
        # 轉換到 Camera 2 座標系
        point_3d_world_hom = np.append(point_3d_world, 1.0)
        point_3d_cam2 = w2cam_2 @ point_3d_world_hom
        point_3d_cam2 = point_3d_cam2[:3]
        
        # 投影到 target 圖片
        focal2 = focals[1]
        pp2 = pps[1]
        res_H2, res_W2 = res_shapes[1]
        orig_H2, orig_W2 = orig_shapes[1]
        
        z = point_3d_cam2[2]
        if z <= 0:
            return None
        
        if pp2 is not None and len(pp2) == 2:
            cx2, cy2 = pp2
        else:
            cx2 = res_W2 / 2.0
            cy2 = res_H2 / 2.0
        
        u_res = focal2 * (point_3d_cam2[0] / z) + cx2
        v_res = focal2 * (point_3d_cam2[1] / z) + cy2
        
        # 轉換到原始圖片座標
        scale_x2 = orig_W2 / res_W2
        scale_y2 = orig_H2 / res_H2
        
        u_orig = u_res * scale_x2
        v_orig = v_res * scale_y2
        
        return (float(u_orig), float(v_orig))
    
    def visualize_transform(self, template_path: str, target_path: str,
                           template_point: Tuple[float, float],
                           polygon: List[List[float]] = None,
                           output_path: str = None) -> np.ndarray:
        """
        視覺化轉換結果
        
        Args:
            template_path: template 圖片路徑
            target_path: target 圖片路徑
            template_point: template 上的標記點
            polygon: 可選的 ROI 多邊形
            output_path: 輸出路徑（可選）
        
        Returns:
            np.ndarray: 視覺化結果圖片
        """
        # 先獲取場景數據（只計算一次）
        scene_data = self.get_aligned_scene(template_path, target_path)
        
        img1 = cv2.imread(template_path)
        img2 = cv2.imread(target_path)
        
        # 轉換質心點
        target_pixel = self._transform_point_with_scene(scene_data, template_point)
        print(f"[DEBUG] template_point: {template_point} -> target_pixel: {target_pixel}")
        
        # 在 template 上畫標記點
        pt1 = (int(template_point[0]), int(template_point[1]))
        cv2.circle(img1, pt1, 15, (0, 255, 0), -1)
        cv2.circle(img1, pt1, 15, (255, 255, 255), 3)
        
        # 如果有多邊形，畫多邊形
        if polygon is not None:
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img1, [pts], True, (0, 255, 0), 3)
            print(f"[DEBUG] polygon has {len(polygon)} vertices")
        
        # 在 target 上畫轉換後的點
        if target_pixel is not None:
            pt2 = (int(target_pixel[0]), int(target_pixel[1]))
            cv2.circle(img2, pt2, 15, (255, 0, 0), -1)
            cv2.circle(img2, pt2, 15, (255, 255, 255), 3)
            
            # 如果有多邊形，將每個頂點也轉換到 target 並畫出（重用 scene_data，不需重新計算）
            if polygon is not None:
                target_polygon = []
                for i, vertex in enumerate(polygon):
                    vertex_target = self._transform_point_with_scene(
                        scene_data, (vertex[0], vertex[1])
                    )
                    print(f"[DEBUG] vertex {i}: {vertex} -> {vertex_target}")
                    if vertex_target is not None:
                        target_polygon.append([
                            int(vertex_target[0]),
                            int(vertex_target[1])
                        ])
                
                print(f"[DEBUG] target_polygon has {len(target_polygon)} vertices")
                # 畫出轉換後的多邊形
                if len(target_polygon) >= 3:
                    pts_target = np.array(target_polygon, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img2, [pts_target], True, (255, 0, 0), 3)
                    print(f"[DEBUG] Drew polygon on target")
                else:
                    print(f"[DEBUG] Not enough vertices to draw polygon")
        
        # 合併圖片
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        max_h = max(h1, h2)
        
        canvas1 = np.zeros((max_h, w1, 3), dtype=np.uint8)
        canvas2 = np.zeros((max_h, w2, 3), dtype=np.uint8)
        canvas1[:h1, :w1] = img1
        canvas2[:h2, :w2] = img2
        
        combined = np.hstack([canvas1, canvas2])
        
        if output_path:
            cv2.imwrite(output_path, combined)
            print(f"Visualization saved to {output_path}")
        
        return combined


def compute_relative_pose(poses: np.ndarray) -> np.ndarray:
    """
    計算兩個相機之間的相對姿態
    
    Args:
        poses: [2, 4, 4] camera-to-world 姿態矩陣
    
    Returns:
        T_rel: 4x4 相對姿態矩陣（Camera 1 到 Camera 2）
    """
    cam2w_1 = poses[0]
    cam2w_2 = poses[1]
    w2cam_2 = np.linalg.inv(cam2w_2)
    T_rel = w2cam_2 @ cam2w_1
    return T_rel


def project_to_image(point_3d: np.ndarray, focal: float, 
                     principal_point: Tuple[float, float]) -> Tuple[float, float]:
    """
    將 3D 點投影到圖片座標
    
    Args:
        point_3d: 相機座標系中的 3D 點 [x, y, z]
        focal: 焦距
        principal_point: 主點 (cx, cy)
    
    Returns:
        (u, v): 像素座標
    """
    cx, cy = principal_point
    x, y, z = point_3d
    
    if z <= 0:
        raise ValueError(f"Invalid depth: {z}")
    
    u = focal * x / z + cx
    v = focal * y / z + cy
    return (u, v)


if __name__ == '__main__':
    # 簡單測試
    import argparse
    
    parser = argparse.ArgumentParser(description='Test 3D Template Matcher')
    parser.add_argument('--template', required=True, help='Template image path')
    parser.add_argument('--target', required=True, help='Target image path')
    parser.add_argument('--point', type=float, nargs=2, required=True, 
                       help='Template point (x, y)')
    parser.add_argument('--output', default='test_3d_transform.png', 
                       help='Output visualization path')
    args = parser.parse_args()
    
    matcher = TemplateMatcher3D()
    result = matcher.transform_point_3d(
        args.template, args.target, tuple(args.point)
    )
    
    print("\n=== 轉換結果 ===")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    if result['success']:
        print(f"Target pixel: {result['target_pixel']}")
        print(f"3D world: {result['point_3d_world']}")
        print(f"3D cam2: {result['point_3d_cam2']}")
        print(f"Focals: {result['focals']}")
        print(f"T_rel:\n{result['T_rel']}")
    
    # 視覺化
    matcher.visualize_transform(
        args.template, args.target, tuple(args.point), output_path=args.output
    )
