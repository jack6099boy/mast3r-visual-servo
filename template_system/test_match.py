#!/usr/bin/env python3
"""
Template Matching Test Script
=============================

測試使用 a_lab/dog/front template 來匹配目標影像，並視覺化結果。

使用方式:
    python3 -m template_system.test_match --rule a_lab --key dog --target test_dog.jpeg
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json

import sys
import os

# 設置 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'mast3r-research'))

# 從 tools 模組導入核心函數
try:
    from tools import (
        load_mast3r_model,
        compute_matches,
        filter_matches_in_polygon,
        visualize_transfer,
        MAST3R_AVAILABLE,
    )
except ImportError as e:
    print(f"Tools import error: {e}. Using fallback ORB matching.")
    MAST3R_AVAILABLE = False
    load_mast3r_model = None
    compute_matches = None
    filter_matches_in_polygon = None
    visualize_transfer = None

# 嘗試匯入 torch (用於設備檢測)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .manager import TemplateManager, Template


class FallbackMatcher:
    """當 MASt3R 不可用時的備用匹配器，使用 ORB 特徵匹配"""

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)

    def match_images(self, template_img: Path, target_img: Path) -> Optional[np.ndarray]:
        """使用 ORB 進行特徵匹配，返回匹配點 [N, 4] (x1,y1,x2,y2)"""
        try:
            # 讀取影像
            img1 = cv2.imread(str(template_img))
            img2 = cv2.imread(str(target_img))

            if img1 is None or img2 is None:
                return None

            # 偵測特徵點和描述子
            kp1, des1 = self.orb.detectAndCompute(img1, None)
            kp2, des2 = self.orb.detectAndCompute(img2, None)

            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                return None

            # 特徵匹配
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # 排序並取前 50 個最佳匹配
            matches = sorted(matches, key=lambda x: x.distance)[:50]

            # 轉換為 [x1,y1,x2,y2] 格式
            match_points = []
            for match in matches:
                pt1 = kp1[match.queryIdx].pt
                pt2 = kp2[match.trainIdx].pt
                match_points.append([pt1[0], pt1[1], pt2[0], pt2[1]])

            return np.array(match_points) if match_points else None

        except Exception as e:
            print(f"Fallback matching error: {e}")
            return None


class TemplateMatcher:
    """模板匹配器，使用 tools/ 提供的核心函數"""

    def __init__(self, manager: TemplateManager):
        self.manager = manager
        self.fallback_matcher = FallbackMatcher()
        self._model = None
        
        # 設備選擇
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = 'cpu'

    @property
    def model(self):
        """延遲載入 MASt3R 模型"""
        if self._model is None and MAST3R_AVAILABLE and load_mast3r_model is not None:
            try:
                self._model = load_mast3r_model(device=self.device)
            except Exception as e:
                print(f"MASt3R model loading failed: {e}. Using fallback.")
                self._model = None
        return self._model

    def match(self, rule: str, key: str, target_image: Path) -> Dict[str, Any]:
        """匹配 rule/key 下的所有 templates 與 target_image"""
        templates = self.manager.load_templates(rule, key)
        if not templates:
            return {'best': None, 'all': []}

        results = []
        for temp in templates:
            matches = self._get_matches(temp.img_path, target_image)
            if matches is None:
                continue

            # 計算 ROI 內的匹配點數量，並保存過濾後的完整匹配點
            matches_in_roi = np.array([])
            if 'polygons' in temp.roi_data and temp.roi_data['polygons']:
                polygon = temp.roi_data['polygons'][0]['points']
                
                # 使用 tools 的 filter_matches_in_polygon
                if filter_matches_in_polygon is not None:
                    filtered_im0, filtered_im1 = filter_matches_in_polygon(
                        matches[:, :2], matches[:, 2:4], polygon
                    )
                    # 組合回 [x1, y1, x2, y2] 格式
                    matches_in_roi = np.column_stack([filtered_im0, filtered_im1]) if len(filtered_im0) > 0 else np.array([])
                else:
                    # Fallback: 手動過濾
                    polygon_np = np.array(polygon, dtype=np.float32)
                    mask = np.array([cv2.pointPolygonTest(polygon_np, (float(pt[0]), float(pt[1])), False) >= 0
                                     for pt in matches[:, :2]])
                    matches_in_roi = matches[mask]
                
                num_in_roi = len(matches_in_roi)
            else:
                num_in_roi = 0

            score = num_in_roi / max(1, len(matches))
            results.append({
                'template': temp.img_path.name,
                'template_path': temp.img_path,
                'score': float(score),
                'num_matches': len(matches),
                'num_in_roi': num_in_roi,
                'matches': matches.tolist() if matches is not None else [],
                'matches_in_roi': matches_in_roi.tolist() if len(matches_in_roi) > 0 else [],
                'roi_data': temp.roi_data
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        best = results[0] if results else None
        return {'best': best, 'all': results}

    def _get_matches(self, template_img: Path, target_img: Path) -> Optional[np.ndarray]:
        """取得匹配點"""
        if self.model is not None and MAST3R_AVAILABLE and compute_matches is not None:
            return self._mast3r_match(template_img, target_img)
        else:
            return self.fallback_matcher.match_images(template_img, target_img)

    def _mast3r_match(self, template_img: Path, target_img: Path) -> Optional[np.ndarray]:
        """使用 tools.compute_matches 進行匹配"""
        try:
            # 使用 tools 的 compute_matches 函數
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
            
            # 組合為 [N, 4] 格式 (x1, y1, x2, y2)
            return np.column_stack([matches_im0, matches_im1])
            
        except Exception as e:
            print(f"MASt3R matching error: {e}")
            return None


def visualize_matches(target_img_path: Path, match_result: Dict[str, Any], output_path: Path,
                      template_img_path: Optional[Path] = None):
    """視覺化匹配結果"""
    # 讀取目標影像
    target_img = cv2.imread(str(target_img_path))
    if target_img is None:
        print(f"無法讀取影像: {target_img_path}")
        return

    vis_img = target_img.copy()

    if match_result['best'] is None:
        print("沒有找到匹配結果")
        cv2.putText(vis_img, "No matches found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        best = match_result['best']

        # 繪製 ROI 多邊形轉移到 target 上 (若有足夠的匹配點)
        if 'roi_data' in best and 'polygons' in best['roi_data'] and len(best['matches_in_roi']) >= 3:
            polygon = best['roi_data']['polygons'][0]['points']
            matches_in_roi = np.array(best['matches_in_roi'])

            # 使用匹配點估計 Affine Transform
            src_pts = matches_in_roi[:, :2].astype(np.float32)
            dst_pts = matches_in_roi[:, 2:4].astype(np.float32)

            # 計算 Affine Transform
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

            if M is not None:
                # 轉換多邊形到 target
                polygon_np = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
                polygon_transformed = cv2.transform(polygon_np, M).reshape(-1, 2)

                # 繪製轉換後的多邊形
                pts = polygon_transformed.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], True, (0, 255, 0), 3)

        # 繪製 ROI 過濾後的匹配點（只顯示 ROI 內的 template 點對應的 target 位置）
        if 'matches_in_roi' in best and best['matches_in_roi']:
            for match in best['matches_in_roi']:
                x2, y2 = int(match[2]), int(match[3])  # ROI 內匹配點在 target 上的位置
                cv2.circle(vis_img, (x2, y2), 3, (255, 0, 0), -1)

        # 顯示資訊
        info_text = [
            f"Template: {best['template']}",
            f"Score: {best['score']:.3f}",
            f"Matches: {best['num_matches']}",
            f"In ROI: {best['num_in_roi']}",
            f"Matcher: {'MASt3R' if MAST3R_AVAILABLE else 'ORB'}"
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(vis_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y_offset += 30

    # 儲存結果
    cv2.imwrite(str(output_path), vis_img)
    print(f"視覺化結果已儲存到: {output_path}")
    
    # 若有 template 路徑且 tools 的 visualize_transfer 可用，生成更詳細的視覺化
    if template_img_path is not None and visualize_transfer is not None and match_result['best'] is not None:
        best = match_result['best']
        if 'roi_data' in best and 'polygons' in best['roi_data'] and len(best['matches_in_roi']) > 0:
            polygon = best['roi_data']['polygons'][0]['points']
            matches_in_roi = np.array(best['matches_in_roi'])

            # 計算轉移後的多邊形 (如果有足夠匹配點)
            polygon_dst = None
            if len(best['matches_in_roi']) >= 3:
                src_pts = matches_in_roi[:, :2].astype(np.float32)
                dst_pts = matches_in_roi[:, 2:4].astype(np.float32)
                M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
                if M is not None:
                    polygon_np = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
                    polygon_dst = cv2.transform(polygon_np, M).reshape(-1, 2)

            # 使用 tools 的 visualize_transfer 生成三面板視覺化
            detailed_output = str(output_path).replace('.png', '_detailed.png').replace('.jpg', '_detailed.jpg')
            visualize_transfer(
                img0_path=str(template_img_path),
                img1_path=str(target_img_path),
                polygon_src=np.array(polygon),
                polygon_dst=polygon_dst,
                matches_in_roi=(matches_in_roi[:, :2], matches_in_roi[:, 2:4]),
                target_point_src=None,
                target_point_dst=None,
                output_path=detailed_output,
                title_info=f"Score: {best['score']:.3f}, ROI matches: {best['num_in_roi']}"
            )
            print(f"詳細視覺化已儲存到: {detailed_output}")


def main():
    parser = argparse.ArgumentParser(description="Template Matching Test")
    parser.add_argument("--rule", required=True, help="Rule name (e.g., a_lab)")
    parser.add_argument("--key", required=True, help="Key name (e.g., dog)")
    parser.add_argument("--target", required=True, help="Target image path")
    parser.add_argument("--output", default="match_result.png", help="Output visualization path")

    args = parser.parse_args()

    # 初始化管理器和匹配器
    manager = TemplateManager()
    matcher = TemplateMatcher(manager)

    target_path = Path(args.target)
    output_path = Path(args.output)

    if not target_path.exists():
        print(f"目標影像不存在: {target_path}")
        return

    print(f"載入 templates: {args.rule}/{args.key}")
    print(f"目標影像: {target_path}")
    print(f"使用匹配器: {'MASt3R' if MAST3R_AVAILABLE else 'ORB (fallback)'}")

    # 執行匹配
    result = matcher.match(args.rule, args.key, target_path)

    # 輸出結果
    print("\n=== 匹配結果 ===")
    if result['best']:
        best = result['best']
        print(f"最佳模板: {best['template']}")
        print(f"匹配分數: {best['score']:.3f}")
        print(f"總匹配點數: {best['num_matches']}")
        print(f"ROI 內匹配點數: {best['num_in_roi']}")
        template_path = best.get('template_path')
    else:
        print("沒有找到匹配結果")
        template_path = None

    print(f"\n所有結果 ({len(result['all'])} 個模板):")
    for i, res in enumerate(result['all'][:5]):  # 只顯示前 5 個
        print(f"  {i+1}. {res['template']}: {res['score']:.3f}")

    # 視覺化
    print("\n生成視覺化結果...")
    visualize_matches(target_path, result, output_path, template_path)

    print(f"\n完成! 結果已儲存至 {output_path}")


if __name__ == "__main__":
    main()
