#!/usr/bin/env python3
"""測試 match_and_transform 功能並視覺化結果"""

import sys
sys.path.insert(0, '/Users/po-hong/Documents/project3')

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from template_system.matcher import TemplateMatcher
from template_system.manager import TemplateManager


def visualize_keypoints(template_path: str, target_path: str,
                        original_keypoints: dict, transformed_keypoints: dict,
                        output_path: str = 'transform_test_result.png'):
    """
    視覺化原始與轉換後的 keypoints
    
    Args:
        template_path: template 圖片路徑
        target_path: target 圖片路徑
        original_keypoints: 原始 keypoints {name: [x, y]}
        transformed_keypoints: 轉換後 keypoints {name: [x, y]}
        output_path: 輸出圖片路徑
    """
    # 讀取圖片
    template_img = cv2.imread(template_path)
    target_img = cv2.imread(target_path)
    
    if template_img is None:
        print(f"無法讀取 template 圖片: {template_path}")
        return
    if target_img is None:
        print(f"無法讀取 target 圖片: {target_path}")
        return
    
    # BGR 轉 RGB（matplotlib 使用 RGB）
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    
    # 在 template 圖上畫原始 keypoints（綠色）
    template_draw = template_img.copy()
    for name, coord in original_keypoints.items():
        x, y = int(coord[0]), int(coord[1])
        cv2.circle(template_draw, (x, y), 10, (0, 255, 0), -1)  # 綠色實心圓
        cv2.circle(template_draw, (x, y), 10, (0, 0, 0), 2)     # 黑色邊框
    
    # 在 target 圖上畫轉換後 keypoints（紅色）
    target_draw = target_img.copy()
    for name, coord in transformed_keypoints.items():
        x, y = int(coord[0]), int(coord[1])
        cv2.circle(target_draw, (x, y), 10, (255, 0, 0), -1)   # 紅色實心圓
        cv2.circle(target_draw, (x, y), 10, (0, 0, 0), 2)      # 黑色邊框
    
    # 建立並排顯示的 figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # 左圖：Template + 原始 keypoints
    axes[0].imshow(template_draw)
    axes[0].set_title('Template - Original Keypoints (Green)', fontsize=12)
    axes[0].axis('off')
    
    # 在左圖標註點名稱
    for name, coord in original_keypoints.items():
        axes[0].annotate(name, (coord[0], coord[1]),
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=10, color='green', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # 右圖：Target + 轉換後 keypoints
    axes[1].imshow(target_draw)
    axes[1].set_title('Target - Transformed Keypoints (Red)', fontsize=12)
    axes[1].axis('off')
    
    # 在右圖標註點名稱
    for name, coord in transformed_keypoints.items():
        axes[1].annotate(name, (coord[0], coord[1]),
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=10, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"視覺化結果已儲存至: {output_path}")


def main():
    manager = TemplateManager(root_dir=Path('templates'))
    matcher = TemplateMatcher(manager=manager)
    
    # 測試用的原始 keypoints
    original_keypoints = {'center': [376, 1065], 'corner': [100, 100]}
    
    # 測試：dog template 自我匹配
    result = matcher.match_and_transform(
        rule='a_lab',
        key='room',
        target_image_path='/Users/po-hong/Documents/project3/mast3r-research/assets/NLE_tower/FF5599FD-768B-431A-AB83-BDA5FB44CB9D-83120-000041DADDE35483.jpg',
        keypoints=original_keypoints,
        transform_method='homography'
    )
    
    print(f"最佳模板: {result['best_template']}")
    print(f"匹配分數: {result['match_score']:.3f}")
    print(f"轉換後座標:")
    for name, coord in result['transformed_keypoints'].items():
        print(f"  {name}: {coord}")
    print(f"重投影誤差: {result['reprojection_error']}")
    
    # 視覺化結果
    template_path = str(Path('templates/a_lab/room') / result['best_template'])
    target_path = '/Users/po-hong/Documents/project3/mast3r-research/assets/NLE_tower/FF5599FD-768B-431A-AB83-BDA5FB44CB9D-83120-000041DADDE35483.jpg'

    visualize_keypoints(
        template_path=template_path,
        target_path=target_path,
        original_keypoints=original_keypoints,
        transformed_keypoints=result['transformed_keypoints'],
        output_path='transform_test_result.png'
    )


if __name__ == '__main__':
    main()
