"""
MASt3R Pairwise Feature Matching Test
=====================================
基本特徵匹配測試腳本，用於驗證 MASt3R 模型是否正確載入和運行。
"""

import matplotlib
matplotlib.use('Agg')  # 無 GUI 後端，適合 Mac
import matplotlib.pyplot as pl
import numpy as np
import torch
import sys
from pathlib import Path

# Add mast3r-research to Python path
MAST3R_PATH = Path(__file__).parent.parent / 'mast3r-research'
sys.path.insert(0, str(MAST3R_PATH))

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images

device = 'mps'
model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
print("載入模型中...")
model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

# 使用 repo assets 兩張圖作為 template & current
# 路徑相對於 mast3r-research 目錄
img_paths = [str(MAST3R_PATH / 'assets/NLE_tower/1AD85EF5-B651-4291-A5C0-7BDB7D966384-83120-000041DADF639E09.jpg'),
             str(MAST3R_PATH / 'assets/NLE_tower/01D90321-69C8-439F-B0B0-E87E7634741C-83120-000041DAE419D7AE.jpg')]
images = load_images(img_paths, size=512)
output = inference([tuple(images)], model, device, batch_size=1, verbose=True)

view1, pred1 = output['view1'], output['pred1']
view2, pred2 = output['view2'], output['pred2']

desc1 = pred1['desc'].squeeze(0).detach()
desc2 = pred2['desc'].squeeze(0).detach()

matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                               device=device, dist='dot', block_size=2**13)

# 確保是 numpy array
if hasattr(matches_im0, 'cpu'):
    matches_im0 = matches_im0.cpu().numpy()
if hasattr(matches_im1, 'cpu'):
    matches_im1 = matches_im1.cpu().numpy()

matches_im0 = np.asarray(matches_im0, dtype=np.float32)
matches_im1 = np.asarray(matches_im1, dtype=np.float32)

# 取得圖像尺寸
H0, W0 = int(view1['true_shape'][0][0]), int(view1['true_shape'][0][1])
H1, W1 = int(view2['true_shape'][0][0]), int(view2['true_shape'][0][1])

# 濾邊緣
valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < W0 - 3) & \
                    (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < H0 - 3)
valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < W1 - 3) & \
                    (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < H1 - 3)
valid_matches = valid_matches_im0 & valid_matches_im1
matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

print(f'找到 {len(matches_im0)} 個匹配點')

# 可視化前 20 個 matches
n_viz = min(20, len(matches_im0))
if n_viz > 1:
    match_idx_to_viz = np.round(np.linspace(0, len(matches_im0) - 1, n_viz)).astype(int)
else:
    match_idx_to_viz = np.array([0])
viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

image_mean = torch.as_tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
image_std = torch.as_tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
viz_imgs = []
for view in [view1, view2]:
    rgb_tensor = view['img'] * image_std + image_mean
    viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

# 正確提取圖像尺寸
h0, w0 = viz_imgs[0].shape[:2]
h1, w1 = viz_imgs[1].shape[:2]

img0 = np.pad(viz_imgs[0], ((0, max(h1 - h0, 0)), (0, 0), (0, 0)), 'constant')
img1 = np.pad(viz_imgs[1], ((0, max(h0 - h1, 0)), (0, 0), (0, 0)), 'constant')
img = np.concatenate((img0, img1), axis=1)

pl.figure(figsize=(14, 7))
pl.imshow(np.clip(img, 0, 1))
cmap = pl.get_cmap('jet')
for i in range(n_viz):
    x0, y0 = viz_matches_im0[i][0], viz_matches_im0[i][1]
    x1, y1 = viz_matches_im1[i][0], viz_matches_im1[i][1]
    pl.plot([x0, x1 + w0], [y0, y1], '-+', color=cmap(i / max(n_viz - 1, 1)), 
            scalex=False, scaley=False, markersize=8, linewidth=1.5)
pl.title(f'MASt3R Pairwise Matching: {len(matches_im0)} matches (showing {n_viz})')
pl.axis('off')
pl.tight_layout()
pl.savefig('matches.png', dpi=150, bbox_inches='tight')
print('已儲存視覺化結果至 matches.png')

# 相對 pose 資訊 (pts3d from pred)
pts3d_1 = pred1['pts3d'].squeeze(0)
print('Relative 3D points shape:', pts3d_1.shape)
print('Sample pts3d (corner):', pts3d_1[0, 0].cpu().numpy())
print('Sample pts3d (center):', pts3d_1[pts3d_1.shape[0]//2, pts3d_1.shape[1]//2].cpu().numpy())
