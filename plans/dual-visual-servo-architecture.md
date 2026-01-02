# 雙方案視覺伺服架構計劃 (YOLO+PnP & MASt3R，無手眼標定)

## 專案概述
- **目標**：模組化閉環置中 (X/Y/Rz 對齊)，迭代收斂優化 (痛點：收斂慢 → 動態 gain + 預測)。
- **技術**：Python 3.10+ PyTorch 2.0+ OpenCV 4.8+ UR5 SDK (模擬 stub)。
- **模擬策略** (無實手臂)：合成偏移影像序列 + 虛擬 TCP 追蹤，驗證 <10 步收斂。
- **共用率**：80% (僅 Estimator 差異)。

## 檔案結構
```
dual_visual_servo/
├── src/
│   ├── __init__.py
│   ├── camera.py          # 模擬/真相機
│   ├── arm.py             # UR5 stub/sim
│   ├── estimator.py       # PoseEstimator (switch mode)
│   ├── controller.py      # ControlLoop
│   └── utils.py           # 矩陣轉換、log
├── sim/
│   ├── template.jpg       # 完美置中參考
│   ├── offset_images/     # 合成測試影像 (tx=2cm, ty=1cm 等)
│   └── generate_sim.py    # 產生偏移影像
├── tests/
│   └── test_convergence.py # pytest 驗證收斂
├── config.yaml            # mode: mast3r/yolo, gain:0.7, thresh:0.5cm
├── requirements.txt
├── README.md
└── main.py                # python main.py --mode mast3r --sim
```

## 系統流程 (Mermaid)
```mermaid
graph TD
    A[啟動: load config/template] --> B[Capture img<br/>sim: offset_images/next]
    B --> C{PoseEstimator<br/>if yolo: keypoints → PnP<br/>if mast3r: img+template → rel_pose}
    C --> D[delta_pose = -est_pose * gain<br/>動態 gain: if slow → 0.5-0.9]
    D --> E[Arm sim: tcp += delta<br/>log 偏差]
    E --> F{||tx,ty|<0.5cm & |Rz|<1°?}
    F -->|否| B
    F -->|是| G[收斂成功<br/>步數/總誤差]
```

## 關鍵優化 (收斂慢痛點)
1. **動態 gain**：初始0.7，若 3步無改善 → 降0.5；改善 → 升0.8。
2. **自由度解耦**：獨立 PID for tx/ty/Rz (Z凍結)。
3. **預濾**：Kalman 濾 pose 噪聲。
4. **模擬驗證**：產生 100 隨機初始偏移，平均收斂步 <8。

## 依賴
```
torch torchvision
ultralytics  # YOLO
opencv-python
pyyaml
numpy scipy
urx  # UR5 Python SDK (stub 版)
pytest
```

## 下步驟
- 創建結構 & sim 影像。
- 實作 estimator (MASt3R 先)。
- 測試收斂，調 gain。

此計劃聚焦模擬驗證，後接真 UR5。預估高泛化/快收斂。