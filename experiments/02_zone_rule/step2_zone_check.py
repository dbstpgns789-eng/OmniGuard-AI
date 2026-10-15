# 2단계: 자동 시점 판별 + 구역 다각형이 프레스에 맞게 그려졌는지 눈으로 확인
import cv2, numpy as np
from zones import ZONES, mid_frame, view_of
CLIPS = ["data/1_tr54.mp4", "data/1_te1.mp4", "data/5_te12.mp4", "data/5_tr11.mp4",
         "data/4_tr4.mp4", "data/4_te23.mp4", "data/0_tr50.mp4", "data/0_tr31.mp4"]
tiles = []
for fp in CLIPS:
    img = mid_frame(fp); v = view_of(img)
    ov = img.copy(); cv2.fillPoly(ov, [ZONES[v]], (0, 0, 255)); img = cv2.addWeighted(ov, 0.3, img, 0.7, 0)
    cv2.polylines(img, [ZONES[v]], True, (0, 0, 255), 4)
    cv2.putText(img, f"view {v} | {fp}", (20, 1050), 0, 1.6, (255, 255, 255), 4)
    tiles.append(cv2.resize(img, (960, 540)))
cv2.imwrite("step2_zones.jpg", np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, 8, 2)]))
print("saved")
