# 4단계: 통행 클립에서 규칙이 잘못 반응한 이유를 눈으로 확인 (손목이 구역 안인 사람을 빨간 점으로 표시)
import csv, cv2, numpy as np
from huggingface_hub import hf_hub_download
from rtmlib import Body
from zones import REPO, ZONES, view_of
rows = [r for r in csv.DictReader(open("step3_clips.csv", encoding="utf-8")) if r["group"] == "walking"]
rows = sorted(rows, key=lambda r: -int(r["zone_frames"]))[:6]
body = Body(mode="lightweight", backend="onnxruntime", device="cpu")
tiles = []
for r in rows:
    cap = cv2.VideoCapture(hf_hub_download(REPO, "data/" + r["clip"], repo_type="dataset"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2); ok, img = cap.read(); cap.release()
    zone = ZONES[view_of(img)]; kps, sc = body(img)
    cv2.polylines(img, [zone], True, (0, 0, 255), 3)
    for k, s in zip(kps, sc):
        for j in (9, 10):
            if s[j] >= 0.3:
                inside = cv2.pointPolygonTest(zone, (float(k[j][0]), float(k[j][1])), False) >= 0
                cv2.circle(img, (int(k[j][0]), int(k[j][1])), 14, (0, 0, 255) if inside else (0, 255, 0), -1)
    cv2.putText(img, f"{r['label']} {r['clip']} zone={r['zone_frames']}/12", (20, 1050), 0, 1.5, (255, 255, 255), 4)
    tiles.append(cv2.resize(img, (960, 540)))
cv2.imwrite("step4_fp.jpg", np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, 6, 2)]))
