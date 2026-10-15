# 1단계: Camera 9 시점 확인용 프레임 추출 (개입·통행 클립 첫 프레임 몇 장)
import json, random
import cv2, numpy as np
from huggingface_hub import hf_hub_download
REPO = "Voxel51/Safe_and_Unsafe_Behaviours"
samples = json.load(open(hf_hub_download(REPO, "samples.json", repo_type="dataset")))
samples = samples["samples"] if isinstance(samples, dict) else samples
print("keys:", list(samples[0].keys()))
print("example:", {k: samples[0][k] for k in samples[0] if k != "ground_truth"})
random.seed(0)
tiles = []
for lab in ["Unauthorized Intervention", "Authorized Intervention", "Safe Walkway", "Safe Walkway Violation"]:
    pool = [s for s in samples if s["ground_truth"]["label"] == lab]
    for s in random.sample(pool, 2):
        cap = cv2.VideoCapture(hf_hub_download(REPO, s["filepath"], repo_type="dataset"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, img = cap.read(); cap.release()
        img = cv2.resize(img, (960, 540))
        for x in range(0, 960, 96): cv2.line(img, (x, 0), (x, 540), (0, 255, 255), 1); cv2.putText(img, str(x*2), (x+2, 14), 0, 0.4, (0, 255, 255), 1)
        for y in range(0, 540, 54): cv2.line(img, (0, y), (960, y), (0, 255, 255), 1); cv2.putText(img, str(y*2), (2, y+12), 0, 0.4, (0, 255, 255), 1)
        cv2.putText(img, f"{lab} | {s['filepath'].split('/')[-1]}", (10, 530), 0, 0.6, (255, 255, 255), 2)
        tiles.append(img)
sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, 8, 2)])
cv2.imwrite("step1_sheet.jpg", sheet)
print("saved")
