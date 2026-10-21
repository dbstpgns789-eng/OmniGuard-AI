# 실험 22 준비: 시험 영상 20개에 YOLO11n-seg 사람 윤곽 (프레임은 실험 21 포즈와 같은 구간·순서)
# 저장: OmniGuard-data/exp22/<영상>_seg.npz (t, masks=프레임마다 사람 윤곽 다각형 픽셀 좌표). 이미 있으면 건너뜀
import os, sys, numpy as np, cv2
from ultralytics import YOLO
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "21_first_test", "prepare.py"), encoding="utf-8").read(); exec(src[src.index("DATA ="):src.index("orb, bf")])
OUT = DATA + "exp22/"; os.makedirs(OUT, exist_ok=True); model = YOLO("yolo11n-seg.pt")
for name, path, t0, t1 in CLIPS:
    dst = OUT + f"{name}_seg.npz"
    if os.path.exists(dst): continue
    P = np.load(DATA + f"exp21/{name}_pose.npz", allow_pickle=True); cap = cv2.VideoCapture(path); fps = cap.get(5)
    f0 = int(round(t0 * fps)); cap.set(1, f0); T, M = [], []
    for f in range(f0, f0 + len(P["t"])):
        ok, img = cap.read()
        if not ok: break
        r = model.predict(img, classes=[0], conf=0.25, verbose=False, device="cpu")[0]
        T.append(f / fps); M.append(np.array([] if r.masks is None else [p.astype(np.float32) for p in r.masks.xy], dtype=object))
    np.savez(dst, t=np.array(T), masks=np.array(M, dtype=object)); print(name, len(T), flush=True)
print("done", flush=True)
