# 실험 16 준비: YOLO11n-seg(COCO 사전학습)로 프레임마다 사람 윤곽을 구해 저장한다. 프레임 순서는 포즈 파일(실험 11·12)과 같다
# 저장: OmniGuard-data/exp16/<영상>_seg.npz (t, 사람마다 윤곽 다각형 픽셀 좌표, 신뢰도). 이미 있으면 건너뜀
import os, sys, json, glob, numpy as np, cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = "C:/Users/User/Desktop/OmniGuard-data/"; OUT = DATA + "exp16/"; os.makedirs(OUT, exist_ok=True)
model = YOLO("yolo11n-seg.pt")
def run(name, cap, f0, n):
    path = OUT + f"{name}_seg.npz"
    if os.path.exists(path): return
    fps = cap.get(5); cap.set(1, f0); T, M, C = [], [], []
    for f in range(f0, f0 + n):
        ok, img = cap.read()
        if not ok: break
        r = model.predict(img, classes=[0], conf=0.25, verbose=False, device="cpu")[0]
        polys = [] if r.masks is None else [p.astype(np.float32) for p in r.masks.xy]
        T.append(f / fps); M.append(np.array(polys, dtype=object)); C.append(r.boxes.conf.numpy() if r.boxes is not None else np.zeros(0))
    np.savez(path, t=np.array(T), masks=np.array(M, dtype=object), conf=np.array(C, dtype=object)); print(name, len(T), flush=True)
C12 = json.load(open(os.path.join(HERE, "..", "12_accident_set", "clips.json"), encoding="utf-8"))
for k, c in C12.items():
    P = np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True); cap = cv2.VideoCapture(DATA + "raw_cctv/" + c["file"])
    run(k, cap, int(round(c["t0"] * cap.get(5))), len(P["t"]))
for p in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    name = os.path.basename(p)[:-9]; P = np.load(p, allow_pickle=True)
    src = [x for x in glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Voxel51--Safe_and_Unsafe_Behaviours/snapshots/*/data/*.mp4")) if os.path.basename(x)[:-4] == name]
    run(name, cv2.VideoCapture(src[0] if src else hf_hub_download("Voxel51/Safe_and_Unsafe_Behaviours", f"data/{name}.mp4", repo_type="dataset")), 0, len(P["t"]))
