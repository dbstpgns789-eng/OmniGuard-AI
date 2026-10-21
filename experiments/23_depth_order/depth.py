# 실험 23 준비: 시험 영상 20개를 0.2초 간격으로 Depth Anything V2 Small에 넣어 상대 깊이 지도를 저장 (원본의 1/4 크기)
# 저장: OmniGuard-data/exp23/<영상>_depth.npz (t, depth[N,h/4,w/4] float16, ms)
import os, time, numpy as np, cv2, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "21_first_test", "prepare.py"), encoding="utf-8").read(); exec(src[src.index("DATA ="):src.index("orb, bf")])
OUT = DATA + "exp23/"; os.makedirs(OUT, exist_ok=True); ID = "depth-anything/Depth-Anything-V2-Small-hf"
proc = AutoImageProcessor.from_pretrained(ID); model = AutoModelForDepthEstimation.from_pretrained(ID).eval(); torch.set_num_threads(os.cpu_count())
for name, path, t0, t1 in CLIPS:
    dst = OUT + f"{name}_depth.npz"
    if os.path.exists(dst): continue
    P = np.load(DATA + f"exp21/{name}_pose.npz", allow_pickle=True); t = P["t"]; cap = cv2.VideoCapture(path)
    T, D, MS = [], [], []
    for k in np.arange(round(t[0] / 0.2) * 0.2, t[-1] + 1e-6, 0.2):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(k) * 1000); ok, img = cap.read()
        if not ok: continue
        h, w = img.shape[:2]; s = time.perf_counter()
        with torch.no_grad():
            out = model(**proc(images=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), return_tensors="pt")).predicted_depth
        d = torch.nn.functional.interpolate(out[None], size=(h // 4, w // 4), mode="bilinear", align_corners=False)[0, 0].numpy()
        MS.append((time.perf_counter() - s) * 1000); T.append(round(float(k), 1)); D.append(d.astype(np.float16))
    np.savez_compressed(dst, t=np.array(T), depth=np.array(D), ms=np.array(MS)); print(name, len(T), f"{np.median(MS):.0f}ms", flush=True)
print("done", flush=True)
