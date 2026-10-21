# 실험 26 준비: 약한 라벨 사고 18건(닿기 10초 전 ~ 0.5초 뒤)과 Safe and Unsafe 정상 40개의 포즈를 초당 10장으로 계산 (performance, OpenVINO)
import os, glob, random, numpy as np, cv2
from rtmlib import Body
DATA = "C:/Users/User/Desktop/OmniGuard-data/"; OUT = DATA + "exp26/pose/"; os.makedirs(OUT, exist_ok=True)
WEAK = {"A01": 34.9, "A03": 17.6, "A07": 14.3, "A08": 9.9, "B03": 15.5, "B04": 7.6, "B05": 71.3, "B06": 7.4, "B08": 5.0, "B09": 138.8,
        "B15": 15.4, "B21": 19.6, "B22": 18.4, "B31": 37.5, "B33": 27.0, "B35": 33.3, "B37": 11.6, "B39": 17.8}
SU = sorted(glob.glob(DATA + "open/safe_unsafe/*.mp4")); random.Random(0).shuffle(SU); SU = SU[:40]
body = Body(mode="performance", backend="openvino", device="cpu")
def run(name, src, t0, t1):
    dst = OUT + f"{name}.npz"
    if os.path.exists(dst): return
    cap = cv2.VideoCapture(src); fps = cap.get(5); n = int(cap.get(7)); step = max(1, int(round(fps / 10))); f0 = max(0, int(round(t0 * fps))); f1 = min(n, int(round(t1 * fps)))
    T, K, S = [], [], []; wh = None
    for f in range(f0, f1, step):
        cap.set(1, f); ok, img = cap.read()
        if not ok: break
        wh = img.shape[1::-1]; kps, sc = body(img); T.append(f / fps); K.append(np.asarray(kps)); S.append(np.asarray(sc))
    np.savez(dst, t=np.array(T), kps=np.array(K, dtype=object), sc=np.array(S, dtype=object), wh=np.array(wh)); print(name, len(T), flush=True)
for k, tc in WEAK.items(): run(f"weak_{k}", DATA + f"collected/{k[0]}/{k}.mp4", tc - 10, tc + 0.5)
for p in SU: run("su_" + os.path.basename(p)[:-4], p, 0, 1e9)
print("done", flush=True)
