# 실험 12 준비: 사용자가 판단한 사고 영상 5건의 기준 화면과 포즈(performance, OpenVINO). 고정 카메라라 화면 보정 없음(H=단위행렬)
import os, json, numpy as np, cv2
from rtmlib import Body
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = "C:/Users/User/Desktop/OmniGuard-data/"; OUT = DATA + "exp12/"
C = json.load(open(os.path.join(HERE, "clips.json"), encoding="utf-8"))
for k, c in C.items():  # 그리기용 기준 화면
    cap = cv2.VideoCapture(DATA + "raw_cctv/" + c["file"]); cap.set(1, int(round(c["ref_t"] * cap.get(5)))); ok, img = cap.read(); cv2.imwrite(OUT + f"ref_{k}.jpg", img)
body = Body(mode="performance", backend="openvino", device="cpu")
for k, c in C.items():
    path = OUT + f"{k}_pose.npz"
    if os.path.exists(path): continue
    cap = cv2.VideoCapture(DATA + "raw_cctv/" + c["file"]); fps = cap.get(5); f0, f1 = int(round(c["t0"] * fps)), int(round(c["t1"] * fps)); cap.set(1, f0)
    T, K, S = [], [], []; wh = None
    for f in range(f0, f1):
        ok, img = cap.read()
        if not ok: break
        wh = img.shape[1::-1]; kps, sc = body(img); T.append(f / fps); K.append(np.asarray(kps)); S.append(np.asarray(sc))
    np.savez(path, t=np.array(T), H=np.repeat(np.eye(3)[None], len(T), 0), kps=np.array(K, dtype=object), sc=np.array(S, dtype=object), wh=np.array(wh))
    print(k, len(T), flush=True)
