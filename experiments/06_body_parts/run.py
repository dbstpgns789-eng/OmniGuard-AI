# 실험 06: 판단에 쓰는 관절을 손목 2개 → 17개 전부로 바꾸면 오정지와 접촉 순간 정지가 어떻게 달라지는가
# 영상·회전체 다각형·화면 이동 보정은 실험 05와 같다. 포즈 결과를 npz로 저장해 이후 실험에서 다시 쓴다
import json, os
import numpy as np, cv2
from rtmlib import Body

DATA = "C:/Users/User/Desktop/OmniGuard-data/"
CASES = {  # 실험 05와 같은 값
    "rec_001918": dict(t0=4.0, t1=18.5, tc=15.0, ref_t=16.0, stabilize=False,
                       rot=[[(0.15, 0.28), (0.52, 0.20), (0.57, 0.50), (0.53, 0.70), (0.20, 0.54)]]),
    "rec_002103": dict(t0=7.0, t1=46.5, tc=37.0, ref_t=36.0, stabilize=True,
                       rot=[[(0.24, 0.18), (0.95, 0.25), (0.95, 0.36), (0.80, 0.40), (0.62, 0.48), (0.45, 0.45), (0.30, 0.40)],
                            [(0.78, 0.36), (1.0, 0.33), (1.0, 0.80), (0.80, 0.80), (0.76, 0.55)]]),
}
JOINTS = {"wrist": (9, 10), "all": tuple(range(17))}
KP_MIN, NEED_STOP = 0.3, 3

def in_any(pt, polys): return any(cv2.pointPolygonTest(p.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0 for p in polys)

def pose_cache(name, cfg):
    path = f"{name}_pose.npz"
    if os.path.exists(path): return dict(np.load(path, allow_pickle=True))
    body = Body(mode="lightweight", backend="onnxruntime", device="cpu")
    orb, bf = cv2.ORB_create(2000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    cap = cv2.VideoCapture(DATA + f"raw_cctv/{name}.mp4"); fps = cap.get(5)
    cap.set(1, int(cfg["ref_t"] * fps)); ok, ref = cap.read(); kr, dr = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    cap.set(1, int(cfg["t0"] * fps)); M, ts, Ms, K, S = np.eye(2, 3), [], [], [], []
    for f in range(int(cfg["t0"] * fps), int(cfg["t1"] * fps)):
        ok, img = cap.read()
        if not ok: break
        if cfg["stabilize"]:
            k, d = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            m = sorted(bf.match(dr, d), key=lambda x: x.distance)[:400] if d is not None else []
            if len(m) >= 20:
                A, inl = cv2.estimateAffinePartial2D(np.float32([kr[x.queryIdx].pt for x in m]), np.float32([k[x.trainIdx].pt for x in m]), ransacReprojThreshold=4)
                if A is not None and inl.sum() >= 15: M = A
        kps, sc = body(img)
        ts.append(f / fps); Ms.append(M.copy()); K.append(np.asarray(kps)); S.append(np.asarray(sc))
    out = dict(t=np.array(ts), M=np.array(Ms), kps=np.array(K, dtype=object), sc=np.array(S, dtype=object), wh=np.array(ref.shape[1::-1]))
    np.savez(path, **out); return out

summary = {}
for name, cfg in CASES.items():
    P = pose_cache(name, cfg); w, h = P["wh"]
    rot0 = [np.array(p) * [w, h] for p in cfg["rot"]]
    res = {}
    for jn, joints in JOINTS.items():
        raw = []
        for M, kps, sc in zip(P["M"], P["kps"], P["sc"]):
            rot = [np.c_[p, np.ones(len(p))] @ M.T for p in rot0]
            raw.append(any(s[j] >= KP_MIN and in_any(k[j], rot) for k, s in zip(kps, sc) for j in joints))
        stop = [i >= NEED_STOP - 1 and all(raw[i - NEED_STOP + 1:i + 1]) for i in range(len(raw))]
        t, tc = P["t"], cfg["tc"]
        pre = [s for s, x in zip(stop, t) if x < tc - 3]
        near = [s for s, x in zip(stop, t) if tc - 0.5 <= x <= tc + 0.5]
        first = next((x for s, x in zip(stop, t) if s and tc - 3 <= x <= tc), None)
        res[jn] = {"pre_approach_stop_ratio": round(float(np.mean(pre)), 3), "stop_ratio_tc_pm0.5": round(float(np.mean(near)), 3),
                   "lead_s": None if first is None else round(float(tc - first), 2)}
    summary[name] = res; print(name, json.dumps(res))
json.dump(summary, open("result.json", "w"), indent=1)
