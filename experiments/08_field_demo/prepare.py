# 실험 08 준비: 공장 사고 영상 5건의 포즈, 화면 보정, 정답 표시용 1초 간격 시트
# 보정: none(고정 카메라) / affine(편집으로 옆 이동) / homography(모니터를 휴대폰으로 촬영). 기준 프레임(ref_t)에 그린 위험한계를 프레임마다 옮긴다
# 결과는 영상이 들어 있으므로 레포 밖 OmniGuard-data/exp08/에 저장한다
import json, os, sys
import numpy as np, cv2
from rtmlib import Body

DATA = "C:/Users/User/Desktop/OmniGuard-data/"
OUT = DATA + "exp08/"; os.makedirs(OUT + "sheets", exist_ok=True)
CASES = json.load(open(os.path.join(os.path.dirname(__file__), "cases.json"), encoding="utf-8"))

def warp_model(kind, ref, img, orb, bf, kr, dr):
    if kind == "none": return np.eye(3)
    k, d = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    if d is None: return None
    m = sorted(bf.match(dr, d), key=lambda x: x.distance)[:500]
    if len(m) < 20: return None
    src = np.float32([kr[x.queryIdx].pt for x in m]); dst = np.float32([k[x.trainIdx].pt for x in m])
    if kind == "affine":
        A, inl = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=4)
        return None if A is None or inl.sum() < 15 else np.vstack([A, [0, 0, 1]])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return None if H is None or inl.sum() < 20 else H

body = Body(mode="lightweight", backend="onnxruntime", device="cpu")
orb, bf = cv2.ORB_create(3000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
for name, c in CASES.items():
    if sys.argv[1:] and name not in sys.argv[1:]: continue
    npz = OUT + f"{name}_pose.npz"
    if os.path.exists(npz): continue
    cap = cv2.VideoCapture(DATA + "raw_cctv/" + c["file"]); fps = cap.get(5)
    cap.set(1, int(round(c["ref_t"] * fps))); ok, ref = cap.read(); h, w = ref.shape[:2]
    kr, dr = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    rot0 = [np.array(p, np.float64) * [w, h] for p in c["rot"]]
    cap.set(1, int(round(c["t0"] * fps))); T, Hs, K, S, tiles, H = [], [], [], [], [], np.eye(3)
    for f in range(int(round(c["t0"] * fps)), int(round(c["t1"] * fps))):
        ok, img = cap.read()
        if not ok: break
        Hn = warp_model(c["stabilize"], ref, img, orb, bf, kr, dr)
        if Hn is not None: H = Hn  # 실패하면 직전 값을 쓴다
        kps, sc = body(img); t = f / fps
        T.append(t); Hs.append(H.copy()); K.append(np.asarray(kps)); S.append(np.asarray(sc))
        if abs(t - round(t)) < 0.5 / fps + 1e-6:  # 1초마다 시트 타일
            vis = img.copy()
            for p in rot0:
                q = cv2.perspectiveTransform(p.reshape(-1, 1, 2), H).reshape(-1, 2)
                cv2.polylines(vis, [q.astype(np.int32)], True, (0, 0, 255), max(2, w // 300))
            for k, s in zip(kps, sc):
                for j in range(17):
                    if s[j] >= 0.3: cv2.circle(vis, (int(k[j][0]), int(k[j][1])), max(3, w // 250), (0, 255, 0) if j in (9, 10) else (255, 255, 0), -1)
            x0, y0, x1, y1 = c.get("crop", [0, 0, 1, 1])
            vis = vis[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
            vis = cv2.resize(vis, (360, int(360 * vis.shape[0] / vis.shape[1])))
            cv2.putText(vis, f"{t:.0f}s", (4, 22), 0, 0.7, (0, 0, 255), 2); tiles.append(vis)
    np.savez(npz, t=np.array(T), H=np.array(Hs), kps=np.array(K, dtype=object), sc=np.array(S, dtype=object), wh=np.array([w, h]))
    for i in range(0, len(tiles), 24):  # 시트 한 장에 24칸
        part = tiles[i:i + 24]
        while len(part) % 6: part.append(np.zeros_like(part[0]))
        cv2.imwrite(OUT + f"sheets/{name}_{i // 24}.jpg", np.vstack([np.hstack(part[r:r + 6]) for r in range(0, len(part), 6)]))
    print(name, len(T), "frames", flush=True)
