# 실험 10 준비: iSafetyBench 빨간 상의 넓은 화면(trim 2·4·7·8)의 포즈와 화면 보정
# 클립마다 자른 범위가 조금씩 달라서, 모든 프레임을 기준 프레임(trim_02의 0초)에 맞추는 homography를 배경 특징점으로 구한다
# 포즈는 실험 09와 같은 큰 모델(performance, OpenVINO). 결과는 레포 밖 OmniGuard-data/exp10/
import os, json, numpy as np, cv2
from rtmlib import Body

DATA = "C:/Users/User/Desktop/OmniGuard-data/"; SRC = DATA + "curated/isafety_nip/"; OUT = DATA + "exp10/"
os.makedirs(OUT, exist_ok=True)
CLIPS = [2, 4, 7, 8]
def content(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); cols = np.where(g.mean(0) > 15)[0]; return int(cols.min()), int(cols.max()) + 1

c = cv2.VideoCapture(SRC + "0QHQ2nmdE78_clip0_trim_2.mp4"); ok, ref = c.read(); x0, x1 = content(ref); ref = ref[:, x0:x1]
h_ref = ref.shape[0]; ref_mask = np.full(ref.shape[:2], 255, np.uint8); ref_mask[int(h_ref * 0.80):] = 0  # 아래 자막 띠는 특징점에서 뺌
cv2.imwrite(OUT + "ref.jpg", ref)
orb, bf = cv2.ORB_create(4000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
kr, dr = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), ref_mask)
body = Body(mode="performance", backend="openvino", device="cpu")
for n in CLIPS:
    path = OUT + f"trim_{n:02d}_pose.npz"
    if os.path.exists(path): continue
    cap = cv2.VideoCapture(SRC + f"0QHQ2nmdE78_clip0_trim_{n}.mp4"); fps = cap.get(5)
    ok, f0 = cap.read(); a, b = content(f0); cap.set(1, 0)
    T, Hs, K, S, H = [], [], [], [], np.eye(3); i = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        img = img[:, a:b]; m = np.full(img.shape[:2], 255, np.uint8); m[int(img.shape[0] * 0.80):] = 0
        k, d = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), m)
        if d is not None:
            mt = sorted(bf.match(dr, d), key=lambda x: x.distance)[:600]
            if len(mt) >= 30:
                Hn, inl = cv2.findHomography(np.float32([kr[x.queryIdx].pt for x in mt]), np.float32([k[x.trainIdx].pt for x in mt]), cv2.RANSAC, 4.0)
                if Hn is not None and inl.sum() >= 30: H = Hn
        kps, sc = body(img)
        T.append(i / fps); Hs.append(H.copy()); K.append(np.asarray(kps)); S.append(np.asarray(sc)); i += 1
    np.savez(path, t=np.array(T), H=np.array(Hs), kps=np.array(K, dtype=object), sc=np.array(S, dtype=object),
             wh=np.array([ref.shape[1], ref.shape[0]]), crop=np.array([a, b]))
    print(n, len(T), "frames", flush=True)
json.dump({"ref": "trim_02 0.0s", "clips": CLIPS}, open(OUT + "prepare_done.json", "w"))
