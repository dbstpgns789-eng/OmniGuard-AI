# 실험 21 준비: 시험 영상 20개의 포즈(performance, OpenVINO, 매 프레임)와 화면 보정(위험한계를 그린 화면 기준 ORB 호모그래피)
# 결과: OmniGuard-data/exp21/<영상>_pose.npz  (t, H=보정, kps, sc, wh, ms=프레임당 포즈 시간)
import os, json, time, numpy as np, cv2
from rtmlib import Body
DATA = "C:/Users/User/Desktop/OmniGuard-data/"; C = DATA + "collected/"; LAB = DATA + "세훈_작업/판단표_사고19/"; OUT = DATA + "exp21/"
os.makedirs(OUT, exist_ok=True)
CLIPS = [("A04_carpet", C + "A/A04.mp4", 433, 449), ("A09_pipe", C + "A/A09.mp4", 0, 22), ("A10_pipe_view2", C + "A/A10.mp4", 0, 22),
         ("B01_lathe", C + "B/B01.mp4", 71, 90), ("B02_lathe_chuck", C + "B/B02.mp4", 3, 26.5), ("B38_mixer", C + "B/B38.mp4", 0, 10),
         ("C03_drum", C + "C/C03.mp4", 49, 61)] + [(f"D{i:02d}", C + f"D/D{i:02d}.mp4", 0, 1e9) for i in range(1, 14)]
orb, bf = cv2.ORB_create(4000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
body = Body(mode="performance", backend="openvino", device="cpu")
for name, src, t0, t1 in CLIPS:
    path = OUT + f"{name}_pose.npz"
    if os.path.exists(path): continue
    fr = json.load(open(LAB + f"{name}/위험한계.json"))["frame"]
    ref = cv2.imdecode(np.fromfile(LAB + f"{name}/frames/{fr}", np.uint8), cv2.IMREAD_COLOR)
    kr, dr = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    cap = cv2.VideoCapture(src); fps = cap.get(5); n = int(cap.get(7)); f0, f1 = int(round(t0 * fps)), min(n, int(round(t1 * fps))); cap.set(1, f0)
    T, Hs, K, S, MS, H = [], [], [], [], [], np.eye(3)
    for f in range(f0, f1):
        ok, img = cap.read()
        if not ok: break
        k, d = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        if d is not None and dr is not None:
            mt = sorted(bf.match(dr, d), key=lambda x: x.distance)[:600]
            if len(mt) >= 30:
                Hn, inl = cv2.findHomography(np.float32([kr[x.queryIdx].pt for x in mt]), np.float32([k[x.trainIdx].pt for x in mt]), cv2.RANSAC, 4.0)
                if Hn is not None and inl.sum() >= 30: H = Hn
        s = time.perf_counter(); kps, sc = body(img); MS.append((time.perf_counter() - s) * 1000)
        T.append(f / fps); Hs.append(H.copy()); K.append(np.asarray(kps)); S.append(np.asarray(sc))
    np.savez(path, t=np.array(T), H=np.array(Hs), kps=np.array(K, dtype=object), sc=np.array(S, dtype=object),
             wh=np.array(ref.shape[1::-1]), ms=np.array(MS))
    print(name, len(T), "frames", f"pose median {np.median(MS):.0f}ms", flush=True)
print("done", flush=True)
