# 전처리 맞추기: 개발 데이터(실험 11 프레스 14개, 실험 12 사고 5건)에도 시험 데이터(실험 21)와 같은 화면 보정을 적용한다
# 위험한계를 그린 기준 화면에 매 프레임을 ORB 특징점 + RANSAC 호모그래피로 맞춘 H만 새로 계산하고, 포즈는 기존 것을 그대로 쓴다
# 저장: OmniGuard-data/exp20_stab/<영상>_pose.npz (기존 포즈 파일 + 새 H)
import os, json, glob, numpy as np, cv2
from huggingface_hub import hf_hub_download
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, ".."); DATA = "C:/Users/User/Desktop/OmniGuard-data/"; OUT = DATA + "exp20_stab/"
os.makedirs(OUT, exist_ok=True); orb, bf = cv2.ORB_create(4000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def stabilize(src, ref, f0, n, dst, P):
    if os.path.exists(dst): return
    kr, dr = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None); cap = cv2.VideoCapture(src); cap.set(1, f0); Hs, H, moved = [], np.eye(3), 0
    for _ in range(n):
        ok, img = cap.read()
        if not ok: break
        k, d = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        if d is not None and dr is not None:
            mt = sorted(bf.match(dr, d), key=lambda x: x.distance)[:600]
            if len(mt) >= 30:
                Hn, inl = cv2.findHomography(np.float32([kr[x.queryIdx].pt for x in mt]), np.float32([k[x.trainIdx].pt for x in mt]), cv2.RANSAC, 4.0)
                if Hn is not None and inl.sum() >= 30: H = Hn
        Hs.append(H.copy())
    Hs += [Hs[-1]] * (n - len(Hs)); Q = dict(P); Q["H"] = np.array(Hs[:n]); np.savez(dst, **Q)
    shift = np.median([np.linalg.norm(h[:2, 2]) for h in Hs]); print(os.path.basename(dst), n, f"이동 중앙값 {shift:.1f}px", flush=True)

C12 = json.load(open(os.path.join(X, "12_accident_set", "clips.json"), encoding="utf-8"))
for k, c in C12.items():
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True)); src = DATA + "raw_cctv/" + c["file"]; fps = cv2.VideoCapture(src).get(5)
    stabilize(src, cv2.imread(DATA + f"exp12/ref_{k}.jpg"), int(round(c["t0"] * fps)), len(P["t"]), OUT + f"{k}_pose.npz", P)
ref11 = cv2.imread(DATA + "exp11/ref.jpg")
for p in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    name = os.path.basename(p)[:-9]; P = dict(np.load(p, allow_pickle=True))
    stabilize(hf_hub_download("Voxel51/Safe_and_Unsafe_Behaviours", f"data/{name}.mp4", repo_type="dataset"), ref11, 0, len(P["t"]), OUT + f"{name}_pose.npz", P)
print("done", flush=True)
