# 3D 포즈(RTMW3D-x)로 관절의 앞뒤 깊이를 구한다. 2프레임마다 계산하고 사이는 앞 값을 쓴다. 클립마다 저장(이미 있으면 건너뜀)
# 저장: 2D 포즈(이미 구한 performance 결과)와 같은 프레임 순서로, 사람마다 COCO 17관절의 깊이 z와 2D 위치(원본 화면 좌표 추정용 중심)
import os, sys, numpy as np, cv2
from rtmlib import Wholebody3d

DATA = "C:/Users/User/Desktop/OmniGuard-data/"
JOBS = [(DATA + f"curated/isafety_nip/0QHQ2nmdE78_clip0_trim_{n}.mp4", DATA + f"exp10/trim_{n:02d}_pose.npz", DATA + f"exp10/trim_{n:02d}_z.npz", True) for n in (2, 4, 7, 8)]
JOBS.append((DATA + "raw_cctv/rec_002103.mp4", DATA + "exp09/red_crop_pose_performance.npz", DATA + "exp10/red_crop_z.npz", False))
m = Wholebody3d(backend="onnxruntime", device="cpu")
for video, pose2d, out, cropped in JOBS:
    if os.path.exists(out): continue
    P = dict(np.load(pose2d, allow_pickle=True)); t = P["t"]; n = len(t)
    cap = cv2.VideoCapture(video); fps = cap.get(5)
    a, b = P["crop"] if cropped else (0, None)
    Z, C, last = [], [], ([], [])
    for i in range(n):
        if i % 2 == 0:
            cap.set(1, int(round((t[i] if cropped else t[i]) * fps))); ok, img = cap.read()
            if ok:
                img = img[:, a:b] if cropped else img
                k3, sc, _, k2 = m(img)
                last = ([kk[:17, 2].astype(np.float32) for kk in k3], [k2d[:17].astype(np.float32) for k2d in k2])
        Z.append(last[0]); C.append(last[1])
        if i % 100 == 0: print(os.path.basename(out), i, n, flush=True)
    np.savez(out, z=np.array(Z, dtype=object), k2=np.array(C, dtype=object))
print("done", flush=True)
