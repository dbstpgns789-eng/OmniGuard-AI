# 09-3: 큰 포즈 모델(performance)로 red_crop 포즈를 다시 구한다(실험 07에서 야간 작업자 검출을 크게 살린 모델)
# CPU에서 프레임당 약 3초라 오래 걸린다. 100프레임마다 중간 저장하고, 다시 실행하면 이어서 한다
# 화면 보정 행렬 H는 실험 08 결과를 그대로 쓴다(같은 프레임)
import os, sys, numpy as np, cv2
from rtmlib import Body
from common import *

kind = sys.argv[1] if len(sys.argv) > 1 else "performance"
os.makedirs(DATA + "exp09", exist_ok=True)
L = load_pose("lightweight"); t = L["t"]
part = DATA + f"exp09/red_crop_pose_{kind}_partial.npz"
K, S = ([], []) if not os.path.exists(part) else (list(np.load(part, allow_pickle=True)["kps"]), list(np.load(part, allow_pickle=True)["sc"]))
body = Body(mode=kind, backend="openvino", device="cpu")  # OpenVINO가 onnxruntime보다 빨랐다(실험 07-3b)
cap = cv2.VideoCapture(DATA + "raw_cctv/rec_002103.mp4"); fps = cap.get(5)
cap.set(1, int(round(t[len(K)] * fps)) if len(K) < len(t) else 0)
for i in range(len(K), len(t)):
    ok, img = cap.read()
    if not ok: break
    kps, sc = body(img); K.append(np.asarray(kps)); S.append(np.asarray(sc))
    if i % 100 == 0:
        np.savez(part, kps=np.array(K, dtype=object), sc=np.array(S, dtype=object)); print(i, len(t), flush=True)
np.savez(DATA + f"exp09/red_crop_pose_{kind}.npz", t=t[:len(K)], H=L["H"][:len(K)], wh=L["wh"], kps=np.array(K, dtype=object), sc=np.array(S, dtype=object))
done(f"t3_pose_{kind}", {"frames": len(K)})
