# 실험 11 준비: 프레스 공장(Safe_and_Unsafe, 시점 A)에서 롤러 설비(코일 피더·언코일러) 근처에 사람이 온 클립 14개
# 큰 포즈 모델(performance, OpenVINO)로 모든 프레임 포즈. 고정 CCTV라 화면 보정은 없음(H=단위행렬). 판단용 1초 간격 프레임과 판단표도 만든다
import os, json, csv, numpy as np, cv2
from huggingface_hub import hf_hub_download
from rtmlib import Body
REPO = "Voxel51/Safe_and_Unsafe_Behaviours"; DATA = "C:/Users/User/Desktop/OmniGuard-data/"; OUT = DATA + "exp11/"
scan = json.load(open(DATA + "exp02/feeder_hits/scan.json", encoding="utf-8"))
clips = [r["clip"] for r in scan if r["hits"]]
os.makedirs(OUT + "frames", exist_ok=True); rows = []
for c in clips:
    name = c.split("/")[-1][:-4]; cap = cv2.VideoCapture(hf_hub_download(REPO, c, repo_type="dataset")); fps = cap.get(5); n = int(cap.get(7))
    for f in range(0, n, max(1, int(round(fps)))):
        cap.set(1, f); ok, img = cap.read()
        if not ok: break
        fn = f"{name}_{f / fps:04.1f}s.jpg"; cv2.imwrite(OUT + "frames/" + fn, cv2.resize(img[0:760, 380:1300], None, fx=0.9, fy=0.9))
        rows.append({"클립": name, "시각(초)": f"{f / fps:.1f}", "파일": "frames\\" + fn, "단계(정상/주의/감속/정지)": "정상", "메모": ""})
with open(OUT + "판단표_press.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("label rows", len(rows), flush=True)
body = Body(mode="performance", backend="openvino", device="cpu")
for c in clips:
    name = c.split("/")[-1][:-4]; path = OUT + f"{name}_pose.npz"
    if os.path.exists(path): continue
    cap = cv2.VideoCapture(hf_hub_download(REPO, c, repo_type="dataset")); fps = cap.get(5); T, K, S = [], [], []; i = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        kps, sc = body(img); T.append(i / fps); K.append(np.asarray(kps)); S.append(np.asarray(sc)); i += 1
    np.savez(path, t=np.array(T), H=np.repeat(np.eye(3)[None], len(T), 0), kps=np.array(K, dtype=object), sc=np.array(S, dtype=object), wh=np.array([1920, 1080]))
    print(name, len(T), flush=True)
