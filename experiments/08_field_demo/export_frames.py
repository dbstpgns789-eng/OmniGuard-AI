# 사용자가 직접 판단할 수 있도록 공장 영상 5건을 0.2초 간격 프레임으로 내보낸다
# raw/: 원본 프레임, zone/: Claude가 그린 위험한계(빨간 선)를 얹은 프레임, 판단표.csv: 사용자가 채울 표
import csv, json, os
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "C:/Users/User/Desktop/OmniGuard-data/"; OUT = DATA + "exp08/frames_for_review/"
CASES = json.load(open(os.path.join(HERE, "cases.json"), encoding="utf-8"))
STEP = 0.2
for name, c in CASES.items():
    P = dict(np.load(DATA + f"exp08/{name}_pose.npz", allow_pickle=True)); t = P["t"]; w, h = P["wh"]
    d = OUT + name + "/"; os.makedirs(d + "raw", exist_ok=True); os.makedirs(d + "zone", exist_ok=True)
    cap = cv2.VideoCapture(DATA + "raw_cctv/" + c["file"]); fps = cap.get(5); rows = []
    for s in np.arange(c["t0"], c["t1"], STEP):
        i = int(np.argmin(np.abs(t - s))); cap.set(1, int(round(t[i] * fps))); ok, img = cap.read()
        if not ok: continue
        scale = min(1.0, 1280 / w); img = cv2.resize(img, None, fx=scale, fy=scale)
        fn = f"{s:06.1f}s.jpg"; cv2.imencode(".jpg", img)[1].tofile(d + "raw/" + fn)
        z = img.copy()
        for p in c["rot"]:
            q = cv2.perspectiveTransform((np.array(p, np.float64) * [w, h]).reshape(-1, 1, 2), P["H"][i]).reshape(-1, 2) * scale
            cv2.polylines(z, [q.astype(np.int32)], True, (0, 0, 255), 3)
        cv2.putText(z, f"{s:.1f}s", (8, 30), 0, 1.0, (0, 0, 255), 2); cv2.imencode(".jpg", z)[1].tofile(d + "zone/" + fn)
        rows.append({"시각(초)": f"{s:.1f}", "파일": fn, "단계(정상/주의/감속/정지)": "", "회전체에 닿음(Y/N/모름)": "", "메모": ""})
    with open(d + "판단표.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    print(name, len(rows), "frames", flush=True)
