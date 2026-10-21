# 09-5: 데모 영상. 원통 구역을 입체처럼(축 + 정지·감속·주의 두께의 띠) 그리고, 상단에 모델 단계, 하단에 사람 판단과 모델 기록 막대
# 인자: 포즈 종류(lightweight/performance). 있는 예측 파일(t1 규칙, t2 분류기)을 모두 막대로 그린다
import os, sys, json, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from common import *

kind = sys.argv[1] if len(sys.argv) > 1 else "lightweight"
P = load_pose(kind); t = P["t"]; y = load_labels(t)
r1 = json.load(open(OUT + f"/t1_capsule_rule_{kind}/result.json", encoding="utf-8"))["fit_all"]["params"]
axis, radius, th = np.array(r1["axis"]), r1["radius"], r1["th_stop_slow_warn"]
preds = {}
for key, fn in (("규칙", f"/t1_pred_{kind}.npy"), ("분류기", f"/t2_pred_{kind}.npy")):
    try: preds[key] = np.load(OUT + fn)
    except Exception: pass
main = preds.get("분류기", preds.get("규칙"))
COL = {0: (90, 90, 90), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 255), -1: (40, 40, 40)}
FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 26); SM = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 16)
tl_med = np.nanmedian([np.linalg.norm((k[5] + k[6]) / 2 - (k[11] + k[12]) / 2) for ks, ss in zip(P["kps"], P["sc"]) for k, s in zip(ks, ss) if min(s[5], s[6], s[11], s[12]) >= KP_MIN])

def capsule(img, a, b, R, color, alpha):
    m = np.zeros(img.shape[:2], np.uint8); R = max(1, int(R))
    cv2.line(m, tuple(a.astype(int)), tuple(b.astype(int)), 255, 2 * R); cv2.circle(m, tuple(a.astype(int)), R, 255, -1); cv2.circle(m, tuple(b.astype(int)), R, 255, -1)
    ov = img.copy(); ov[m > 0] = color; return cv2.addWeighted(ov, alpha, img, 1 - alpha, 0), m

os.makedirs(DATA + "exp09", exist_ok=True)
cap = cv2.VideoCapture(DATA + "raw_cctv/rec_002103.mp4"); fps = cap.get(5); cap.set(1, int(round(t[0] * fps)))
w, h = P["wh"]; wr = None; n = len(t)
for i in range(n):
    ok, img = cap.read()
    if not ok: break
    a, b = to_frame(P, i, axis); r = radius * w * np.sqrt(abs(np.linalg.det(P["H"][i][:2, :2])))
    for lvl, d in ((1, th[2]), (2, th[1]), (3, th[0])):  # 바깥 띠부터 그려서 안쪽이 위에 오게
        img, mk = capsule(img, a, b, r + max(0, d) * tl_med, COL[lvl], 0.18)
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img, cs, -1, COL[lvl], 2)
    img, mk = capsule(img, a, b, r, (255, 80, 0), 0.25)  # 회전체 몸통(파랑)
    cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (255, 80, 0), 3)
    for k, s in zip(P["kps"][i], P["sc"][i]):
        for j in range(17):
            if s[j] >= KP_MIN: cv2.circle(img, (int(k[j][0]), int(k[j][1])), 6 if j in (9, 10) else 3, (0, 255, 0) if j in (9, 10) else (255, 255, 0), -1)
    lv = int(main[i]) if main is not None else 0
    cv2.rectangle(img, (0, 0), (w, 48), COL[lv], -1)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil)
    dr.text((10, 8), f"모델: {STAGES[lv]}   사람: {STAGES[y[i]] if y[i] >= 0 else '-'}   t={t[i]:4.1f}s", font=FONT, fill=(255, 255, 255))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR); bars = []
    for lab, arr in [("사람", y)] + list(preds.items()):
        bar = np.full((24, w, 3), 25, np.uint8)
        for j in range(i + 1): x = 70 + int(j * (w - 70) / n); cv2.line(bar, (x, 2), (x, 21), COL[int(arr[j])], max(1, (w - 70) // n + 1))
        p2 = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)); ImageDraw.Draw(p2).text((4, 2), lab, font=SM, fill=(255, 255, 255))
        bars.append(cv2.cvtColor(np.array(p2), cv2.COLOR_RGB2BGR))
    frame = np.vstack([img] + bars)
    if wr is None: wr = cv2.VideoWriter(DATA + f"exp09/demo_red_{kind}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
    wr.write(frame)
    if i in (int(n * 0.1), int(n * 0.75), int(n * 0.8)): cv2.imwrite(DATA + f"exp09/demo_red_{kind}_{i}.jpg", frame)
wr.release()
done(f"t5_render_{kind}", {"video": DATA + f"exp09/demo_red_{kind}.mp4", "bars": ["사람"] + list(preds)})
