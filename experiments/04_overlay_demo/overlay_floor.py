# 개선 1: 바닥 기준 거리. 화면에서 손이 회전체와 겹쳐 보여도, 발이 팔이 닿는 거리 안에 있을 때만 믿는다
# 같은 클립(trim_7)의 개선 전 결과(frames.csv clip 1)와 단계 기록을 나란히 그린다
import csv, time
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from rtmlib import Body

SRC = "C:/Users/User/Desktop/OmniGuard-data/curated/isafety_nip/0QHQ2nmdE78_clip0_trim_7.mp4"
OUT = "C:/Users/User/Desktop/OmniGuard-data/demo/roll_winder_floor.mp4"
X0, X1 = 362, 918
# 화면 비율 좌표. 회전체 = 앞쪽 큰 롤 몸통 + 끝단 원판, 바닥선 = 설비 아래 레일이 바닥과 만나는 선
ROLL = np.array([(0.0, 0.34), (0.85, 0.27), (0.88, 0.47), (0.55, 0.58), (0.40, 0.66), (0.30, 0.66), (0.0, 0.60)])
B0, B1 = np.array([0.35, 0.68]), np.array([0.95, 0.52])
HEIGHT_M, FORESHORTEN = 1.6, 0.65  # 키로 축척을 정하고, 위에서 내려다보는 각도 때문에 바닥 깊이 방향이 줄어 보이는 비율(추정)
REACH_M, WARN_M = 1.0, 2.0
NAME = {0: "반응 없음", 1: "주의", 2: "감속", 3: "정지"}
COLOR = {0: (90, 90, 90), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 255)}
NEED = {1: 1, 2: 2, 3: 3}
KP_MIN = 0.3
FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 24)
SMALL = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 18)

def in_poly(pt, poly): return cv2.pointPolygonTest(poly.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0

def person_level(k, s, roll, b0, b1):
    if max(s[15], s[16]) < KP_MIN: return 0, None  # 발목이 안 보이면 거리를 잴 수 없다
    foot = np.mean([k[j] for j in (15, 16) if s[j] >= KP_MIN], axis=0)
    top = min(k[j][1] for j in range(5) if s[j] >= KP_MIN) if max(s[:5]) >= KP_MIN else None
    if top is None: return 0, None
    px_per_m = (foot[1] - top) * 1.1 / HEIGHT_M * FORESHORTEN  # 머리 꼭대기는 눈·코보다 조금 위
    line_y = b0[1] + (b1[1] - b0[1]) * (foot[0] - b0[0]) / (b1[0] - b0[0])
    d = max(0.0, (foot[1] - line_y) / px_per_m)  # 바닥선 앞쪽(화면 아래)으로 떨어진 거리
    hand_on_roll = any(s[j] >= KP_MIN and in_poly(k[j], roll) for j in (9, 10))
    if d < REACH_M and hand_on_roll: return 3, d
    if d < REACH_M: return 2, d
    if d < WARN_M: return 1, d
    return 0, d

before = [int(r["shown"]) for r in csv.DictReader(open("frames.csv", encoding="utf-8")) if r["clip"] == "1"]
body = Body(mode="lightweight", backend="onnxruntime", device="cpu")
cap = cv2.VideoCapture(SRC); fps = cap.get(5)
writer, raw_hist, shown_hist, rows, f = None, [], [], [], 0
while True:
    ok, full = cap.read()
    if not ok: break
    img = full[:, X0:X1].copy(); h, w = img.shape[:2]
    roll = ROLL * [w, h]; b0, b1 = B0 * [w, h], B1 * [w, h]
    t0 = time.perf_counter(); kps, sc = body(img); t_pose = (time.perf_counter() - t0) * 1000
    per = [person_level(k, s, roll, b0, b1) for k, s in zip(kps, sc)]
    raw = max([p[0] for p in per] or [0]); raw_hist.append(raw)
    shown = max([L for L in (3, 2, 1) if len(raw_hist) >= NEED[L] and min(raw_hist[-NEED[L]:]) >= L] or [0]); shown_hist.append(shown)
    dist = min([p[1] for p in per if p[1] is not None] or [None], key=lambda x: 99 if x is None else x)
    rows.append({"frame": f, "t": round(f / fps, 2), "raw": raw, "shown": shown, "before": before[f] if f < len(before) else "",
                 "foot_dist_m": None if dist is None else round(dist, 2), "pose_ms": round(t_pose, 1)})
    over = img.copy(); cv2.fillPoly(over, [roll.astype(np.int32)], (0, 0, 255))
    img = cv2.addWeighted(over, 0.18, img, 0.82, 0)
    cv2.polylines(img, [roll.astype(np.int32)], True, (0, 0, 255), 2)
    cv2.line(img, tuple(b0.astype(int)), tuple(b1.astype(int)), (255, 255, 255), 2)
    for k, s in zip(kps, sc):
        for j in range(17):
            if s[j] >= KP_MIN: cv2.circle(img, tuple(int(v) for v in k[j]), 6 if j in (9, 10, 15, 16) else 3, (0, 255, 0) if j in (9, 10) else (255, 0, 255) if j in (15, 16) else (255, 255, 0), -1)
    cv2.rectangle(img, (0, 0), (w, 44), COLOR[shown], -1)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil)
    d.text((10, 8), f"개선 후: {NAME[shown]}   발-설비 {'-' if dist is None else f'{dist:.1f}m'}   t={f / fps:4.1f}s", font=FONT, fill=(255, 255, 255))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    bars = []
    for label, hist in (("개선 전", before[:f + 1]), ("개선 후", shown_hist)):
        bar = np.full((26, w, 3), 30, np.uint8)
        for i, L in enumerate(hist):
            x = 70 + int(i * (w - 70) / 180); cv2.line(bar, (x, 2), (x, 23), COLOR[L], 3)
        p = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)); ImageDraw.Draw(p).text((4, 2), label, font=SMALL, fill=(255, 255, 255))
        bars.append(cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR))
    img = np.vstack([img] + bars)
    if writer is None: writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
    writer.write(img); f += 1
writer.release()

with open("frames_floor.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
for key in ("before", "shown"):
    v = [int(r[key]) for r in rows if r[key] != ""]
    print(key, {NAME[L]: v.count(L) for L in range(4)}, "".join("-WSX"[max(v[i:i + 15])] for i in range(0, len(v), 15)))
print("foot dist m: min", min(r["foot_dist_m"] for r in rows if r["foot_dist_m"] is not None), "max", max(r["foot_dist_m"] for r in rows if r["foot_dist_m"] is not None))
