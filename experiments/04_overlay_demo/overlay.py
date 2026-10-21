# 현장 사고 영상에 빠른 경로(A안)를 프레임마다 돌려 주의·감속·정지를 화면에 얹는다
# 영상은 재배포하지 않으므로 결과 영상은 레포 밖 OmniGuard-data/demo에 저장하고, 프레임 기록(csv)만 레포에 남긴다
import csv, time
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from rtmlib import Body

SRC = "C:/Users/User/Desktop/OmniGuard-data/curated/isafety_nip/"
OUT = "C:/Users/User/Desktop/OmniGuard-data/demo/roll_winder_A.mp4"
CLIPS = ["0QHQ2nmdE78_clip0_trim_4.mp4", "0QHQ2nmdE78_clip0_trim_7.mp4", "0QHQ2nmdE78_clip0_trim_8.mp4"]  # 사고 순서
X0, X1 = 362, 918  # 세로로 잘린 뉴스 화면에서 실제 영상이 있는 열

# 맞물림선: 앞쪽 큰 롤 아랫면이 아래 롤러와 맞닿는 선(화면 비율 좌표). 띠는 이 선에서 작업자 쪽(아래)으로 넓어진다
L0, L1 = np.array([0.36, 0.60]), np.array([0.86, 0.46])
def band(top, bottom):  # 맞물림선을 위아래로 평행 이동한 사각형
    return np.array([L0 + [0, top], L1 + [0, top], L1 + [0, bottom], L0 + [0, bottom]])
BANDS = {3: band(-0.06, 0.02), 2: band(0.02, 0.08), 1: band(0.08, 0.20)}  # 3 맞물림부 안, 2 가까운 띠, 1 먼 띠
NAME = {0: "반응 없음", 1: "주의", 2: "감속", 3: "정지"}
COLOR = {0: (90, 90, 90), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 255)}  # BGR
NEED = {1: 1, 2: 2, 3: 3}  # 단계를 띄우기 전에 연속으로 확인할 프레임 수
KP_MIN, BEND_DEG = 0.3, 30
FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 26)

def in_poly(pt, poly): return cv2.pointPolygonTest(poly.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0

def person_level(k, s, polys):
    # 몸의 위치: 신뢰도 이상인 관절 중 가장 위험한 띠
    pos = max([b for b in (3, 2, 1) for j in range(17) if s[j] >= KP_MIN and in_poly(k[j], polys[b])] or [0])
    # 위험 동작(느린 경로 대신 임시 규칙): 몸통이 수직에서 BEND_DEG 넘게 숙여짐
    bend = False
    if min(s[5], s[6], s[11], s[12]) >= KP_MIN:
        v = (k[5] + k[6]) / 2 - (k[11] + k[12]) / 2
        bend = np.degrees(np.arctan2(abs(v[0]), -v[1])) > BEND_DEG
    # 조합 표 (가동 여부는 켜짐으로 둔다)
    if pos == 3: return 3, pos, bend
    if pos == 2: return (2 if bend else 1), pos, bend
    return (1 if pos == 1 else 0), pos, bend

body = Body(mode="lightweight", backend="onnxruntime", device="cpu")
writer, raw_hist, shown_hist, rows, frame_no = None, [], [], [], 0
for ci, clip in enumerate(CLIPS):
    cap = cv2.VideoCapture(SRC + clip); fps = cap.get(5)
    while True:
        ok, full = cap.read()
        if not ok: break
        img = full[:, X0:X1].copy(); h, w = img.shape[:2]
        polys = {b: p * [w, h] for b, p in BANDS.items()}
        t0 = time.perf_counter(); kps, sc = body(img); t_pose = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        per = [person_level(k, s, polys) for k, s in zip(kps, sc)]
        raw = max([p[0] for p in per] or [0]); raw_hist.append(raw)
        shown = max([L for L in (3, 2, 1) if len(raw_hist) >= NEED[L] and min(raw_hist[-NEED[L]:]) >= L] or [0])
        t_rule = (time.perf_counter() - t1) * 1000; shown_hist.append(shown)
        rows.append({"frame": frame_no, "clip": ci, "t": round(frame_no / fps, 2), "people": len(per), "raw": raw, "shown": shown,
                     "pos": max([p[1] for p in per] or [0]), "bend": any(p[2] for p in per), "pose_ms": round(t_pose, 1), "rule_ms": round(t_rule, 3)})
        # 그리기: 띠, 관절, 상단 배너, 하단 단계 기록 막대
        over = img.copy()
        for b in (1, 2, 3): cv2.fillPoly(over, [polys[b].astype(np.int32)], COLOR[b])
        img = cv2.addWeighted(over, 0.25, img, 0.75, 0)
        for k, s in zip(kps, sc):
            for j in range(17):
                if s[j] >= KP_MIN: cv2.circle(img, tuple(int(v) for v in k[j]), 6 if j in (9, 10) else 3, (0, 255, 0) if j in (9, 10) else (255, 255, 0), -1)
        cv2.rectangle(img, (0, 0), (w, 44), COLOR[shown], -1)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil)
        d.text((10, 7), f"{NAME[shown]}   t={frame_no / fps:4.1f}s   포즈 {t_pose:.0f}ms", font=FONT, fill=(255, 255, 255))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        bar = np.full((24, w, 3), 30, np.uint8)
        for i, L in enumerate(shown_hist):
            x = int(i * w / 650); cv2.line(bar, (x, 0), (x, 23), COLOR[L], max(1, w // 650 + 1))
        img = np.vstack([img, bar])
        if writer is None: writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        writer.write(img); frame_no += 1
    cap.release()
writer.release()

with open("frames.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
for ci in range(len(CLIPS)):
    r = [x for x in rows if x["clip"] == ci]
    cnt = {NAME[L]: sum(x["shown"] == L for x in r) for L in range(4)}
    first_stop = next((x["t"] for x in r if x["shown"] == 3), None)
    print(CLIPS[ci], len(r), "frames", cnt, "first STOP t=", first_stop, "no-person frames", sum(x["people"] == 0 for x in r))
print("pose_ms median", np.median([x["pose_ms"] for x in rows]), "rule_ms median", np.median([x["rule_ms"] for x in rows]))
