# 09-12: 판단기 v3. 사용자가 말한 "걷는 사람" 단서를 관절로 재서 맥락(지나감 / 돌아섬 / 다가옴)을 판단에 넣는다
#   걸음: 최근 1초 손 흔들림과 발 흔들림이 엇박자(반대 손·반대 발)이고 발이 움직임
#   몸 방향: 어깨선이 설비 방향(바닥 선 1·2)과 이루는 각. 설비를 마주 보면 어깨선이 설비와 나란함
#   발 방향: 최근 0.5초 발 이동이 설비 방향과 나란하면 "지나감", 원통 쪽이면 "다가옴"
# 규칙: 지나가는 중이면 화면상 겹침으로 생긴 주의·감속·정지를 무시하되, 손이 원통 안 깊숙이(hand_deep) 들어가면 걷는 중이어도 정지
# 판단기 v2(몸통 규칙, 손 추정, 단계 유지)는 그대로 쓰고 맥락만 덧붙인다. 정답은 v2와 같은 반영본
import os, sys, csv, json, numpy as np, cv2
LATCH = "--latch" in sys.argv; LATCH_OUT = 0.0  # 원통 표면(0) 밖으로 나와야 해제
from PIL import Image, ImageDraw, ImageFont
from common import *

Z = json.load(open(os.path.join(HERE, "zone_user.json"))); FL = json.load(open(os.path.join(HERE, "floor_user.json")))
P = load_pose("performance"); t = P["t"]; n = len(t); fps = 1 / np.median(np.diff(t)); w, h = P["wh"]
REV = DATA + "exp09/frames_humanlike/판단표_판단기비교.csv"
lab = {round(float(r["시각(초)"]), 1): r["내 판단(기존)"].strip() for r in csv.DictReader(open(REV, encoding="utf-8-sig"))}
lab.update({8.6: "정지", 22.6: "주의", 35.2: "정지"}); lab[7.0] = ""
y = np.array([STAGES.index(lab.get(round(round(x / 0.2) * 0.2, 1), "")) if lab.get(round(round(x / 0.2) * 0.2, 1), "") in STAGES else -1 for x in t])

def hull_at(i): return cv2.convexHull(np.vstack([to_frame(P, i, o) for o in Z["outlines"]]).astype(np.float32)).reshape(-1, 2)
def dir_at(i, a, b):  # 기준 프레임의 바닥 선(화면 비율) → 이 프레임 화면에서의 단위 방향
    q = to_frame(P, i, [a, b]); v = q[1] - q[0]; return v / np.linalg.norm(v)
L = FL["lines"]
def dir_at_ref(a, b): v = (np.array(b) - np.array(a)) * [w, h]; return v / np.linalg.norm(v)

# 프레임마다 가장 가까운 사람의 값
dh, db, last_tl = np.full(n, np.nan), np.full(n, np.nan), None
good = np.zeros(n, bool)  # 몸통까지 보이는 믿을 만한 측정인가
ank, wr_, an_lr, ori, axis_dir, foot_c, toward = (np.full((n, 2), np.nan), np.full((n, 2), np.nan), np.full((n, 2), np.nan),
                                                 np.full(n, np.nan), np.zeros((n, 2)), np.full((n, 2), np.nan), np.zeros((n, 2)))
for i, (kps, sc) in enumerate(zip(P["kps"], P["sc"])):
    hull = hull_at(i); best = None; mdir = dir_at(i, L[0], L[1])
    for k, s in zip(kps, sc):
        tl = torso_len(k, s) or last_tl or 0.25 * h
        if torso_len(k, s): last_tl = tl
        sd = lambda p: -cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), True) / tl
        hands = []
        for wj, ej, sj in ((9, 7, 5), (10, 8, 6)):
            if s[wj] >= KP_MIN: hands.append(sd(k[wj]))
            elif s[ej] >= KP_MIN and s[sj] >= KP_MIN and torso_len(k, s): hands.append(sd(k[ej] + 0.9 * (k[ej] - k[sj])))  # 몸통이 보일 때만 손 추정
        if not hands: continue
        body = [sd(k[j]) for j in (5, 6, 7, 8) if s[j] >= KP_MIN]
        if best is None or min(hands) < best[0]: best = (min(hands), min(body) if body else np.nan, k, s, tl, torso_len(k, s) is not None)
    if not best: continue
    dh[i], db[i], k, s, tl, good[i] = best
    axis_dir[i] = mdir
    if min(s[15], s[16]) >= KP_MIN:
        fr = cv2.perspectiveTransform(((k[15] + k[16]) / 2).reshape(1, 1, 2).astype(np.float64), np.linalg.inv(P["H"][i])).reshape(2)  # 카메라가 따라 움직인 만큼 되돌림
        foot_c[i] = fr / tl
        an_lr[i] = [np.dot(k[15] - k[16], mdir) / tl, 0]  # 두 발의 앞뒤 차이(설비 방향 성분)
        hc = cv2.perspectiveTransform(hull.mean(0).reshape(1, 1, 2).astype(np.float64), np.linalg.inv(P["H"][i])).reshape(2)
        v = hc - fr; mref = dir_at_ref(L[0], L[1]); v -= np.dot(v, mref) * mref; toward[i] = v / (np.linalg.norm(v) + 1e-9)  # 발에서 원통으로 수직 방향
    if min(s[9], s[10]) >= KP_MIN: wr_[i] = [np.dot(k[9] - k[10], mdir) / tl, 0]  # 두 손의 앞뒤 차이
    if min(s[5], s[6]) >= KP_MIN:
        sv = k[5] - k[6]; ori[i] = abs(np.dot(sv / np.linalg.norm(sv), mdir))  # 1: 어깨선이 설비와 나란함(설비를 마주봄 또는 등짐), 0: 직각(설비 방향으로 걷는 자세)

gap = int(0.5 * fps)
for arr in (dh, db):
    for i in range(1, n):
        if np.isnan(arr[i]) and not np.isnan(arr[i - 1]) and np.isnan(arr[max(0, i - gap):i]).sum() < gap: arr[i] = arr[i - 1]
W1, W5 = int(fps), int(0.5 * fps)
def win(a, i, k): seg = a[max(0, i - k + 1):i + 1]; return seg[~np.isnan(seg)]
gait, speed, along, approach, face = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, np.nan)
for i in range(n):
    a, b = win(an_lr[:, 0], i, W1), win(wr_[:, 0], i, W1)
    m = min(len(a), len(b))
    if m >= W1 // 2 and np.std(a[-m:]) > 0.05 and np.std(b[-m:]) > 0.05:
        gait[i] = -np.corrcoef(a[-m:], b[-m:])[0, 1]  # 엇박자면 +1 쪽
    if i >= W5 and not np.isnan(foot_c[i]).any() and not np.isnan(foot_c[i - W5]).any():
        mv = (foot_c[i] - foot_c[i - W5]) / 0.5; speed[i] = np.linalg.norm(mv)
        if speed[i] > 1e-6:
            along[i] = abs(np.dot(mv, dir_at_ref(L[0], L[1]))) / speed[i]; approach[i] = np.dot(mv, toward[i])  # 원통 쪽 속도(몸통/초)
    o = win(ori, i, W5); face[i] = o.mean() if len(o) else np.nan

def context(p, i):
    walk_spd, gait_th, along_th, appr_th = p[7:11]
    passing = speed[i] > walk_spd and along[i] > along_th and (gait[i] > gait_th or (not np.isnan(face[i]) and face[i] < 0.5))
    if passing and approach[i] < appr_th: return "지나감"
    if approach[i] > appr_th: return "다가옴"
    if speed[i] < walk_spd * 0.5 and not np.isnan(face[i]) and face[i] > 0.6: return "설비를 마주봄"
    return ""

def run(p):
    warn, slow, inside, body_in, v_th, hold_s, deep = p[:7]
    lag = int(fps); v = np.array([(dh[i - lag] - dh[i]) if i >= lag and good[i] and good[i - lag] and good[i - lag:i + 1].mean() > 0.8 else 0.0 for i in range(n)])
    st, out, hold, ctx = 0, np.zeros(n, int), 0, []
    for i in range(n):
        d, b = dh[i], db[i]; want = 0; c = context(p, i); ctx.append(c)
        if not np.isnan(d):
            if d < inside or (not np.isnan(b) and b < body_in): want = 3
            elif d < slow and (v[i] > 0 or d < slow * 0.5): want = 2
            elif (d < warn and v[i] > v_th) or (c == "다가옴" and d < warn + 0.5): want = 1
            if c == "지나감" and not d < deep: want = 0  # 지나가는 중: 겹침 무시, 손이 깊으면 정지 유지
        lo = p[11] if len(p) > 11 else LATCH_OUT
        latched = st == 3 and LATCH and not np.isnan(d) and (d < lo or (not np.isnan(b) and b < lo))  # 정지 유지: 손·몸이 원통 밖으로 나올 때까지
        if want >= st: st, hold = want, 0
        elif not latched:
            hold += 1
            if hold >= int(hold_s * fps) or c == "지나감": st, hold = max(want, st - 1), 0
        out[i] = st
    return out, ctx

v2p = json.load(open(OUT + "/t11_judge_v2/result.json", encoding="utf-8"))["params"]
base = [v2p["near_warn"], v2p["near_slow"], v2p["hand_inside"], v2p["body_inside"], v2p["approach_per_s"], v2p["hold_s"]]
rng = np.random.default_rng(2); best = None
for it in range(12000):
    if it == 0: p = base + [-1.2, 1.0, 0.3, 0.6, 0.5, 0.0]
    else:
        q = np.array(base) + rng.normal(0, 0.15, 6) * [1, 1, 1, 1, 0.5, 0]; q[5] = rng.choice([0.3, 0.5, 1.0])
        p = list(q) + [q[2] - rng.uniform(0.0, 0.8), rng.uniform(0.3, 2.5), rng.uniform(-0.2, 0.6), rng.uniform(0.4, 0.95), rng.uniform(0.1, 1.5), rng.uniform(-0.6, 0.2)]
    o, _ = run(p); f1 = macro_f1(y, o)[0]
    if best is None or f1 > best[0]: best = (f1, p)
out, ctx = run(best[1]); v2 = np.load(OUT + "/t11_judge_v2.npy")
names = ["near_warn", "near_slow", "hand_inside", "body_inside", "approach_per_s", "hold_s", "hand_deep", "walk_speed", "gait_corr", "along_ratio", "approach_speed", "latch_release"]
per = lambda o: {STAGES[c]: {STAGES[k]: int((o[y == c] == k).sum()) for k in range(4)} for c in range(4)}
yy = np.where(y < 0, 0, y)
probe = {f"{x}": {"ctx": ctx[int(np.argmin(np.abs(t - x)))], "speed": round(float(speed[int(np.argmin(np.abs(t - x)))]), 2), "along": round(float(along[int(np.argmin(np.abs(t - x)))]), 2),
                  "approach": round(float(approach[int(np.argmin(np.abs(t - x)))]), 2), "gait": round(float(gait[int(np.argmin(np.abs(t - x)))]), 2),
                  "face": None if np.isnan(face[int(np.argmin(np.abs(t - x)))]) else round(float(face[int(np.argmin(np.abs(t - x)))]), 2),
                  "stage": STAGES[out[int(np.argmin(np.abs(t - x)))]]} for x in (25.8, 26.6, 27.4, 27.8, 28.4, 29.0, 29.2, 29.4, 30.0)}
res = {"params": dict(zip(names, [round(float(x), 3) for x in best[1]])), "macro_f1_v3": macro_f1(y, out), "macro_f1_v2": macro_f1(y, v2),
       "per_stage_v3": per(out), "per_stage_v2": per(v2), "onsets_v3": {"stop": onsets(t, out, 3), "slow": onsets(t, out, 2), "warn": onsets(t, out, 1)},
       "human": {"stop": onsets(t, yy, 3), "slow": onsets(t, yy, 2), "warn": onsets(t, yy, 1)}, "probe_user_times": probe,
       "ctx_counts": {c: ctx.count(c) for c in set(ctx)}}
print(json.dumps(res, ensure_ascii=False, default=float), flush=True)
np.save(OUT + ("/t13_latch.npy" if LATCH else "/t12_context.npy"), out)

COL = {0: (90, 90, 90), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 255), -1: (40, 40, 40)}
REASON = {0: "", 1: "가까워지는 중", 2: "손을 넣으려 함", 3: "손이 들어감"}
FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 24); SM = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 16)
cap = cv2.VideoCapture(DATA + "raw_cctv/rec_002103.mp4"); cap.set(1, int(round(t[0] * cap.get(5)))); vw = None
D = DATA + ("exp09/frames_context_latch/" if LATCH else "exp09/frames_context/"); os.makedirs(D + "img", exist_ok=True); rows = []
for i in range(n):
    ok, img = cap.read()
    if not ok: break
    hull = hull_at(i).astype(np.int32); ov = img.copy(); cv2.fillPoly(ov, [hull], (0, 200, 0)); img = cv2.addWeighted(ov, 0.2, img, 0.8, 0); cv2.polylines(img, [hull], True, (0, 200, 0), 2)
    for k, s in zip(P["kps"][i], P["sc"][i]):
        for j in range(17):
            if s[j] >= KP_MIN: cv2.circle(img, (int(k[j][0]), int(k[j][1])), 7 if j in (9, 10) else 3, (0, 255, 0) if j in (9, 10) else (255, 255, 0), -1)
    lv = int(out[i]); why = REASON[lv] + (f" · {ctx[i]}" if ctx[i] else "")
    cv2.rectangle(img, (0, 0), (w, 52), COL[lv], -1)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil)
    dr.text((10, 10), f"{STAGES[lv]}  {why}", font=FONT, fill=(255, 255, 255)); dr.text((w - 230, 16), f"사람: {STAGES[y[i]] if y[i] >= 0 else '-'}  t={t[i]:4.1f}s", font=SM, fill=(255, 255, 255))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR); bars = []
    for lb, arr in (("사람", y), ("판단기v2", v2), ("v3+유지" if LATCH else "판단기v3", out)):
        bar = np.full((24, w, 3), 25, np.uint8)
        for j in range(i + 1): x = 80 + int(j * (w - 80) / n); cv2.line(bar, (x, 2), (x, 21), COL[int(arr[j])], max(1, (w - 80) // n + 1))
        p2 = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)); ImageDraw.Draw(p2).text((4, 2), lb, font=SM, fill=(255, 255, 255)); bars.append(cv2.cvtColor(np.array(p2), cv2.COLOR_RGB2BGR))
    frame = np.vstack([img] + bars)
    if vw is None: vw = cv2.VideoWriter(DATA + ("exp09/demo_context_latch.mp4" if LATCH else "exp09/demo_context.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cap.get(5), (frame.shape[1], frame.shape[0]))
    vw.write(frame)
    s_ = round(round(t[i] / 0.2) * 0.2, 1)
    if abs(t[i] - s_) < 0.5 / fps and (not rows or rows[-1]["시각(초)"] != f"{s_:.1f}"):
        fn = f"{s_:05.1f}s.jpg"; cv2.imencode(".jpg", frame)[1].tofile(D + "img/" + fn); hum = STAGES[y[i]] if y[i] >= 0 else "-"
        rows.append({"시각(초)": f"{s_:.1f}", "파일": fn, "내 판단": hum, "v3 단계": STAGES[lv], "v3 이유·맥락": why, "일치": "O" if hum == STAGES[lv] else "X",
                     "걷기 속도": round(float(speed[i]), 2), "발-설비방향 나란함": round(float(along[i]), 2), "원통쪽 접근": round(float(approach[i]), 2),
                     "엇박자": round(float(gait[i]), 2), "몸이 설비와 나란함": "" if np.isnan(face[i]) else round(float(face[i]), 2), "내 판단(수정)": "", "메모": ""})
vw.release()
import time as _t
with open(D + f"판단표_v3비교_{_t.strftime('%m%d_%H%M')}.csv", "w", newline="", encoding="utf-8-sig") as fh:  # 엑셀로 열어 둔 파일과 겹치지 않게 새 이름
    wc = csv.DictWriter(fh, fieldnames=rows[0].keys()); wc.writeheader(); wc.writerows(rows)
done("t13_latch" if LATCH else "t12_context", res)
