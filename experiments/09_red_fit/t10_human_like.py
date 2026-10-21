# 09-10: 안전관리자처럼 "흐름"을 보고 이유를 말하는 판단기(상태 기계). 학습 없이 규칙 몇 개 + 임계값만 사람 판단에 맞춤
#   주의: 손(또는 몸)이 원통에 가까운 범위 안에 있고, 최근 1초 동안 원통 쪽으로 다가오는 중        → "가까워지는 중"
#   감속: 손이 원통 표면 근처까지 왔고, 원통 쪽으로 움직이는 중이거나 그 근처에 머무름             → "손을 넣으려 함"
#   정지: 손이 원통(위험한계) 안쪽으로 들어감                                                   → "손이 들어감"
#   올릴 때는 바로, 내릴 때는 HOLD초 동안 조건이 안 맞아야 한 단계씩 내린다(사람이 경계를 늦게 푸는 것처럼)
# 거리는 사용자가 그린 원통(2D, 몸통 단위). 정지 직전 사람을 놓친 프레임은 직전 값을 0.5초까지 이어 쓴다
import os, sys, json, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from common import *

Z = json.load(open(os.path.join(HERE, "zone_user.json")))
P = load_pose("performance"); t = P["t"]; y = load_labels(t); n = len(t); fps = 1 / np.median(np.diff(t))
REASON = {0: "", 1: "가까워지는 중", 2: "손을 넣으려 함", 3: "손이 들어감"}

def hull_at(i): return cv2.convexHull(np.vstack([to_frame(P, i, o) for o in Z["outlines"]]).astype(np.float32)).reshape(-1, 2)

# 프레임마다 가장 가까운 사람의 손 거리(dh)와 몸 거리(db, 코·어깨 중 가까운 것). 원통 안이면 음수, 몸통 단위
dh, db, last_tl = np.full(n, np.nan), np.full(n, np.nan), None
for i, (kps, sc) in enumerate(zip(P["kps"], P["sc"])):
    hull = hull_at(i)
    for k, s in zip(kps, sc):
        tl = torso_len(k, s) or last_tl or 0.25 * P["wh"][1]
        if torso_len(k, s): last_tl = tl
        d = lambda j: -cv2.pointPolygonTest(hull, (float(k[j][0]), float(k[j][1])), True) / tl if s[j] >= KP_MIN else np.nan
        hand = np.nanmin([d(9), d(10), 99]); body = np.nanmin([d(0), d(5), d(6), 99])
        if hand < 99 and (np.isnan(dh[i]) or hand < dh[i]): dh[i], db[i] = hand, body
gap = int(0.5 * fps)  # 사람을 잠깐 놓치면 직전 값을 이어 씀
for arr in (dh, db):
    for i in range(1, n):
        if np.isnan(arr[i]) and i - np.max(np.where(~np.isnan(arr[:i]))[0], initial=-10 ** 9) <= gap: arr[i] = arr[i - 1]
sm = lambda a, k: np.array([np.nanmean(a[max(0, i - k + 1):i + 1]) if not np.all(np.isnan(a[max(0, i - k + 1):i + 1])) else np.nan for i in range(n)])

def run(p):
    near_warn, near_slow, inside, v_th, hold_s, k_s = p
    k = max(1, int(k_s * fps)); dh_s = sm(dh, k); lag = int(1.0 * fps)
    v = np.array([(dh_s[i - lag] - dh_s[i]) if i >= lag and not np.isnan(dh_s[i - lag]) and not np.isnan(dh_s[i]) else 0.0 for i in range(n)])  # 1초 동안 가까워진 양
    st, out, why, hold = 0, np.zeros(n, int), [], 0
    for i in range(n):
        d = dh_s[i]; want = 0
        if not np.isnan(d):
            if d < inside: want = 3
            elif d < near_slow and (v[i] > 0 or d < near_slow * 0.5): want = 2
            elif d < near_warn and v[i] > v_th: want = 1
        if want >= st: st, hold = want, 0
        else:
            hold += 1
            if hold >= int(hold_s * fps): st, hold = st - 1, 0
        out[i] = st
    return out, v

rng = np.random.default_rng(0); best = None
for _ in range(6000):
    inside = rng.uniform(-0.8, 0.2); near_slow = inside + rng.uniform(0.05, 1.0); near_warn = near_slow + rng.uniform(0.05, 1.5)
    p = (near_warn, near_slow, inside, rng.uniform(0.0, 0.6), rng.choice([0.3, 0.5, 1.0, 1.5]), rng.choice([0.1, 0.2, 0.3]))
    out, _ = run(p); f1 = macro_f1(y, out)[0]
    if best is None or f1 > best[0]: best = (f1, p)
out, v = run(best[1])
first, second = t < 27, t >= 27
res = {"params": dict(zip(["near_warn", "near_slow", "inside", "approach_per_s", "hold_s", "smooth_s"], [round(float(x), 3) for x in best[1]])),
       "macro_f1": macro_f1(y, out), "macro_f1_first_half": macro_f1(np.where(first, y, -1), out), "macro_f1_second_half": macro_f1(np.where(second, y, -1), out),
       "stop_onsets": onsets(t, out, 3), "slow_onsets": onsets(t, out, 2), "warn_onsets": onsets(t, out, 1),
       "human": {"stop": onsets(t, np.where(y < 0, 0, y), 3), "slow": onsets(t, np.where(y < 0, 0, y), 2), "warn": onsets(t, np.where(y < 0, 0, y), 1)}}
per = {}
for c in range(4):
    sel = (y == c); per[STAGES[c]] = {STAGES[k]: int((out[sel] == k).sum()) for k in range(4)}
res["per_stage"] = per; print(json.dumps(res, ensure_ascii=False, default=float), flush=True)
np.save(OUT + "/t10_humanlike.npy", out)

# 데모: 원통 + 단계 + 판단 이유 + 손 거리 그래프
COL = {0: (90, 90, 90), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 255), -1: (40, 40, 40)}
FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 26); SM = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 16)
cap = cv2.VideoCapture(DATA + "raw_cctv/rec_002103.mp4"); cap.set(1, int(round(t[0] * cap.get(5)))); w, h = P["wh"]; wr = None
for i in range(n):
    ok, img = cap.read()
    if not ok: break
    hull = hull_at(i).astype(np.int32); ov = img.copy(); cv2.fillPoly(ov, [hull], (0, 200, 0)); img = cv2.addWeighted(ov, 0.2, img, 0.8, 0); cv2.polylines(img, [hull], True, (0, 200, 0), 2)
    for k, s in zip(P["kps"][i], P["sc"][i]):
        for j in range(17):
            if s[j] >= KP_MIN: cv2.circle(img, (int(k[j][0]), int(k[j][1])), 7 if j in (9, 10) else 3, (0, 255, 0) if j in (9, 10) else (255, 255, 0), -1)
    lv = int(out[i]); cv2.rectangle(img, (0, 0), (w, 52), COL[lv], -1)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil)
    dr.text((10, 10), f"{STAGES[lv]}  {REASON[lv]}", font=FONT, fill=(255, 255, 255))
    dr.text((w - 230, 16), f"사람: {STAGES[y[i]] if y[i] >= 0 else '-'}  t={t[i]:4.1f}s", font=SM, fill=(255, 255, 255))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR); bars = []
    for lab, arr in (("사람", y), ("판단기", out)):
        bar = np.full((24, w, 3), 25, np.uint8)
        for j in range(i + 1): x = 70 + int(j * (w - 70) / n); cv2.line(bar, (x, 2), (x, 21), COL[int(arr[j])], max(1, (w - 70) // n + 1))
        p2 = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)); ImageDraw.Draw(p2).text((4, 2), lab, font=SM, fill=(255, 255, 255)); bars.append(cv2.cvtColor(np.array(p2), cv2.COLOR_RGB2BGR))
    gr = np.full((60, w, 3), 25, np.uint8); ys = lambda d: int(np.clip(30 - d * 20, 2, 57))  # 손-원통 거리(위가 가까움)
    cv2.line(gr, (70, ys(0)), (w, ys(0)), (0, 200, 0), 1)
    for j in range(1, i + 1):
        if not np.isnan(dh[j]) and not np.isnan(dh[j - 1]): cv2.line(gr, (70 + int((j - 1) * (w - 70) / n), ys(dh[j - 1])), (70 + int(j * (w - 70) / n), ys(dh[j])), COL[int(out[j])], 2)
    p3 = Image.fromarray(cv2.cvtColor(gr, cv2.COLOR_BGR2RGB)); ImageDraw.Draw(p3).text((4, 20), "손 거리", font=SM, fill=(255, 255, 255)); bars.append(cv2.cvtColor(np.array(p3), cv2.COLOR_RGB2BGR))
    frame = np.vstack([img] + bars)
    if wr is None: wr = cv2.VideoWriter(DATA + "exp09/demo_humanlike.mp4", cv2.VideoWriter_fourcc(*"mp4v"), cap.get(5), (frame.shape[1], frame.shape[0]))
    wr.write(frame)
    if i in (int(n * 0.07), int(n * 0.6), int(n * 0.72), int(n * 0.76)): cv2.imwrite(DATA + f"exp09/demo_humanlike_{i}.jpg", frame)
wr.release()
done("t10_human_like", res)
