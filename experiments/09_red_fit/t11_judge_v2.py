# 09-11: 판단기 v2. 사용자가 판단표에 남긴 이유를 규칙으로 옮긴다
#   (1) 몸통·팔이 회전축에 가까우면 손과 상관없이 위험: 팔꿈치·어깨가 원통 안으로 깊이 들어가면 정지
#   (2) 손이 가려져 안 보이면: 어깨→팔꿈치 방향으로 팔뚝 길이만큼 연장한 점을 손으로 추정
#   (3) 걷는 중이면(발이 빠르게 움직임) 원근 착시로 보고 정지·감속을 내지 않음
# 정답은 수정 판단표(판단표_판단기비교.csv)에서 사용자가 바꾼 칸을 반영. 7.0초는 보류라 제외
import os, csv, json, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestClassifier
from common import *

Z = json.load(open(os.path.join(HERE, "zone_user.json")))
P = load_pose("performance"); t = P["t"]; n = len(t); fps = 1 / np.median(np.diff(t))
REV = DATA + "exp09/frames_humanlike/판단표_판단기비교.csv"
lab = {round(float(r["시각(초)"]), 1): r["내 판단(기존)"].strip() for r in csv.DictReader(open(REV, encoding="utf-8-sig"))}
lab.update({8.6: "정지", 22.6: "주의", 35.2: "정지"}); lab[7.0] = ""
y = np.array([STAGES.index(lab.get(round(round(x / 0.2) * 0.2, 1), "")) if lab.get(round(round(x / 0.2) * 0.2, 1), "") in STAGES else -1 for x in t])
REASON = {0: "", 1: "가까워지는 중", 2: "손을 넣으려 함", 3: "손이 들어감"}

def hull_at(i): return cv2.convexHull(np.vstack([to_frame(P, i, o) for o in Z["outlines"]]).astype(np.float32)).reshape(-1, 2)

dh, db, foot, inferred, last_tl = np.full(n, np.nan), np.full(n, np.nan), np.full((n, 2), np.nan), np.zeros(n, bool), None
for i, (kps, sc) in enumerate(zip(P["kps"], P["sc"])):
    hull = hull_at(i); best = None
    for k, s in zip(kps, sc):
        tl = torso_len(k, s) or last_tl or 0.25 * P["wh"][1]
        if torso_len(k, s): last_tl = tl
        sd = lambda p: -cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), True) / tl
        hands, inf = [], False
        for wj, ej, sj in ((9, 7, 5), (10, 8, 6)):
            if s[wj] >= KP_MIN: hands.append(sd(k[wj]))
            elif s[ej] >= KP_MIN and s[sj] >= KP_MIN: hands.append(sd(k[ej] + 0.9 * (k[ej] - k[sj]))); inf = True  # (2)
        body = [sd(k[j]) for j in (5, 6, 7, 8) if s[j] >= KP_MIN]
        if not hands: continue
        hmin = min(hands)
        if best is None or hmin < best[0]:
            ft = np.mean([k[j] for j in (15, 16) if s[j] >= KP_MIN], axis=0) / tl if max(s[15], s[16]) >= KP_MIN else np.array([np.nan, np.nan])
            best = (hmin, min(body) if body else np.nan, ft, inf)
    if best: dh[i], db[i], foot[i], inferred[i] = best
gap = int(0.5 * fps)
for arr in (dh, db):
    for i in range(1, n):
        if np.isnan(arr[i]) and not np.isnan(arr[i - 1]) and np.isnan(arr[max(0, i - gap):i]).sum() < gap: arr[i] = arr[i - 1]
# (3) 걷는 속도: 발 위치(몸통 단위)의 0.5초 이동량
k5 = int(0.5 * fps); walk = np.array([np.linalg.norm(foot[i] - foot[i - k5]) / 0.5 if i >= k5 and not np.isnan(foot[i]).any() and not np.isnan(foot[i - k5]).any() else 0.0 for i in range(n)])

def run(p):
    warn, slow, inside, body_in, v_th, walk_th, hold_s = p
    lag = int(fps); v = np.array([(dh[i - lag] - dh[i]) if i >= lag and not np.isnan(dh[i - lag]) and not np.isnan(dh[i]) else 0.0 for i in range(n)])
    st, out, hold = 0, np.zeros(n, int), 0
    for i in range(n):
        d, b = dh[i], db[i]; want = 0
        if not np.isnan(d):
            if d < inside or (not np.isnan(b) and b < body_in): want = 3
            elif d < slow and (v[i] > 0 or d < slow * 0.5): want = 2
            elif d < warn and v[i] > v_th: want = 1
            if walk[i] > walk_th and want >= 2 and not (not np.isnan(b) and b < body_in): want = 1 if v[i] > v_th else 0  # 걷는 중
        if want >= st: st, hold = want, 0
        else:
            hold += 1
            if hold >= int(hold_s * fps): st, hold = st - 1, 0
        out[i] = st
    return out

rng = np.random.default_rng(1); best = None
for _ in range(15000):
    inside = rng.uniform(-0.9, 0.1); slow = inside + rng.uniform(0.05, 1.0); warn = slow + rng.uniform(0.05, 1.5)
    p = (warn, slow, inside, rng.uniform(-1.2, 0.0), rng.uniform(0, 0.6), rng.uniform(0.3, 3.0), rng.choice([0.3, 0.5, 1.0]))
    o = run(p); f1 = macro_f1(y, o)[0]
    if best is None or f1 > best[0]: best = (f1, p)
out = run(best[1])
names = ["near_warn", "near_slow", "hand_inside", "body_inside", "approach_per_s", "walk_speed", "hold_s"]
per = {STAGES[c]: {STAGES[k]: int((out[y == c] == k).sum()) for k in range(4)} for c in range(4)}
yy = np.where(y < 0, 0, y)
res = {"params": dict(zip(names, [round(float(x), 3) for x in best[1]])), "macro_f1": macro_f1(y, out), "per_stage": per,
       "onsets": {"stop": onsets(t, out, 3), "slow": onsets(t, out, 2), "warn": onsets(t, out, 1)},
       "human": {"stop": onsets(t, yy, 3), "slow": onsets(t, yy, 2), "warn": onsets(t, yy, 1)},
       "hand_inferred_frames": int(inferred.sum())}
# 비교: 같은 새 정답으로 v1 판단기(손만)와 분류기
v1 = np.load(OUT + "/t10_humanlike.npy"); res["v1_on_new_labels"] = macro_f1(y, v1)
X = np.nan_to_num(np.c_[dh, db, walk, np.r_[np.zeros(int(fps)), dh[:-int(fps)]] - dh], nan=9.0); m = y >= 0
pc = smooth(RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", random_state=0).fit(X[m], y[m]).predict(X), 5)
res["classifier_on_new_labels"] = {"macro_f1": macro_f1(y, pc), "stop_onsets": onsets(t, pc, 3)}
print(json.dumps(res, ensure_ascii=False, default=float), flush=True)
np.save(OUT + "/t11_judge_v2.npy", out)

COL = {0: (90, 90, 90), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 255), -1: (40, 40, 40)}
FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 26); SM = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 16)
cap = cv2.VideoCapture(DATA + "raw_cctv/rec_002103.mp4"); cap.set(1, int(round(t[0] * cap.get(5)))); w, h = P["wh"]; wr = None
D = DATA + "exp09/frames_judge_v2/"; os.makedirs(D + "img", exist_ok=True); rows = []
for i in range(n):
    ok, img = cap.read()
    if not ok: break
    hull = hull_at(i).astype(np.int32); ov = img.copy(); cv2.fillPoly(ov, [hull], (0, 200, 0)); img = cv2.addWeighted(ov, 0.2, img, 0.8, 0); cv2.polylines(img, [hull], True, (0, 200, 0), 2)
    for k, s in zip(P["kps"][i], P["sc"][i]):
        for j in range(17):
            if s[j] >= KP_MIN: cv2.circle(img, (int(k[j][0]), int(k[j][1])), 7 if j in (9, 10) else 3, (0, 255, 0) if j in (9, 10) else (255, 255, 0), -1)
    lv = int(out[i]); why = REASON[lv]
    if lv == 3 and not np.isnan(db[i]) and db[i] < best[1][3]: why = "몸이 회전축에 붙음"
    if walk[i] > best[1][5]: why += " (걷는 중)"
    if inferred[i]: why += " [손 추정]"
    cv2.rectangle(img, (0, 0), (w, 52), COL[lv], -1)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil)
    dr.text((10, 10), f"{STAGES[lv]}  {why}", font=FONT, fill=(255, 255, 255)); dr.text((w - 230, 16), f"사람: {STAGES[y[i]] if y[i] >= 0 else '-'}  t={t[i]:4.1f}s", font=SM, fill=(255, 255, 255))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR); bars = []
    for lb, arr in (("사람", y), ("판단기v1", v1), ("판단기v2", out)):
        bar = np.full((24, w, 3), 25, np.uint8)
        for j in range(i + 1): x = 80 + int(j * (w - 80) / n); cv2.line(bar, (x, 2), (x, 21), COL[int(arr[j])], max(1, (w - 80) // n + 1))
        p2 = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)); ImageDraw.Draw(p2).text((4, 2), lb, font=SM, fill=(255, 255, 255)); bars.append(cv2.cvtColor(np.array(p2), cv2.COLOR_RGB2BGR))
    frame = np.vstack([img] + bars)
    if wr is None: wr = cv2.VideoWriter(DATA + "exp09/demo_judge_v2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), cap.get(5), (frame.shape[1], frame.shape[0]))
    wr.write(frame)
    s_ = round(round(t[i] / 0.2) * 0.2, 1)
    if abs(t[i] - s_) < 0.5 / fps and (not rows or rows[-1]["시각(초)"] != f"{s_:.1f}"):
        fn = f"{s_:05.1f}s.jpg"; cv2.imencode(".jpg", frame)[1].tofile(D + "img/" + fn)
        hum = STAGES[y[i]] if y[i] >= 0 else "-"
        rows.append({"시각(초)": f"{s_:.1f}", "파일": fn, "내 판단(반영본)": hum, "판단기v2 단계": STAGES[lv], "판단기v2 이유": why, "일치": "O" if hum == STAGES[lv] else "X", "내 판단(수정)": "", "메모": ""})
wr.release()
with open(D + "판단표_v2비교.csv", "w", newline="", encoding="utf-8-sig") as fh:
    wcsv = csv.DictWriter(fh, fieldnames=rows[0].keys()); wcsv.writeheader(); wcsv.writerows(rows)
done("t11_judge_v2", res)
