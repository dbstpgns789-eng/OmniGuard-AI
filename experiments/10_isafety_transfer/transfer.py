# 실험 10: 빨간 상의 확대 영상(rec_002103)에 맞춘 판단기(09 t13, 일치도 0.80)를 처음 보는 넓은 화면(iSafetyBench)에 그대로 돌린다
# 판단기의 규칙·경계값은 바꾸지 않는다(그대로 옮김). 위험한계(원판 2개)와 바닥 선만 이 화면에서 새로 그림
# 비교용 상한: 같은 판단기를 이 영상에 다시 맞춘 결과(과적합). 둘의 차이가 "옮겼을 때 잃는 정도"
# 정답: 사용자 판단표(판단표_isafety.csv). 모자이크 구간·빈칸은 제외. 클립마다 판단기 상태는 처음부터 시작
import os, sys, csv, json, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "09_red_fit"))
from common import STAGES, KP_MIN, torso_len, to_frame, macro_f1, onsets

DATA = "C:/Users/User/Desktop/OmniGuard-data/"; EXP = DATA + "exp10/"; SRC = DATA + "curated/isafety_nip/"
Z = json.load(open(os.path.join(HERE, "zone_user.json"))); Z["outlines"] = [o for o in Z["outlines"] if len(o) >= 10]  # 실수로 생긴 작은 윤곽 제외
L = json.load(open(os.path.join(HERE, "floor_user.json")))["lines"]
PARAMS = json.load(open(os.path.join(HERE, "..", "09_red_fit", "out", "t13_latch", "result.json"), encoding="utf-8"))["params"]
NAMES = ["near_warn", "near_slow", "hand_inside", "body_inside", "approach_per_s", "hold_s", "hand_deep", "walk_speed", "gait_corr", "along_ratio", "approach_speed", "latch_release"]
p_red = [PARAMS[k] for k in NAMES]
labs = {}
for r in csv.DictReader(open(DATA + "exp09/frames_isafety/판단표_isafety.csv", encoding="utf-8-sig")):
    if not r["시각(초)"].strip(): continue
    v = r["단계(정상/주의/감속/정지)"].strip(); labs[(r["클립"], round(float(r["시각(초)"]), 1))] = STAGES.index(v) if v in STAGES else -1
CLIPS = [2, 4, 7, 8]

def features(P, objects=None):
    """09 t12와 같은 특징: 손·몸의 원통 거리, 발 이동(화면 보정 후), 엇박자, 몸 방향
    objects가 주어지면 설비마다 따로 볼록 껍질을 만들고, 거리는 가장 가까운 설비 기준(서로 다른 설비 여러 대일 때)"""
    t = P["t"]; n = len(t); fps = 1 / np.median(np.diff(t)); w, h = P["wh"]
    if objects is None: hull_at = lambda i: cv2.convexHull(np.vstack([to_frame(P, i, o) for o in Z["outlines"]]).astype(np.float32)).reshape(-1, 2)
    else: hull_at = lambda i: [cv2.convexHull(to_frame(P, i, o).astype(np.float32)).reshape(-1, 2) for o in objects]
    def sdist(hull, q, tl):  # 안쪽이면 음수(몸통 단위). 설비 여러 대면 가장 가까운 것
        hs = hull if isinstance(hull, list) else [hull]
        return min(-cv2.pointPolygonTest(hh, (float(q[0]), float(q[1])), True) for hh in hs) / tl
    center_of = lambda hull: np.mean(np.vstack(hull if isinstance(hull, list) else [hull]), axis=0)
    def dir_at(i): q = to_frame(P, i, [L[0], L[1]]); v = q[1] - q[0]; return v / np.linalg.norm(v)
    mref = (np.array(L[1]) - np.array(L[0])) * [w, h]; mref /= np.linalg.norm(mref)
    dh, db, good, last_tl = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool), None
    wr_, an_lr, ori, foot_c, toward = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full((n, 2), np.nan), np.zeros((n, 2))
    for i, (kps, sc) in enumerate(zip(P["kps"], P["sc"])):
        hull = hull_at(i); best = None; mdir = dir_at(i)
        for k, s in zip(kps, sc):
            tl = torso_len(k, s) or last_tl or 0.25 * h
            if torso_len(k, s): last_tl = tl
            sd = lambda q: sdist(hull, q, tl)
            hands = []
            for wj, ej, sj in ((9, 7, 5), (10, 8, 6)):
                if s[wj] >= KP_MIN: hands.append(sd(k[wj]))
                elif s[ej] >= KP_MIN and s[sj] >= KP_MIN and torso_len(k, s): hands.append(sd(k[ej] + 0.9 * (k[ej] - k[sj])))
            if not hands: continue
            body = [sd(k[j]) for j in (5, 6, 7, 8) if s[j] >= KP_MIN]
            if best is None or min(hands) < best[0]: best = (min(hands), min(body) if body else np.nan, k, s, tl, torso_len(k, s) is not None)
        if not best: continue
        dh[i], db[i], k, s, tl, good[i] = best
        if min(s[15], s[16]) >= KP_MIN:
            Hi = np.linalg.inv(P["H"][i]); fr = cv2.perspectiveTransform(((k[15] + k[16]) / 2).reshape(1, 1, 2).astype(np.float64), Hi).reshape(2)
            foot_c[i] = fr / tl; an_lr[i] = np.dot(k[15] - k[16], mdir) / tl
            hc = cv2.perspectiveTransform(center_of(hull).reshape(1, 1, 2).astype(np.float64), Hi).reshape(2); v = hc - fr; v -= np.dot(v, mref) * mref; toward[i] = v / (np.linalg.norm(v) + 1e-9)
        if min(s[9], s[10]) >= KP_MIN: wr_[i] = np.dot(k[9] - k[10], mdir) / tl
        if min(s[5], s[6]) >= KP_MIN: sv = k[5] - k[6]; ori[i] = abs(np.dot(sv / np.linalg.norm(sv), mdir))
    gap = int(0.5 * fps)
    for arr in (dh, db):
        for i in range(1, n):
            if np.isnan(arr[i]) and not np.isnan(arr[i - 1]) and np.isnan(arr[max(0, i - gap):i]).sum() < gap: arr[i] = arr[i - 1]
    W1, W5 = int(fps), int(0.5 * fps); win = lambda a, i, k: (lambda s: s[~np.isnan(s)])(a[max(0, i - k + 1):i + 1])
    gait, speed, along, approach, face = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, np.nan)
    for i in range(n):
        a, b = win(an_lr, i, W1), win(wr_, i, W1); m = min(len(a), len(b))
        if m >= W1 // 2 and np.std(a[-m:]) > 0.05 and np.std(b[-m:]) > 0.05: gait[i] = -np.corrcoef(a[-m:], b[-m:])[0, 1]
        if i >= W5 and not np.isnan(foot_c[i]).any() and not np.isnan(foot_c[i - W5]).any():
            mv = (foot_c[i] - foot_c[i - W5]) / 0.5; speed[i] = np.linalg.norm(mv)
            if speed[i] > 1e-6: along[i] = abs(np.dot(mv, mref)) / speed[i]; approach[i] = np.dot(mv, toward[i])
        o = win(ori, i, W5); face[i] = o.mean() if len(o) else np.nan
    return dict(t=t, fps=fps, dh=dh, db=db, good=good, gait=gait, speed=speed, along=along, approach=approach, face=face, hull_at=hull_at)

def judge(F, p):
    warn, slow, inside, body_in, v_th, hold_s, deep, walk_spd, gait_th, along_th, appr_th, lo = p
    dh, db, good, fps, n = F["dh"], F["db"], F["good"], F["fps"], len(F["t"]); lag = int(fps)
    v = np.array([(dh[i - lag] - dh[i]) if i >= lag and good[i] and good[i - lag] and good[i - lag:i + 1].mean() > 0.8 else 0.0 for i in range(n)])
    st, out, hold, ctx = 0, np.zeros(n, int), 0, []
    for i in range(n):
        sp, al, ap, ga, fa = F["speed"][i], F["along"][i], F["approach"][i], F["gait"][i], F["face"][i]
        passing = sp > walk_spd and al > along_th and (ga > gait_th or (not np.isnan(fa) and fa < 0.5))
        c = "지나감" if passing and ap < appr_th else "다가옴" if ap > appr_th else "설비를 마주봄" if sp < walk_spd * 0.5 and not np.isnan(fa) and fa > 0.6 else ""
        ctx.append(c); d, b = dh[i], db[i]; want = 0
        if not np.isnan(d):
            if d < inside or (not np.isnan(b) and b < body_in): want = 3
            elif d < slow and (v[i] > 0 or d < slow * 0.5): want = 2
            elif (d < warn and v[i] > v_th) or (c == "다가옴" and d < warn + 0.5): want = 1
            if c == "지나감" and not d < deep: want = 0
        latched = st == 3 and not np.isnan(d) and (d < lo or (not np.isnan(b) and b < lo))
        if want >= st: st, hold = want, 0
        elif not latched:
            hold += 1
            if hold >= int(hold_s * fps) or c == "지나감": st, hold = max(want, st - 1), 0
        out[i] = st
    return out, ctx

clips = {}
for n_ in CLIPS:
    P = dict(np.load(EXP + f"trim_{n_:02d}_pose.npz", allow_pickle=True)); F = features(P)
    y = np.array([labs.get((f"trim_{n_:02d}", round(round(x / 0.2) * 0.2, 1)), -1) for x in F["t"]]); clips[n_] = (P, F, y)
if __name__ == "__main__":  # 아래는 직접 실행할 때만 (joint.py가 불러 쓸 때는 건너뜀)
    Y = np.concatenate([c[2] for c in clips.values()])
    def run_all(p): return np.concatenate([judge(F, p)[0] for _, F, _ in clips.values()])
    out_red = run_all(p_red)
    rng = np.random.default_rng(3); best = (macro_f1(Y, out_red)[0], p_red)
    for _ in range(4000):  # 비교용: 이 영상에 다시 맞추기(과적합 상한)
        q = np.array(p_red) + rng.normal(0, 0.2, 12) * [1, 1, 1, 1, 0.3, 0, 1, 1, 0.3, 0.2, 0.5, 1]; q[5] = rng.choice([0.3, 0.5, 1.0])
        f1 = macro_f1(Y, run_all(list(q)))[0]
        if f1 > best[0]: best = (f1, list(q))
    out_fit = run_all(best[1])
    per = lambda o: {STAGES[c]: {STAGES[k]: int((o[Y == c] == k).sum()) for k in range(4)} for c in range(4)}
    res = {"frames_scored": int((Y >= 0).sum()), "transfer_as_is": {"macro_f1": macro_f1(Y, out_red), "per_stage": per(out_red)},
           "refit_here_upper_bound": {"macro_f1": macro_f1(Y, out_fit), "per_stage": per(out_fit), "params": dict(zip(NAMES, [round(float(x), 3) for x in best[1]]))}}
    off = 0
    for n_, (P, F, y) in clips.items():
        o = out_red[off:off + len(y)]; off += len(y)
        res[f"trim_{n_:02d}"] = {"human_stop": onsets(F["t"], np.where(y < 0, 0, y), 3), "judge_stop": onsets(F["t"], o, 3),
                                 "human_slow": onsets(F["t"], np.where(y < 0, 0, y), 2), "judge_slow": onsets(F["t"], o, 2),
                                 "macro_f1": macro_f1(y, o)[0] if (y >= 0).any() else None}
    print(json.dumps(res, ensure_ascii=False, default=float), flush=True)
    json.dump(res, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

    # 데모: 클립 4개를 이어서, 옮긴 판단기(그대로) 결과
    COL = {0: (90, 90, 90), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 255), -1: (40, 40, 40)}
    REASON = {0: "", 1: "가까워지는 중", 2: "손을 넣으려 함", 3: "손이 들어감"}
    FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22); SM = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 15)
    vw = None; os.makedirs(EXP + "frames_demo", exist_ok=True)
    for n_, (P, F, y) in clips.items():
        o, ctx = judge(F, p_red); cap = cv2.VideoCapture(SRC + f"0QHQ2nmdE78_clip0_trim_{n_}.mp4"); a, b = P["crop"]; N = len(F["t"])
        for i in range(N):
            ok, img = cap.read()
            if not ok: break
            img = img[:, a:b].copy(); W = img.shape[1]; hl = F["hull_at"](i).astype(np.int32)
            ov = img.copy(); cv2.fillPoly(ov, [hl], (0, 200, 0)); img = cv2.addWeighted(ov, 0.22, img, 0.78, 0); cv2.polylines(img, [hl], True, (0, 200, 0), 2)
            for k, s in zip(P["kps"][i], P["sc"][i]):
                for j in range(17):
                    if s[j] >= KP_MIN: cv2.circle(img, (int(k[j][0]), int(k[j][1])), 5 if j in (9, 10) else 2, (0, 255, 0) if j in (9, 10) else (255, 255, 0), -1)
            lv = int(o[i]); cv2.rectangle(img, (0, 0), (W, 44), COL[lv], -1)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil)
            dr.text((8, 8), f"{STAGES[lv]} {REASON[lv]}{(' · ' + ctx[i]) if ctx[i] else ''}", font=FONT, fill=(255, 255, 255))
            dr.text((8, 48), f"trim{n_} t={F['t'][i]:.1f}s  사람: {STAGES[y[i]] if y[i] >= 0 else '-'}", font=SM, fill=(255, 255, 255))
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR); bars = []
            for lb, arr in (("사람", y), ("판단기", o)):
                bar = np.full((22, W, 3), 25, np.uint8)
                for j in range(i + 1): x = 55 + int(j * (W - 55) / N); cv2.line(bar, (x, 2), (x, 19), COL[int(arr[j])], max(1, (W - 55) // N + 1))
                p2 = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)); ImageDraw.Draw(p2).text((4, 2), lb, font=SM, fill=(255, 255, 255)); bars.append(cv2.cvtColor(np.array(p2), cv2.COLOR_RGB2BGR))
            frame = np.vstack([img] + bars)
            if vw is None: vw = cv2.VideoWriter(EXP + "demo_transfer.mp4", cv2.VideoWriter_fourcc(*"mp4v"), F["fps"], (frame.shape[1], frame.shape[0])); fsize = (frame.shape[1], frame.shape[0])
            vw.write(cv2.resize(frame, fsize))
            if abs(F["t"][i] * 5 - round(F["t"][i] * 5)) < 0.5 / F["fps"] * 5: cv2.imencode(".jpg", frame)[1].tofile(EXP + f"frames_demo/trim{n_:02d}_{F['t'][i]:04.1f}s.jpg")
    vw.release()
