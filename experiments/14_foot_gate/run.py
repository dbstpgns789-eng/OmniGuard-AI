# 실험 14: 겹침이 진짜인지 확인하는 정보를 하나씩 쌓는다 (설계: README.md, 경계값은 실행 전에 ISO 13855에서 정함)
#   A0 V1: 손목이 위험한계 안이면 정지(3프레임 확인)          A1: 실험 09 판단기(깊은 경계값), 실험 13 판단기(원칙 경계값)
#   S2: 겹침 + 발이 바닥 앞선에서 1.0m 안 → 정지, 겹침인데 발이 밖·안 보임 → 감속, 발 1.65m 안 → 주의
#   S21: S2 + 안전 증거(바닥에서 설비와 나란히 걸어 지나감, 또는 꼿꼿이 서서 손이 어깨 아래)면 정지 → 감속
#   S214: S21 + 정지는 최근 0.2초 간격 3스텝 중 2스텝, 해제는 1초 동안 모든 사람의 발이 1.3m 밖
# 주 결과: 바닥 보정이 된 영상. 보정이 안 된 사고 3건(slitter, night_paper, bag_machine)은 화각 30·50·70°를 가정한 참고 결과
import os, sys, csv, json, glob, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
for d in ("09_red_fit", "10_isafety_transfer"): sys.path.insert(0, os.path.join(X, d))
import transfer as T
from common import STAGES, DATA, KP_MIN, torso_len, to_frame, onsets
from calib import Floor, seg_dist

REACH, WARN, RELEASE, WALK, ALONG_DEG, APPROACH = 1.0, 1.65, 1.3, 0.5, 30.0, 0.2
P_RED = list(T.p_red); P_RULE = list(T.p_red)
for k, v in (("near_warn", 1.0), ("near_slow", 0.5), ("hand_inside", 0.0), ("body_inside", 0.0)): P_RULE[T.NAMES.index(k)] = v

def hands_body(k, s):  # 손(안 보이면 팔꿈치로 추정)과 어깨·팔꿈치
    pts = []
    for wj, ej, sj in ((9, 7, 5), (10, 8, 6)):
        if s[wj] >= KP_MIN: pts.append(k[wj])
        elif s[ej] >= KP_MIN and s[sj] >= KP_MIN and torso_len(k, s): pts.append(k[ej] + 0.9 * (k[ej] - k[sj]))
    return pts + [k[j] for j in (5, 6, 7, 8) if s[j] >= KP_MIN]

def track(P):  # 화면 중심이 가까운 사람끼리 이어 붙이는 단순 추적
    ids, prev, nxt = [], [], 0; diag = float(np.hypot(*P["wh"]))
    for K, S in zip(P["kps"], P["sc"]):
        cur = []
        for k, s in zip(K, S):
            g = s >= KP_MIN; c = k[g].mean(0) if g.any() else None; best, bd = None, 0.1 * diag
            for pid, pc in prev:
                if c is not None and pc is not None and pid not in [x[0] for x in cur] and np.linalg.norm(c - pc) < bd: best, bd = pid, np.linalg.norm(c - pc)
            if best is None: best, nxt = nxt, nxt + 1
            cur.append((best, c))
        ids.append([x[0] for x in cur]); prev = cur
    return ids

def judge14(P, F, fl, fronts):
    t = P["t"]; n = len(t); fps = 1 / np.median(np.diff(t)); segs = [fl.seg(l) for l in fronts]; ids = track(P)
    hulls = lambda i: (lambda h: h if isinstance(h, list) else [h])(F["hull_at"](i))
    lv = {v: np.zeros(n, int) for v in ("V1", "S2", "S21")}; s2_stop, near_any, hist = np.zeros(n, bool), np.zeros(n, bool), {}
    foot_unknown_overlap = 0; overlap_frames = 0
    for i, (K, S) in enumerate(zip(P["kps"], P["sc"])):
        hs = [hh.astype(np.float32) for hh in hulls(i)]; inside = lambda q, idx=None: any(cv2.pointPolygonTest(hh, (float(q[0]), float(q[1])), False) >= 0 for j, hh in enumerate(hs) if idx is None or j == idx)
        for pid, k, s in zip(ids[i], K, S):
            if any(inside(k[j]) for j in (9, 10) if s[j] >= KP_MIN): lv["V1"][i] = 3
            hb = hands_body(k, s); Fp = fl.foot(k, s, i)  # 설비가 여러 대면 겹친 설비와 그 설비의 앞선을 짝지음
            objs = [j for j in range(len(hs)) if any(inside(q, j) for q in hb)] if len(hs) == len(segs) else ([0] if any(inside(q) for q in hb) else [])
            over = bool(objs); pick = objs if len(hs) == len(segs) and objs else range(len(segs))
            d = None if Fp is None else min(seg_dist(Fp, segs[j]) for j in pick)
            if d is not None:
                hist.setdefault(pid, []).append((i, Fp, d))
                if d <= RELEASE: near_any[i] = True
            a = 3 if over and d is not None and d <= REACH else 2 if over else 1 if d is not None and d <= WARN else 0
            if over: overlap_frames += 1; foot_unknown_overlap += d is None
            lv["S2"][i] = max(lv["S2"][i], a)
            if a == 3:  # 안전 증거
                h = [x for x in hist.get(pid, []) if x[0] >= i - int(0.5 * fps)]; passing = False
                if len(h) >= 3 and h[-1][0] > h[0][0]:
                    dt = (h[-1][0] - h[0][0]) / fps; v = (h[-1][1] - h[0][1]) / dt; sp = np.linalg.norm(v)
                    ang = np.degrees(np.arccos(min(1.0, abs(np.dot(v, fl.along)) / (sp + 1e-9)))); appr = (h[0][2] - h[-1][2]) / dt
                    passing = sp > WALK and ang < ALONG_DEG and appr < APPROACH
                sh, hp = (k[5] + k[6]) / 2, (k[11] + k[12]) / 2
                upright = min(s[5], s[6], s[11], s[12]) >= KP_MIN and abs(np.degrees(np.arctan2((sh - hp)[0], -(sh - hp)[1]))) < 15 \
                    and all(k[w][1] > min(k[5][1], k[6][1]) for w in (9, 10) if s[w] >= KP_MIN)
                if passing or upright: a = 2
            lv["S21"][i] = max(lv["S21"][i], a)
    conf = lambda a: np.array([a[i] if a[i] < 3 or (i >= 2 and (a[i - 2:i + 1] == 3).all()) else 2 for i in range(n)])  # 정지는 3프레임 연속
    out = {"V1": conf(lv["V1"]), "S2": conf(lv["S2"]), "S21": conf(lv["S21"])}
    k3 = max(1, int(round(0.2 * fps))); a = lv["S21"]; o4 = np.zeros(n, int); latched, clear = False, 0
    for i in range(n):  # 머무름: 0.2초 간격 3스텝 중 2스텝, 해제는 1초 동안 모두 1.3m 밖
        steps = [a[j] == 3 for j in (i, i - k3, i - 2 * k3) if j >= 0]
        if sum(steps) >= 2: latched = True
        clear = clear + 1 if not near_any[i] else 0
        if latched and clear >= int(fps): latched = False
        o4[i] = 3 if latched else min(a[i], 2)
    out["S214"] = o4
    out["A1_red"] = T.judge(F, P_RED)[0]; out["A1_rule"] = T.judge(F, P_RULE)[0]
    return out, {"overlap_foot_unknown": round(foot_unknown_overlap / max(1, overlap_frames), 3)}

def score(t, fps, out, y):
    m = y >= 0; hs = next((t[i] for i in range(len(y)) if y[i] == 3), None); r = {}
    for v, o in out.items():
        js = next((t[i] for i in range(len(o)) if o[i] == 3 and y[i] >= 0), None)
        r[v] = {"stop_recall": round(float((o[y == 3] == 3).mean()), 3) if (y == 3).any() else None,
                "slow_or_stop_on_human_stop": round(float((o[y == 3] >= 2).mean()), 3) if (y == 3).any() else None,
                "false_stop_ratio": round(float((o[m & (y < 3)] == 3).mean()), 3) if (m & (y < 3)).any() else None,
                "first_stop_minus_human": None if hs is None or js is None else round(float(js - hs), 2),
                "stop_onsets": len(onsets(t, np.where(m, o, 0), 3))}
    r["_"] = {"human_first_stop": None if hs is None else round(float(hs), 2), "scored_s": round(float(m.sum() / fps), 1)}
    return r

# 정답 (실험 13과 같은 규칙)
def lab_red(t):
    lab = {round(float(r["시각(초)"]), 1): r["내 판단(기존)"].strip() for r in csv.DictReader(open(DATA + "exp09/frames_humanlike/판단표_판단기비교.csv", encoding="utf-8-sig"))}
    lab.update({8.6: "정지", 22.6: "주의", 35.2: "정지"}); lab[7.0] = ""; g = lambda x: lab.get(round(round(x / 0.2) * 0.2, 1), "")
    return np.array([STAGES.index(g(x)) if g(x) in STAGES else -1 for x in t])
def lab12(k, t):
    rows = list(csv.DictReader(open(DATA + f"exp08/frames_for_review/{k}/판단표.csv", encoding="utf-8-sig")))
    seq = [(round(float(r["시각(초)"]), 1), r["단계(정상/주의/감속/정지)"].strip()) for r in rows if r["시각(초)"].strip()]
    last = max(i for i, (_, v) in enumerate(seq) if v in STAGES); f, cur = {}, ""
    for i, (s, v) in enumerate(seq):
        if v in STAGES: cur = v
        f[s] = cur if i <= last else ""
    g = lambda x: f.get(round(round(x / 0.2) * 0.2, 1), "")
    return np.array([STAGES.index(g(x)) if g(x) in STAGES else -1 for x in t])
front = lambda k: json.load(open(os.path.join(HERE, f"front_{k}.json")))["lines"]
def run(name, P, F, floor_lines, fronts, y, fov=None):
    if "H" not in P: P["H"] = np.repeat(np.eye(3)[None], len(P["t"]), 0)
    fl = Floor(floor_lines, P, fov).fit_scale(); out, extra = judge14(P, F, fl, fronts)
    r = score(P["t"], 1 / np.median(np.diff(P["t"])), out, y); r["_"].update(extra, camera_h=round(fl.Hc, 2)); return r

R = {}
T.Z = json.load(open(X + "/09_red_fit/zone_user.json")); T.L = json.load(open(X + "/09_red_fit/floor_user.json"))["lines"]
P = dict(np.load(DATA + "exp09/red_crop_pose_performance.npz", allow_pickle=True))
R["red"] = run("red", P, T.features(P), T.L, front("red"), lab_red(P["t"])); print("red", flush=True)
T.Z = json.load(open(X + "/10_isafety_transfer/zone_user.json")); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]; T.L = json.load(open(X + "/10_isafety_transfer/floor_user.json"))["lines"]
for n_, (P, F, y) in T.clips.items(): R[f"wide_{n_}"] = run("wide", P, F, T.L, front("wide"), y)
print("wide", flush=True)
OBJ = list(json.load(open(X + "/11_press_normal/zone_objects.json", encoding="utf-8"))["objects"].values()); L11 = json.load(open(X + "/11_press_normal/floor_user.json"))["lines"]; T.L = L11
for p in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    P = dict(np.load(p, allow_pickle=True)); F = T.features(P, OBJ); R["press_" + os.path.basename(p)[:-9]] = run("press", P, F, L11, front("press"), np.zeros(len(P["t"]), int))
print("press", flush=True)
for k in ("paper_rewinder", "conveyor_parcel", "slitter", "night_paper", "bag_machine"):
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True))
    T.Z = json.load(open(X + f"/12_accident_set/zone_{k}.json")); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    T.L = json.load(open(X + f"/12_accident_set/floor_{k}.json"))["lines"]; F = T.features(P); y = lab12(k, F["t"])
    for fov in ((None,) if k in ("paper_rewinder", "conveyor_parcel") else (30, 50, 70)):
        R[k + ("" if fov is None else f"_fov{fov}")] = run(k, dict(P), F, T.L, front(k), y, fov)
    print(k, flush=True)
json.dump(R, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
