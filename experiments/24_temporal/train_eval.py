# 실험 24: 관절 시계열(최근 1초, 0.2초 간격 6시점 × 9개 값)로 정지 여부를 배우는 모델. 학습 = 맞춤용 10개 화면 + 프레스 평소 작업 14개, 시험 = 시험용 10건 (README)
import os, sys, json, glob, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
HERE = os.path.dirname(os.path.abspath(__file__)); X_ = os.path.join(HERE, ".."); sys.argv = sys.argv[:1]
src = open(os.path.join(X_, "21_first_test", "evaluate.py"), encoding="utf-8").read(); exec(src[:src.index("res = {}")])
sys.path.insert(0, os.path.join(X, "09_red_fit")); from common import KP_MIN
STEPS, DT, NEAR, AWAY_S = 6, 0.2, 0.5, 1.0

def frame_feats(P, F):
    lost = np.array([not any(int((np.asarray(s) >= KP_MIN).sum()) >= 3 for s in sc) for sc in P["sc"]])
    dall = J.db_rule4(P, F, None)  # 17개 관절 중 가장 가까운 거리
    cols = [F["dh"], F["db"], dall, F["speed"], F["along"], F["approach"], F["gait"], F["face"], lost.astype(float)]
    M = np.stack([np.nan_to_num(np.asarray(c, float), nan=9.0) for c in cols], 1); M[:, 7] = np.where(np.isnan(F["face"]), -1, F["face"])
    near = lost | (np.nan_to_num(F["dh"], nan=9) < NEAR) | (np.nan_to_num(F["db"], nan=9) < NEAR)
    return M, near

def windows(M, fps):
    k = max(1, int(round(DT * fps))); n = len(M)
    return np.concatenate([M[np.clip(np.arange(n) - j * k, 0, n - 1)] for j in range(STEPS)], 1)

def hold(base, near, fps):  # 실험 22 C1과 같은 정지 유지
    out = base.copy(); on = False; away = 0
    for i in range(len(out)):
        if base[i] == 3: on, away = True, 0
        elif on:
            away = 0 if near[i] else away + 1
            if away >= int(AWAY_S * fps): on = False
        if on: out[i] = 3
    return out

def clip_data(name):
    P = dict(np.load(POSE + f"{name}_pose.npz", allow_pickle=True)); lab, tc, hs = labels(name); zone, objects, floor, fronts = geometry(name)
    F = J.features(P, zone, floor, objects); t = F["t"]; M, near = frame_feats(P, F)
    y = np.array([lab.get(round(round(v / 0.2) * 0.2, 1), -1) for v in t]); return dict(W=windows(M, F["fps"]), y=y, t=t, tc=tc, near=near, fps=F["fps"])

clips = {n: clip_data(n) for n in sorted(x for x in os.listdir(LAB) if os.path.isdir(LAB + x))}
objs = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values())
floor = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json"))); press = []
for p in sorted(glob.glob(DATA + "exp20_stab/*_pose.npz")):
    if not os.path.basename(p)[0].isdigit(): continue
    P = dict(np.load(p, allow_pickle=True)); J.T.L = floor["lines"]; F = J.T.features(P, objs); M, _ = frame_feats(P, F); press.append(windows(M, F["fps"]))
print("loaded", len(clips), "clips,", len(press), "press clips", flush=True)

def train(names):
    Xs = [clips[n]["W"][clips[n]["y"] >= 0][::2] for n in names] + [w[::3] for w in press]
    ys = [(clips[n]["y"][clips[n]["y"] >= 0][::2] == 2).astype(int) for n in names] + [np.zeros(len(w[::3]), int) for w in press]
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=0).fit(np.concatenate(Xs), np.concatenate(ys))

def score(model, names):
    out = {}; tot = {"raw": [0, 0, 0], "hold": [0, 0, 0]}; mins = 0
    for n in names:
        c = clips[n]; base = np.where(model.predict_proba(c["W"])[:, 1] >= 0.5, 3, 0); win = (c["t"] >= c["tc"] - 0.5) & (c["t"] <= c["tc"])
        mins += (c["y"] == 0).sum() / c["fps"] / 60; out[n] = {}
        for kind, o in (("raw", base), ("hold", hold(base, c["near"], c["fps"]))):
            hit = bool(win.any() and (o[win] == 3).all()); fs = len(episodes(c["t"], (o == 3) & (c["y"] == 0))); ns = int(((o == 3) & (c["y"] == 0)).sum())
            out[n][kind] = {"hit": hit, "false_stops": fs, "normal_stop_frac": round(ns / max(1, (c["y"] == 0).sum()), 2)}
            tot[kind][0] += hit; tot[kind][1] += fs; tot[kind][2] += ns
    nf = sum((clips[n]["y"] == 0).sum() for n in names)
    summ = {k: {"stop_rate": f"{v[0]}/{len(names)}", "false_per_min": round(v[1] / mins, 2) if mins else None, "normal_stop_time": f"{v[2] / max(1, nf):.0%}"} for k, v in tot.items()}
    return summ, out

TUNE = [n for n in clips if n not in TEST]; TESTN = [n for n in clips if n in TEST]
groups = [["A09_pipe", "A10_pipe_view2"]] + [[n] for n in TUNE if n not in ("A09_pipe", "A10_pipe_view2")]
cv = {}
for g in groups:  # 맞춤용 안에서 사고 하나씩 빼고 학습해 그 사고로 평가 (같은 사고의 두 화면은 함께 뺌)
    _, o = score(train([n for n in TUNE if n not in g]), g); cv.update(o)
cv_hits = {k: sum(o[k]["hit"] for o in cv.values()) for k in ("raw", "hold")}
print("교차 검증 (맞춤용 10개 화면):", {k: f"{v}/{len(cv)}" for k, v in cv_hits.items()}, flush=True)
model = train(TUNE); summ, out = score(model, TESTN)
for n, o in out.items(): print(n, o, flush=True)
print("시험 10건:", json.dumps(summ, ensure_ascii=False))
json.dump({"cv": cv, "cv_hits": cv_hits, "test_summary": summ, "test_clips": out}, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
