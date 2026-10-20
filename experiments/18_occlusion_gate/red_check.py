# 대표 판단기(조용한 안)를 빨간 상의 두 각도(확대 화면 09, 넓은 화면 10 클립 4개)에 돌려 세훈 판단표로 채점. 비교: J09, J13
import os, sys, json, csv, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
src = open(os.path.join(X, "15_two_raters", "evaluate.py"), encoding="utf-8").read(); exec(src[:src.index("R, AGREE, pooled")])
QUIET = list(T.p_red)
for k, v in (("hand_inside", -0.7), ("body_inside", -99.0), ("near_slow", -0.45), ("near_warn", -0.2), ("hold_s", 1.0)): QUIET[T.NAMES.index(k)] = v
J = {"J09": T.p_red, "J13": P_RULE, "조용한안": QUIET}
def lab_red(t):
    lab = {round(float(r["시각(초)"]), 1): r["내 판단(기존)"].strip() for r in csv.DictReader(open(DATA + "exp09/frames_humanlike/판단표_판단기비교.csv", encoding="utf-8-sig"))}
    lab.update({8.6: "정지", 22.6: "주의", 35.2: "정지"}); lab[7.0] = ""; f = lambda x: lab.get(g(x), "")
    return np.array([STAGES.index(f(x)) if f(x) in STAGES else -1 for x in t])
sets = []
T.Z = json.load(open(X + "/09_red_fit/zone_user.json")); T.L = json.load(open(X + "/09_red_fit/floor_user.json"))["lines"]
P = dict(np.load(DATA + "exp09/red_crop_pose_performance.npz", allow_pickle=True)); F = T.features(P); sets.append(("빨간상의_확대", F, lab_red(F["t"])))
T.Z = json.load(open(X + "/10_isafety_transfer/zone_user.json")); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]; T.L = json.load(open(X + "/10_isafety_transfer/floor_user.json"))["lines"]
for n_, (P, F, y) in T.clips.items(): sets.append((f"넓은화면_{n_}", F, y))
out = {}
for name, F, y in sets:
    m = y >= 0; tt = F["t"][m]; yy = y[m]; gs = yy == 3; h0 = float(tt[gs][0]) if gs.any() else None
    for jn, p in J.items():
        o = T.judge(F, p)[0][m]; r = score(tt, o, gs, yy < 3, h0, None, yy)
        out.setdefault(jn, []).append((name, r)); print(name, jn, {k: r.get(k) for k in ("hit", "first_stop", "margin_vs_human_s", "stop_recall", "false_stops", "normal_min")}, flush=True)
json.dump({j: [(n, r) for n, r in v] for j, v in out.items()}, open(os.path.join(HERE, "red_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
