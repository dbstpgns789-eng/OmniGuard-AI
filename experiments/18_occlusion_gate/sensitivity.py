# 경계값 민감도: J13의 정지 거리를 0(지금)에서 ISO 13855식 안전 거리 쪽으로 넓히면 사고 잡음과 평소 작업 헛정지가 어떻게 변하나
# 정지 거리 x(몸통 길이 단위, 몸통 ≈ 0.5m), 감속 x+0.5, 주의 x+1.0. 나머지 규칙은 J13 그대로
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
src = open(os.path.join(X, "15_two_raters", "evaluate.py"), encoding="utf-8").read(); exec(src[:src.index("R, AGREE, pooled")])
from common import onsets
XS = [-0.93, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
def params(x):
    p = list(T.p_red)
    for k, v in (("hand_inside", x), ("body_inside", x), ("near_slow", x + 0.5), ("near_warn", x + 1.0)): p[T.NAMES.index(k)] = v
    return p
acc = {}
for k in VIDEOS:
    labA, tcA = read(SRC["A"].format(k)); labB, tcB = read(SRC["B"].format(k)); tc = min([v for v in (tcA, tcB) if v is not None], default=None)
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True))
    T.Z = json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    T.L = json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json")))["lines"]; F = T.features(P); t = F["t"]
    m = np.array([g(v) in labA and g(v) in labB for v in t]); tt = t[m]; yA = np.array([labA[g(v)] for v in tt]); yB1 = np.minimum(np.array([labB[g(v)] for v in tt]) + 1, 3)
    for x in XS:
        o = T.judge(F, params(x))[0][m]
        for gname, y in (("A", yA), ("B+1", yB1)):
            gs = y == 3; h0 = float(tt[gs][0]) if gs.any() else None
            acc.setdefault((x, gname), []).append(score(tt, o, gs, y < 3, h0, tc, y))
OBJ = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values()); T.L = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))["lines"]
press, total = {x: [0, 0] for x in XS}, 0.0
for p in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    P = dict(np.load(p, allow_pickle=True)); F = T.features(P, OBJ); t = F["t"]; total += float(t[-1] - t[0]) + 1 / F["fps"]
    for x in XS:
        o = T.judge(F, params(x))[0]; press[x][0] += len(onsets(t, o, 3)); press[x][1] += len(episodes(t, o == 3, 1.0))
rows = []
for x in XS:
    rA, rB = acc[(x, "A")], acc[(x, "B+1")]
    rows.append({"stop_dist_torso": x, "approx_m": round(x * 0.5, 2), "hits_A": f"{sum(r['hit'] for r in rA)}/5", "hits_B+1": f"{sum(r['hit'] for r in rB)}/5",
                 "margin_vs_A_s": [r.get("margin_vs_human_s") for r in rA], "stop_recall_A": [r["stop_recall"] for r in rA],
                 "press_stop_onsets": press[x][0], "press_stop_episodes_merged1s": press[x][1], "press_per_min": round(press[x][1] / total * 60, 2)})
    print(rows[-1], flush=True)
json.dump(rows, open(os.path.join(HERE, "sensitivity.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
