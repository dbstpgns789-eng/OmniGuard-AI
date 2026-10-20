# 원칙 확인: 몸의 어느 관절이든(17개) 위험한계에 들어오면 정지. 조용한 안(손만)과 비교
# 구현: 판단기의 "몸" 거리(db)를 어깨·팔꿈치 대신 모든 사람·모든 보이는 관절의 최소 거리로 바꾸고 body_inside를 켠다
import os, sys, json, glob, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
src = open(os.path.join(X, "15_two_raters", "evaluate.py"), encoding="utf-8").read(); exec(src[:src.index("R, AGREE, pooled")])
from common import torso_len, onsets
def base():
    p = list(T.p_red)
    for k, v in (("hand_inside", -0.7), ("body_inside", -99.0), ("near_slow", -0.45), ("near_warn", -0.2), ("hold_s", 1.0)): p[T.NAMES.index(k)] = v
    return p
def allpart_db(P, F):
    h = float(P["wh"][1]); out = np.full(len(P["t"]), np.nan); last = None
    for i, (K, S) in enumerate(zip(P["kps"], P["sc"])):
        hl = F["hull_at"](i); hl = [hh.astype(np.float32) for hh in (hl if isinstance(hl, list) else [hl])]; best = np.inf
        for k, s in zip(K, S):
            tl = torso_len(k, s) or last or 0.25 * h
            if torso_len(k, s): last = tl
            for j in range(17):
                if s[j] >= KP_MIN: best = min(best, min(-cv2.pointPolygonTest(hh, (float(k[j][0]), float(k[j][1])), True) for hh in hl) / tl)
        if best < np.inf: out[i] = best
    return out
VARS = {"조용한안(손만)": (None, None), "모든관절_안0.7": ("all", -0.7), "모든관절_경계0": ("all", 0.0)}
def run(F, P, how):
    kind, x = how; p = base()
    if kind is None: return T.judge(F, p)[0]
    F2 = dict(F); F2["db"] = allpart_db(P, F); p[T.NAMES.index("body_inside")] = x; return T.judge(F2, p)[0]
res = {v: {"A": 0, "B+1": 0, "recall": [], "first": {}} for v in VARS}
for k in VIDEOS:
    labA, tcA = read(SRC["A"].format(k)); labB, tcB = read(SRC["B"].format(k)); tc = min([v for v in (tcA, tcB) if v is not None], default=None)
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True))
    T.Z = json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    T.L = json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json")))["lines"]; F = T.features(P); t = F["t"]
    m = np.array([g(v) in labA and g(v) in labB for v in t]); tt = t[m]; yA = np.array([labA[g(v)] for v in tt]); yB1 = np.minimum(np.array([labB[g(v)] for v in tt]) + 1, 3)
    for vn, how in VARS.items():
        o = run(F, P, how)[m]
        for gname, y in (("A", yA), ("B+1", yB1)):
            gs = y == 3; h0 = float(tt[gs][0]) if gs.any() else None; r = score(tt, o, gs, y < 3, h0, tc, y)
            res[vn][gname] += bool(r.get("hit"))
            if gname == "A": res[vn]["recall"].append(r["stop_recall"]); res[vn]["first"][k] = None if r.get("first_stop") is None else round(r["first_stop"] - h0, 2)
OBJ = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values()); T.L = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))["lines"]
total = 0.0; fs = {v: 0 for v in VARS}
for p in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    P = dict(np.load(p, allow_pickle=True)); F = T.features(P, OBJ); total += float(F["t"][-1] - F["t"][0]) + 1 / F["fps"]
    for vn, how in VARS.items(): fs[vn] += len(episodes(F["t"], run(F, P, how) == 3, 1.0))
for vn in VARS:
    r = res[vn]; print(vn, "| 사고 세훈", f"{r['A']}/5", "채우", f"{r['B+1']}/5", "| 재현율", [round(x, 2) for x in r["recall"]], "| 첫정지차", r["first"], "| 평소헛정지/분", round(fs[vn] / total * 60, 2), flush=True)
