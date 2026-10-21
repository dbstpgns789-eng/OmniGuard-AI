# 실험 21 채점: 고정한 판단기 3개(20_frozen/judges.py)를 시험 영상 20개에 한 번 돌려 README의 지표로 잰다
# 사용: python evaluate.py          (화면 보정 켬, 1차 시험)
#       python evaluate.py nostab   (화면 보정 끔, 나중 비교용)
import os, sys, csv, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(X, "20_frozen")); import judges as J
DATA = "C:/Users/User/Desktop/OmniGuard-data/"; LAB = DATA + "세훈_작업/판단표_사고19/"; POSE = DATA + "exp21/"
STAB = "nostab" not in sys.argv[1:]
RULES = json.load(open(LAB + "제외구간.json", encoding="utf-8")); SPLIT = json.load(open(os.path.join(X, "20_frozen", "split.json"), encoding="utf-8"))
TEST = set(SPLIT["test"]); NAMES3 = ["정상", "주의", "정지"]

def labels(name):
    """판단표 → {0.2초 시각: 단계(0 정상, 1 주의, 2 정지)}, 닿음 시각, 세훈 첫 정지. 제외 구간·마지막 기록 뒤는 뺀다"""
    rows = list(csv.reader(open(LAB + f"{name}/판단표.csv", encoding="utf-8-sig")))[1:]
    shift = name in RULES["칸보정"]  # A10: 단계가 닿음 칸에, 닿음이 메모 칸에 밀려 있음
    st_col, y_col = (3, 4) if shift else (2, 3)
    ex = RULES["제외"].get(name); cur = 0; lab, tc, hs = {}, None, None
    last = max(i for i, r in enumerate(rows) if any(v.strip() for v in r[2:]))
    for i, r in enumerate(rows[:last + 1]):
        t = round(float(r[0]), 1); s = r[st_col].strip() if len(r) > st_col else ""; y = r[y_col].strip().upper() if len(r) > y_col else ""
        if s in NAMES3: cur = NAMES3.index(s)
        if ex and ex[0] <= t < ex[1]: continue
        lab[t] = cur
        if y == "Y" and tc is None: tc = t
        if cur == 2 and hs is None: hs = t
    return lab, tc, hs

def geometry(name):
    Z = json.load(open(LAB + f"{name}/위험한계.json")); F = json.load(open(LAB + f"{name}/앞선.json"))
    roll = Z.get("roll") or [0] * len(Z["outlines"]); groups = sorted({k for o, k in zip(Z["outlines"], roll) if len(o) >= 10})
    outs = [[o for o, k in zip(Z["outlines"], roll) if k == g and len(o) >= 10] for g in groups]
    zone = {"outlines": outs[0]}; objects = None if len(outs) == 1 else [[q for o in g for q in o] for g in outs]
    return zone, objects, {"lines": F["lines"][0]}, F["lines"]

def episodes(t, on, merge=1.0):
    eps = []
    for i in np.flatnonzero(on):
        if eps and t[i] - eps[-1][1] <= merge: eps[-1][1] = t[i]
        else: eps.append([t[i], t[i]])
    return eps

def macro_f1(y, o):
    f = []
    for c in range(3):
        tp = ((o == c) & (y == c)).sum(); fp = ((o == c) & (y != c)).sum(); fn = ((o != c) & (y == c)).sum()
        if tp + fn: f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))

res = {}
for name in sorted(os.listdir(LAB)):
    if not os.path.isdir(LAB + name): continue
    P = dict(np.load(POSE + f"{name}_pose.npz", allow_pickle=True))
    if not STAB: P["H"] = np.repeat(np.eye(3)[None], len(P["t"]), 0)
    lab, tc, hs = labels(name); zone, objects, floor, fronts = geometry(name); F = J.features(P, zone, floor, objects); t = F["t"]
    key = np.array([round(round(v / 0.2) * 0.2, 1) for v in t]); m = np.array([k in lab for k in key]); y = np.array([lab.get(k, -1) for k in key])
    r = {"split": "test" if name in TEST else "tune", "contact": tc, "human_first_stop": hs, "normal_sec": round(float((y == 0).sum() / F["fps"]), 2),
         "pose_ms_median": round(float(np.median(P["ms"])), 1)}
    for jn in J.PARAMS:
        o = J.run(jn, P, F, fronts)
        win = (t >= tc - 0.5) & (t <= tc); hit = bool(win.any() and (o[win] == 3).all())
        fs = len(episodes(t, (o == 3) & (y == 0)))
        js = next((float(t[i]) for i in range(len(t)) if o[i] == 3 and m[i]), None)
        o3 = np.minimum(np.where(o == 3, 2, np.where(o == 2, 1, o)), 2)  # 판단기 감속은 주의로
        r[jn] = {"hit": hit, "false_stops": fs, "first_stop_diff": None if js is None or hs is None else round(js - hs, 2),
                 "acc3": round(float((o3[m] == y[m]).mean()), 3), "f1_3": round(macro_f1(y[m], o3[m]), 3)}
    res[name] = r
    print(name, r["split"], {jn: (r[jn]["hit"], r[jn]["false_stops"], r[jn]["first_stop_diff"]) for jn in J.PARAMS}, flush=True)

summary = {}
for part, keep in (("all20", lambda r: True), ("test10", lambda r: r["split"] == "test"), ("tune10", lambda r: r["split"] == "tune")):
    rs = [r for r in res.values() if keep(r)]; mins = sum(r["normal_sec"] for r in rs) / 60
    summary[part] = {"clips": len(rs), "normal_min": round(mins, 2), **{jn: {
        "stop_rate": f"{sum(r[jn]['hit'] for r in rs)}/{len(rs)}", "false_per_min": round(sum(r[jn]["false_stops"] for r in rs) / mins, 2) if mins else None,
        "false_stops": sum(r[jn]["false_stops"] for r in rs),
        "first_stop_diff_median": float(np.median([r[jn]["first_stop_diff"] for r in rs if r[jn]["first_stop_diff"] is not None] or [np.nan])),
        "f1_3_mean": round(float(np.mean([r[jn]["f1_3"] for r in rs])), 3)} for jn in J.PARAMS}}
summary["pose_ms_median_all"] = round(float(np.median([r["pose_ms_median"] for r in res.values()])), 1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
json.dump({"stabilization": STAB, "summary": summary, "clips": res}, open(os.path.join(HERE, f"result{'' if STAB else '_nostab'}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
