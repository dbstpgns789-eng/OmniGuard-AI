# 실험 28: 포즈 품질 필터(뼈 길이, 보이는 관절 수, 평균 신뢰도)를 적용한 뒤 J13·C1을 개발 데이터와 시험·검증 영상에 다시 잰다
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); X_ = os.path.join(HERE, "..")
src = open(os.path.join(X_, "24_temporal", "train_eval.py"), encoding="utf-8").read(); exec(src[:src.index("clips = {n: clip_data(n)")])
sys.path.insert(0, os.path.join(X, "09_red_fit")); from common import torso_len
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]; BONE_MAX, MIN_J, MIN_MEAN = 1.5, 6, 0.4

def filt(P):
    K2, S2, dropped = [], [], 0
    for K, S in zip(P["kps"], P["sc"]):
        ks, ss = [], []
        for k, s in zip(K, S):
            k = np.asarray(k, float); s = np.asarray(s, float).copy(); tl = torso_len(k, s)
            if tl:
                for a, b in BONES:
                    if s[a] >= KP_MIN and s[b] >= KP_MIN and np.linalg.norm(k[a] - k[b]) > BONE_MAX * tl: s[b] = 0.0
            vis = s >= KP_MIN
            if vis.sum() < MIN_J or s[vis].mean() < MIN_MEAN: dropped += 1; continue
            ks.append(k); ss.append(s)
        K2.append(np.array(ks).reshape(-1, 17, 2)); S2.append(np.array(ss).reshape(-1, 17))
    Q = dict(P); Q["kps"] = np.array(K2, dtype=object); Q["sc"] = np.array(S2, dtype=object); return Q, dropped

def judge_pair(P, F):
    M, near = frame_feats(P, F); j = J.run("J13", P, F); return j, hold(j, near, F["fps"])

# 1) 개발 데이터 (실험 20 check_dev와 같은 채점)
dev = {}; _episodes = episodes
s0 = open(os.path.join(X, "19_remove_false", "step0.py"), encoding="utf-8").read(); exec(s0[:s0.index('print("== 사고 4건")')])
for use in (False, True):
    hits, fs = 0, 0
    for k in VIDEOS:
        labA, tcA = read(SRC["A"].format(k)); labB, tcB = read(SRC["B"].format(k)); tc = min([v for v in (tcA, tcB) if v is not None], default=None)
        P = dict(np.load(DATA + f"exp20_stab/{k}_pose.npz", allow_pickle=True)); P = filt(P)[0] if use else P
        F = J.features(P, json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))), json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json"))))
        m = np.array([g(v) in labA and g(v) in labB for v in F["t"]]); tt = F["t"][m]; o = J.run("J13", P, F)[m]; y = np.array([labA[g(v)] for v in tt])
        gs = y == 3; hits += bool(score(tt, o, gs, y < 3, float(tt[gs][0]), tc, y)["hit"])
    objs = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values()); floor = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))
    for p in sorted(glob.glob(DATA + "exp20_stab/*_pose.npz")):
        if not os.path.basename(p)[0].isdigit(): continue
        P = dict(np.load(p, allow_pickle=True)); P = filt(P)[0] if use else P; J.T.L = floor["lines"]; F = J.T.features(P, objs); fs += len(episodes(F["t"], J.run("J13", P, F) == 3, 1.0))
    dev["filtered" if use else "raw"] = {"J13_accidents": f"{hits}/4", "J13_press_false_stops": fs}; print("개발", "필터" if use else "원본", dev["filtered" if use else "raw"], flush=True)

# 2) 시험·검증 영상
episodes = _episodes
res = {"raw": {}, "filtered": {}}; drops = {}
for n in sorted(x for x in os.listdir(LAB) if os.path.isdir(LAB + x)):
    P0 = dict(np.load(POSE + f"{n}_pose.npz", allow_pickle=True)); lab, tc, hs = labels(n); zone, objects, floor, fronts = geometry(n)
    for key, P in (("raw", P0), ("filtered", None)):
        if P is None: P, drops[n] = filt(P0)
        F = J.features(P, zone, floor, objects); t = F["t"]; y = np.array([lab.get(round(round(v / 0.2) * 0.2, 1), -1) for v in t]); win = (t >= tc - 0.5) & (t <= tc)
        j, c = judge_pair(P, F); r = {}
        for nm, o in (("J13", j), ("C1", c)):
            r[nm] = {"hit": bool(win.any() and (o[win] == 3).all()), "false_stops": len(episodes(t, (o == 3) & (y == 0))), "ns": int(((o == 3) & (y == 0)).sum()), "nf": int((y == 0).sum()), "min": (y == 0).sum() / F["fps"] / 60}
        res[key][n] = r
summ = {}
for key in res:
    for part, keep in (("test10", lambda n: n in TEST), ("tune10", lambda n: n not in TEST)):
        for nm in ("J13", "C1"):
            rs = [v[nm] for n, v in res[key].items() if keep(n)]; mins = sum(r["min"] for r in rs)
            summ[f"{key}/{part}/{nm}"] = {"stop_rate": f"{sum(r['hit'] for r in rs)}/{len(rs)}", "false_per_min": round(sum(r["false_stops"] for r in rs) / mins, 2), "normal_stop_time": f"{sum(r['ns'] for r in rs) / max(1, sum(r['nf'] for r in rs)):.0%}"}
for k, v in summ.items(): print(k, v)
print("버린 사람 검출 수(영상별):", drops)
json.dump({"dev": dev, "summary": summ, "dropped": drops, "clips": res}, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
