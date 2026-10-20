# 변수 조합 탐색: 정지 거리, 단계 간격, 몸만 들어와도 정지, 정지 유지 시간, 옷 신호, 가림 확인(G2)의 모든 조합
# 사고 5건(정답 세훈 A, 채우 한 칸 맞춤 B+1)과 프레스 평소 작업(헛정지, 1초 안 재점등은 한 번)으로 채점. 결과 sweep.csv
import os, sys, json, glob, csv, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
src = open(os.path.join(X, "17_reach_cues", "evaluate.py"), encoding="utf-8").read(); exec(src[:src.index("\nR = {}\n")])
NS0 = 0.5
XS, OFFS, BODY, HOLDS, CLOTH, GATE = [-0.93, -0.7, -0.5, -0.25, 0.0, 0.25, 0.5], [0.25, 0.5, 1.0], [True, False], [0.3, 1.0], [False, True], [False, True]

def gate2(out, occ, keep, fps):  # 정지 구간 안에서 가림을 본 순간부터 그 구간 끝까지만 정지, 그 전은 감속
    hk = max(1, int(round(HOLD * fps))); ev = np.array([occ[max(0, i - hk):i + 1].any() for i in range(len(occ))]); o = out.copy(); ok = False
    for i in range(len(o)):
        if o[i] != 3: ok = False; continue
        ok = ok or ev[i]
        if not ok and not keep[i]: o[i] = 2
    return o

def prep(P, F, S):
    up1, ov = cloth_signals(P, F, S, NS0); occ, _, _ = cues(P, F, NS0); return up1, ov, ov_stop(ov, F["fps"]), occ

ACC = []
for k in VIDEOS:
    labA, tcA = read(SRC["A"].format(k)); labB, tcB = read(SRC["B"].format(k)); tc = min([v for v in (tcA, tcB) if v is not None], default=None)
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True)); S = dict(np.load(SEG + f"{k}_seg.npz", allow_pickle=True))
    T.Z = json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    T.L = json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json")))["lines"]; F = T.features(P); t = F["t"]
    m = np.array([g(v) in labA and g(v) in labB for v in t]); tt = t[m]; yA = np.array([labA[g(v)] for v in tt]); yB1 = np.minimum(np.array([labB[g(v)] for v in tt]) + 1, 3)
    ACC.append((k, F, prep(P, F, S), m, tt, yA, yB1, tc)); print("prep", k, flush=True)
OBJ = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values()); T.L = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))["lines"]
PR, total = [], 0.0
for p in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    name = os.path.basename(p)[:-9]; P = dict(np.load(p, allow_pickle=True)); S = dict(np.load(SEG + f"{name}_seg.npz", allow_pickle=True))
    F = T.features(P, OBJ); total += float(F["t"][-1] - F["t"][0]) + 1 / F["fps"]; PR.append((F, prep(P, F, S)))
print("prep press", flush=True)

def system(F, sig, p, cloth, gate):
    up1, ov, ovs, occ = sig; o = T.judge(F, p)[0]
    if cloth: o = apply(o, up1, ov, F["fps"])
    if gate: o = gate2(o, occ, ovs if cloth else np.zeros(len(o), bool), F["fps"])
    return o

rows = []
for x, off, body, hold, cloth, gate in itertools.product(XS, OFFS, BODY, HOLDS, CLOTH, GATE):
    p = list(T.p_red)
    for kk, v in (("hand_inside", x), ("body_inside", x if body else -99.0), ("near_slow", x + off), ("near_warn", x + 2 * off), ("hold_s", hold)): p[T.NAMES.index(kk)] = v
    row = {"정지거리": x, "단계간격": off, "몸만들어와도정지": body, "유지시간": hold, "옷": cloth, "가림확인": gate}; hits = {"A": 0, "B+1": 0}; rec = []; firsts = {}
    for k, F, sig, m, tt, yA, yB1, tc in ACC:
        o = system(F, sig, p, cloth, gate)[m]
        for gname, y in (("A", yA), ("B+1", yB1)):
            gs = y == 3; h0 = float(tt[gs][0]) if gs.any() else None; r = score(tt, o, gs, y < 3, h0, tc, y)
            hits[gname] += bool(r.get("hit")); rec.append(r["stop_recall"] or 0) if gname == "A" else None
            if gname == "A": firsts[k] = None if r.get("first_stop") is None else round(r["first_stop"] - h0, 2)
    fs = sl = 0
    for F, sig in PR:
        o = system(F, sig, p, cloth, gate); fs += len(episodes(F["t"], o == 3, 1.0)); sl += len(episodes(F["t"], o >= 2, 1.0))
    row.update({"사고잡음_세훈": hits["A"], "사고잡음_채우": hits["B+1"], "정지구간재현율_평균": round(float(np.mean(rec)), 3),
                "평소헛정지_분당": round(fs / total * 60, 2), "평소헛감속이상_분당": round(sl / total * 60, 2),
                **{f"첫정지차_{k}": v for k, v in firsts.items()}})
    rows.append(row)
    if len(rows) % 24 == 0: print(len(rows), flush=True)
with open(os.path.join(HERE, "sweep.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("done", len(rows))
