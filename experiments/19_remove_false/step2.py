# 실험 19 규칙 2 (세훈 확인표 "안전 구역을 지나쳐 걸어간다"): 규칙 1 위에, 바닥에서 설비와 나란히 1초에 0.5m 이상 걷고 설비 쪽으로 다가오지 않으면 상체 겹침 무시
# 무릎·발목은 항상 정지. 바닥 보정이 안 되는 영상·발이 안 보이는 사람은 그대로. 프레스는 같은 카메라라 높이를 한 번만 추정(3.89m)
import os, sys, json, glob, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
src = open(os.path.join(HERE, "step1.py"), encoding="utf-8").read(); exec(src[:src.index('\nprint("== 사고 4건")')])
from calib import NOSE_H
WALK, ALONG_DEG, APPROACH, WIN = 0.5, 30.0, 0.2, 1.0

def track(P, max_age_s=0.5):  # 화면 중심이 가까운 사람끼리 이어 붙임. 한동안 안 보여도 max_age_s까지는 같은 사람으로 유지
    ids, alive, nxt = [], {}, 0; diag = float(np.hypot(*P["wh"]))
    for t_, K, S in zip(P["t"], P["kps"], P["sc"]):
        alive = {pid: v for pid, v in alive.items() if t_ - v[1] <= max_age_s}; cur = []; used = set()
        for k, s in zip(K, S):
            gm = s >= KP_MIN; c = k[gm].mean(0) if gm.any() else None; best, bd = None, 0.1 * diag
            for pid, (pc, _) in alive.items():
                if c is not None and pid not in used and np.linalg.norm(c - pc) < bd: best, bd = pid, np.linalg.norm(c - pc)
            if best is None: best, nxt = nxt, nxt + 1
            used.add(best); cur.append(best)
            if c is not None: alive[best] = (c, t_)
        ids.append(cur)
    return ids

def gated_db2(P, F, fl, segs, use_pass):
    h = float(P["wh"][1]); n = len(P["t"]); fps = F["fps"]; db = np.full(n, np.nan); last = None; ids = track(P) if fl is not None else None; hist = {}; npass = 0
    for i, (K, S) in enumerate(zip(P["kps"], P["sc"])):
        hl = F["hull_at"](i); hl = [hh.astype(np.float32) for hh in (hl if isinstance(hl, list) else [hl])]; b = np.inf
        for pi, (k, s) in enumerate(zip(K, S)):
            tl = torso_len(k, s) or last or 0.25 * h
            if torso_len(k, s): last = tl
            Fp = fl.foot(k, s, i) if fl is not None else None; passing = {}
            if Fp is not None:
                hh_ = hist.setdefault(ids[i][pi], []); hh_.append((P["t"][i], Fp)); hh_[:] = [x for x in hh_ if x[0] >= P["t"][i] - WIN]
            for j in range(17):
                if s[j] < KP_MIN: continue
                ds = [-cv2.pointPolygonTest(hh, (float(k[j][0]), float(k[j][1])), True) / tl for hh in hl]; d = min(ds)
                if j in LEGS or Fp is None: b = min(b, d); continue
                ms = [m for m in range(len(hl)) if ds[m] <= 0.25] if len(segs) == len(hl) else list(range(len(segs)))
                ok = True
                for m in ms:
                    sg = segs[m] if len(segs) == len(hl) else segs[0]
                    if seg_dist(Fp, sg) > REACH: continue  # 규칙 1
                    if use_pass:
                        if m not in passing:
                            hh_ = hist[ids[i][pi]]; dt = hh_[-1][0] - hh_[0][0]; pz = False
                            if dt >= 0.5 * WIN:
                                v = (hh_[-1][1] - hh_[0][1]) / dt; sp = np.linalg.norm(v); u = (sg[1] - sg[0]) / np.linalg.norm(sg[1] - sg[0])
                                ang = np.degrees(np.arccos(min(1.0, abs(np.dot(v, u)) / (sp + 1e-9)))); appr = (seg_dist(hh_[0][1], sg) - seg_dist(hh_[-1][1], sg)) / dt
                                pz = sp > WALK and ang < ALONG_DEG and appr < APPROACH
                            passing[m] = pz
                        if passing[m]: npass += d <= 0; continue
                    ok = False
                if not ms or not ok: b = min(b, d)
        if b < np.inf: db[i] = b
    return db, npass

print("== 사고 4건")
for k in VIDEOS:
    labA, tcA = read(SRC["A"].format(k)); labB, tcB = read(SRC["B"].format(k)); tc = min([v for v in (tcA, tcB) if v is not None], default=None)
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True))
    T.Z = json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    T.L = json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json")))["lines"]; F = T.features(P)
    fl = Floor(T.L, P).fit_scale() if k in CALIB_OK else None; segs = [fl.seg(l) for l in FRONT(k)] if fl else []
    m = np.array([g(v) in labA and g(v) in labB for v in F["t"]]); tt = F["t"][m]
    for name, up in (("규칙1", False), ("규칙1+2", True)):
        db, npass = gated_db2(P, F, fl, segs, up); o = judge_db(F, db)[m]; line = []
        for gname, y in (("A", np.array([labA[g(v)] for v in tt])), ("B+1", np.minimum(np.array([labB[g(v)] for v in tt]) + 1, 3))):
            gs = y == 3; h0 = float(tt[gs][0]); r = score(tt, o, gs, y < 3, h0, tc, y)
            line.append(f"{gname} 잡음 {r['hit']} 첫정지차 {None if r.get('first_stop') is None else round(r['first_stop'] - h0, 2)} 재현율 {r['stop_recall']}")
        print(k, name, " | ".join(line), f"(지나감으로 걸러진 관절 {npass})", flush=True)

print("== 평소 작업")
OBJ = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values()); L11 = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))["lines"]; T.L = L11
clips = [(os.path.basename(p)[:-9], dict(np.load(p, allow_pickle=True))) for p in sorted(glob.glob(DATA + "exp11/*_pose.npz"))]
hs = []
for _, P in clips: hs += list(Floor(L11, P).fit_scale().hs_raw)
HC = NOSE_H / float(np.median(hs)); total = 0.0; cnt = {"규칙1": 0, "규칙1+2": 0}
for name, P in clips:
    F = T.features(P, OBJ); t = F["t"]; total += float(t[-1] - t[0]) + 1 / F["fps"]; fl = Floor(L11, P); fl.Hc = HC; segs = [fl.seg(l) for l in FRONT("press")]
    for nm, up in (("규칙1", False), ("규칙1+2", True)):
        db, npass = gated_db2(P, F, fl, segs, up); e = episodes(t, judge_db(F, db) == 3, 1.0); cnt[nm] += len(e)
        if e and nm == "규칙1+2": print(name, "남은 헛정지", [(round(a, 1), round(b_, 1)) for a, b_ in e], "지나감으로 걸러진 관절", npass)
print(f"평소 작업 헛정지: 규칙1 {cnt['규칙1']}회 → 규칙1+2 {cnt['규칙1+2']}회 ({cnt['규칙1+2']/total*60:.2f}/분)")
