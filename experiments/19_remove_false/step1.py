# 실험 19 규칙 1 (세훈 확인표): 상체가 위험한계와 겹쳐도 발이 그 설비의 바닥 앞선에서 1.0m보다 멀면 정지하지 않는다
# 무릎·발목이 위험한계 안이면 항상 정지. 바닥 보정이 안 되는 영상이나 발이 안 보이면 "모름" → 정지 그대로
# 출발점은 step0의 S0(모든 관절, 경계 0, 간격 0.25, 유지 1.0초). 손 경로도 같은 문을 통과하도록 손은 몸 거리(db)에 합친다
import os, sys, json, glob, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
src = open(os.path.join(HERE, "step0.py"), encoding="utf-8").read(); exec(src[:src.index('print("== 사고 4건")')])
sys.path.insert(0, os.path.join(X, "14_foot_gate")); from calib import Floor, seg_dist
REACH = 1.0; LEGS = {13, 14, 15, 16}
CALIB_OK = {"paper_rewinder", "press"}  # 실험 14 확인: slitter·night_paper·bag_machine은 보정 실패
FRONT = lambda k: json.load(open(os.path.join(X, "14_foot_gate", f"front_{k}.json")))["lines"]

def gated_db(P, F, fl, segs):
    h = float(P["wh"][1]); db0 = np.full(len(P["t"]), np.nan); db1 = np.full(len(P["t"]), np.nan); last = None; gated = 0
    for i, (K, S) in enumerate(zip(P["kps"], P["sc"])):
        hl = F["hull_at"](i); hl = [hh.astype(np.float32) for hh in (hl if isinstance(hl, list) else [hl])]; b0 = b1 = np.inf
        for k, s in zip(K, S):
            tl = torso_len(k, s) or last or 0.25 * h
            if torso_len(k, s): last = tl
            Fp = fl.foot(k, s, i) if fl is not None else None
            for j in range(17):
                if s[j] < KP_MIN: continue
                ds = [-cv2.pointPolygonTest(hh, (float(k[j][0]), float(k[j][1])), True) / tl for hh in hl]; d = min(ds); b0 = min(b0, d)
                if j in LEGS or Fp is None: b1 = min(b1, d); continue
                near = [seg_dist(Fp, segs[m]) for m in range(len(hl)) if ds[m] <= 0.25 and m < len(segs)] if len(segs) == len(hl) else [min(seg_dist(Fp, sg) for sg in segs)]
                if not near or min(near) <= REACH: b1 = min(b1, d)
                elif d <= 0: gated += 1
        if b0 < np.inf: db0[i] = b0
        if b1 < np.inf: db1[i] = b1
    return db0, db1, gated

def judge_db(F, db):
    p = params(True); p[T.NAMES.index("hand_inside")] = -99.0; F2 = dict(F); F2["db"] = db; return T.judge(F2, p)[0]

print("== 사고 4건")
for k in VIDEOS:
    labA, tcA = read(SRC["A"].format(k)); labB, tcB = read(SRC["B"].format(k)); tc = min([v for v in (tcA, tcB) if v is not None], default=None)
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True))
    T.Z = json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    T.L = json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json")))["lines"]; F = T.features(P)
    fl = Floor(T.L, P).fit_scale() if k in CALIB_OK else None; segs = [fl.seg(l) for l in FRONT(k)] if fl else []
    db0, db1, gated = gated_db(P, F, fl, segs)
    m = np.array([g(v) in labA and g(v) in labB for v in F["t"]]); tt = F["t"][m]
    for name, o in (("S0", judge_db(F, db0)), ("S0+규칙1", judge_db(F, db1))):
        o = o[m]; line = []
        for gname, y in (("A", np.array([labA[g(v)] for v in tt])), ("B+1", np.minimum(np.array([labB[g(v)] for v in tt]) + 1, 3))):
            gs = y == 3; h0 = float(tt[gs][0]); r = score(tt, o, gs, y < 3, h0, tc, y)
            line.append(f"{gname} 잡음 {r['hit']} 첫정지차 {None if r.get('first_stop') is None else round(r['first_stop'] - h0, 2)} 재현율 {r['stop_recall']}")
        print(k, name, " | ".join(line), f"(걸러진 관절 {gated})" if name != "S0" else "", flush=True)

print("== 평소 작업")
OBJ = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values()); L11 = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))["lines"]; T.L = L11
total = 0.0; cnt = {"S0": 0, "S0+규칙1": 0}
for p in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    name = os.path.basename(p)[:-9]; P = dict(np.load(p, allow_pickle=True)); F = T.features(P, OBJ); t = F["t"]; total += float(t[-1] - t[0]) + 1 / F["fps"]
    fl = Floor(L11, P).fit_scale(); segs = [fl.seg(l) for l in FRONT("press")]; db0, db1, gated = gated_db(P, F, fl, segs)
    e0 = episodes(t, judge_db(F, db0) == 3, 1.0); e1 = episodes(t, judge_db(F, db1) == 3, 1.0); cnt["S0"] += len(e0); cnt["S0+규칙1"] += len(e1)
    if e0 or e1: print(name, "S0", [(round(s, 1), round(e, 1)) for s, e in e0], "→ 규칙1", [(round(s, 1), round(e, 1)) for s, e in e1], flush=True)
print(f"평소 작업 {total:.0f}초 헛정지: S0 {cnt['S0']}회 ({cnt['S0']/total*60:.2f}/분) → 규칙1 {cnt['S0+규칙1']}회 ({cnt['S0+규칙1']/total*60:.2f}/분)")
