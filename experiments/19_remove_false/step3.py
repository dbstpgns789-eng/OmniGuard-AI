# 실험 19 규칙 3: 발목이 안 보이면 무릎(무릎 + 허벅지 길이) 또는 엉덩이(엉덩이 + 몸통 1.7배, 몸 방향)로 발 위치를 추정해 규칙 1·2에 쓴다
# 둘 다 안 되면 "모름"으로 정지 유지. 나머지는 step2 그대로
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
src = open(os.path.join(HERE, "step2.py"), encoding="utf-8").read(); exec(src[:src.index('\nprint("== 사고 4건")')])
from calib import NOSE_H
LEG = 1.7
class Floor3(Floor):
    est = 0
    def foot(self, k, s, i):
        f = super().foot(k, s, i)
        if f is not None: return f
        knees = [(kn, hp) for kn, hp in ((13, 11), (14, 12)) if s[kn] >= KP_MIN and s[hp] >= KP_MIN]
        if knees: q = np.mean([k[kn] + (k[kn] - k[hp]) for kn, hp in knees], 0)
        elif min(s[5], s[6], s[11], s[12]) >= KP_MIN:
            hip = (k[11] + k[12]) / 2; sh = (k[5] + k[6]) / 2; q = hip + LEG * (hip - sh)
        else: return None
        Floor3.est += 1; return self.on_floor(self.ray([q], i))[0]

def run_acc(use3):
    for k in VIDEOS:
        labA, tcA = read(SRC["A"].format(k)); labB, tcB = read(SRC["B"].format(k)); tc = min([v for v in (tcA, tcB) if v is not None], default=None)
        P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True))
        T.Z = json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
        T.L = json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json")))["lines"]; F = T.features(P); C = Floor3 if use3 else Floor
        fl = C(T.L, P).fit_scale() if k in CALIB_OK else None; segs = [fl.seg(l) for l in FRONT(k)] if fl else []
        db, npass = gated_db2(P, F, fl, segs, True); m = np.array([g(v) in labA and g(v) in labB for v in F["t"]]); tt = F["t"][m]; o = judge_db(F, db)[m]; line = []
        for gname, y in (("A", np.array([labA[g(v)] for v in tt])), ("B+1", np.minimum(np.array([labB[g(v)] for v in tt]) + 1, 3))):
            gs = y == 3; h0 = float(tt[gs][0]); r = score(tt, o, gs, y < 3, h0, tc, y)
            line.append(f"{gname} 잡음 {r['hit']} 첫정지차 {None if r.get('first_stop') is None else round(r['first_stop'] - h0, 2)} 재현율 {r['stop_recall']}")
        print(k, "규칙1+2+3" if use3 else "규칙1+2", " | ".join(line), flush=True)
print("== 사고 4건"); run_acc(True)
print("== 평소 작업")
OBJ = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values()); L11 = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))["lines"]; T.L = L11
clips = [(os.path.basename(p)[:-9], dict(np.load(p, allow_pickle=True))) for p in sorted(glob.glob(DATA + "exp11/*_pose.npz"))]
hs = []
for _, P in clips: hs += list(Floor(L11, P).fit_scale().hs_raw)
HC = NOSE_H / float(np.median(hs)); total = 0.0; c = 0; Floor3.est = 0
for name, P in clips:
    F = T.features(P, OBJ); t = F["t"]; total += float(t[-1] - t[0]) + 1 / F["fps"]; fl = Floor3(L11, P); fl.Hc = HC; segs = [fl.seg(l) for l in FRONT("press")]
    db, npass = gated_db2(P, F, fl, segs, True); e = episodes(t, judge_db(F, db) == 3, 1.0); c += len(e)
    if e: print(name, "남은 헛정지", [(round(a, 1), round(b_, 1)) for a, b_ in e])
print(f"평소 작업 헛정지: 규칙1+2 6회 → 규칙1+2+3 {c}회 ({c/total*60:.2f}/분), 추정한 발 {Floor3.est}개")
