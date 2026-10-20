# 실험 19 규칙 4 (세훈 판단 "tr38·tr54·te6·tr86 모두 안전"): 발이 설비 바닥 앞선의 바깥(사람 쪽)이면 머리·어깨·엉덩이 겹침 무시,
# 손·팔꿈치는 조용한 안 깊이(안쪽 0.7)까지 들어와야 정지. 무릎·발목은 항상 정지. 발이 앞선 안쪽이거나 알 수 없으면 모든 관절 경계 0(원칙)
# 앞선의 안/밖은 화면에서 판단: 앞선 직선을 기준으로 위험한계 중심과 같은 쪽이면 안쪽
import os, sys, json, glob, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
src = open(os.path.join(HERE, "step1.py"), encoding="utf-8").read(); exec(src[:src.index('\nprint("== 사고 4건")')])
from calib import NOSE_H
ARMS, DEEP = {7, 8, 9, 10}, -0.7

def side(p, a, b): return np.sign((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))

def db4(P, F, fl, fronts_px):
    h = float(P["wh"][1]); n = len(P["t"]); db = np.full(n, np.nan); last = None; cnt = {"원칙": 0, "밖": 0}
    for i, (K, S) in enumerate(zip(P["kps"], P["sc"])):
        hl = F["hull_at"](i); hl = [hh.astype(np.float32) for hh in (hl if isinstance(hl, list) else [hl])]; b = np.inf
        for k, s in zip(K, S):
            tl = torso_len(k, s) or last or 0.25 * h
            if torso_len(k, s): last = tl
            feet = [k[j] for j in (15, 16) if s[j] >= KP_MIN]; fp = np.mean(feet, 0) if feet else None
            for j in range(17):
                if s[j] < KP_MIN: continue
                ds = [-cv2.pointPolygonTest(hh, (float(k[j][0]), float(k[j][1])), True) / tl for hh in hl]; m = int(np.argmin(ds)); d = ds[m]
                outside = False
                if fl is not None and fp is not None and len(fronts_px) == len(hl):
                    a, bb = fronts_px[m]; c = hl[m].mean(0); outside = side(fp, a, bb) != side(c, a, bb)
                if j in LEGS or not outside: b = min(b, d); cnt["원칙"] += d <= 0
                else:
                    cnt["밖"] += d <= 0
                    if j in ARMS: b = min(b, d - DEEP)  # 손·팔꿈치는 0.7 더 깊어야 경계 0으로 취급
        if b < np.inf: db[i] = b
    return db, cnt

def fronts_px_of(k, wh): return [(np.array(l[0]) * wh, np.array(l[1]) * wh) for l in FRONT(k)]

print("== 사고 4건")
for k in VIDEOS:
    labA, tcA = read(SRC["A"].format(k)); labB, tcB = read(SRC["B"].format(k)); tc = min([v for v in (tcA, tcB) if v is not None], default=None)
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True))
    T.Z = json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    T.L = json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json")))["lines"]; F = T.features(P)
    fl = Floor(T.L, P).fit_scale() if k in CALIB_OK else None
    db, cnt = db4(P, F, fl, fronts_px_of(k, P["wh"]) if fl else []); m = np.array([g(v) in labA and g(v) in labB for v in F["t"]]); tt = F["t"][m]; o = judge_db(F, db)[m]; line = []
    for gname, y in (("A", np.array([labA[g(v)] for v in tt])), ("B+1", np.minimum(np.array([labB[g(v)] for v in tt]) + 1, 3))):
        gs = y == 3; h0 = float(tt[gs][0]); r = score(tt, o, gs, y < 3, h0, tc, y)
        line.append(f"{gname} 잡음 {r['hit']} 첫정지차 {None if r.get('first_stop') is None else round(r['first_stop'] - h0, 2)} 재현율 {r['stop_recall']}")
    print(k, "규칙4", " | ".join(line), cnt, flush=True)
print("== 평소 작업")
OBJ = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values()); L11 = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))["lines"]; T.L = L11
total = 0.0; c = 0
for p in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    name = os.path.basename(p)[:-9]; P = dict(np.load(p, allow_pickle=True)); F = T.features(P, OBJ); t = F["t"]; total += float(t[-1] - t[0]) + 1 / F["fps"]
    db, cnt = db4(P, F, True, fronts_px_of("press", P["wh"])); e = episodes(t, judge_db(F, db) == 3, 1.0); c += len(e)
    if e: print(name, "남은 헛정지", [(round(a, 1), round(b_, 1)) for a, b_ in e], cnt)
print(f"평소 작업 헛정지: 출발점 9회 → 규칙4 {c}회 ({c/total*60:.2f}/분)")
