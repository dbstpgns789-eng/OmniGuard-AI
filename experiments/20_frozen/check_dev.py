# 고정한 판단기가 개발용 데이터에서 예전 결과를 그대로 내는지 확인 (실험 13·18·19와 같은 채점)
# 기대: J13 사고 4/4, RULE4 사고 4/4·평소 헛정지 1회, QUIET 사고 3/4·평소 헛정지 0회 (택배 컨베이어 제외 4건, 정답 세훈 A)
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
src = open(os.path.join(X, "19_remove_false", "step0.py"), encoding="utf-8").read(); exec(src[:src.index('print("== 사고 4건")')])
sys.path.insert(0, HERE); import judges as J
FRONT = lambda k: json.load(open(os.path.join(X, "14_foot_gate", f"front_{k}.json")))["lines"]
CALIB_OK = {"paper_rewinder"}  # 실험 19와 같게: 바닥 보정이 된 영상만 앞선 사용

res = {n: {"A": 0, "B+1": 0, "false": 0} for n in J.PARAMS}
for k in VIDEOS:
    labA, tcA = read(SRC["A"].format(k)); labB, tcB = read(SRC["B"].format(k)); tc = min([v for v in (tcA, tcB) if v is not None], default=None)
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True))
    F = J.features(P, json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))), json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json"))))
    m = np.array([g(v) in labA and g(v) in labB for v in F["t"]]); tt = F["t"][m]
    for n in J.PARAMS:
        o = J.run(n, P, F, FRONT(k) if k in CALIB_OK else None)[m]
        for gname, y in (("A", np.array([labA[g(v)] for v in tt])), ("B+1", np.minimum(np.array([labB[g(v)] for v in tt]) + 1, 3))):
            gs = y == 3; r = score(tt, o, gs, y < 3, float(tt[gs][0]), tc, y); res[n][gname] += bool(r["hit"])
objs = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values())
floor = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json"))); total = 0.0
for p in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    P = dict(np.load(p, allow_pickle=True)); T.L = floor["lines"]; F = T.features(P, objs); t = F["t"]; total += float(t[-1] - t[0]) + 1 / F["fps"]
    for n in J.PARAMS: res[n]["false"] += len(episodes(t, J.run(n, P, F, FRONT("press")) == 3, 1.0))
for n, r in res.items(): print(f"{n}: 사고 세훈 {r['A']}/4, 채우 {r['B+1']}/4 | 평소 작업 {total:.0f}초 헛정지 {r['false']}회 ({r['false']/total*60:.2f}/분)")
json.dump({"total_sec": round(total, 1), **res}, open(os.path.join(HERE, "check_dev.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
