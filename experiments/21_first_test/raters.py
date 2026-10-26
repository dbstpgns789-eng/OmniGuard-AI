# 시험 영상 20개에서 판단자 2명(세훈, 채우)의 일치도. 판단 기준(docs/판단기준.md, 3단계)으로 각자 작성한 판단표
# 읽는 규칙은 evaluate.py와 같다(편집 구간 제외, 앞 빈칸은 정상). 지표는 실험 15와 같다: 서열 α, 완전 일치, 첫 정지 시각 차이, 닿음 시각 차이
import os, sys, csv, json, shutil, glob, numpy as np, krippendorff
HERE = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(HERE, "evaluate.py"), encoding="utf-8").read(); exec(src[:src.index("def geometry")])
CW = DATA + "판단표_채우_사고19/"
if not os.path.exists(CW):  # 받은 압축을 레포 밖 데이터 폴더로 옮겨 둔다
    for p in glob.glob("C:/Users/User/Downloads/채우_결과/채우_전달/판단표_사고19/*/판단표.csv"):
        d = CW + os.path.basename(os.path.dirname(p)) + "/"; os.makedirs(d, exist_ok=True); shutil.copy(p, d + "판단표.csv")

def read_b(name):
    """채우 판단표: 세훈 판단표와 같은 해석 규칙(제외 구간은 같은 영상 편집 구간이라 똑같이 적용, 칸 보정은 세훈 것만)"""
    rows = list(csv.reader(open(CW + f"{name}/판단표.csv", encoding="utf-8-sig")))[1:]
    ex = RULES["제외"].get(name); cur = 0; lab, tc, hs = {}, None, None
    last = max((i for i, r in enumerate(rows) if any(v.strip() for v in r[2:])), default=-1)
    for r in rows[:last + 1]:
        t = round(float(r[0]), 1); s = r[2].strip() if len(r) > 2 else ""; y = r[3].strip().upper() if len(r) > 3 else ""
        if s in NAMES3: cur = NAMES3.index(s)
        if ex and ex[0] <= t < ex[1]: continue
        lab[t] = cur
        if y == "Y" and tc is None: tc = t
        if cur == 2 and hs is None: hs = t
    return lab, tc, hs

res = {}; A_all, B_all = [], []
for name in sorted(n for n in os.listdir(LAB) if os.path.isdir(LAB + n)):
    la, tca, hsa = labels(name); lb, tcb, hsb = read_b(name); ks = sorted(set(la) & set(lb))
    a = np.array([la[k] for k in ks]); b = np.array([lb[k] for k in ks]); A_all += list(a); B_all += list(b)
    alpha = krippendorff.alpha(reliability_data=[a, b], level_of_measurement="ordinal") if len(set(a) | set(b)) > 1 else None
    res[name] = {"frames": len(ks), "alpha": None if alpha is None else round(float(alpha), 2), "exact": round(float((a == b).mean()), 2),
                 "first_stop_A": hsa, "first_stop_B": hsb, "stop_diff": None if hsa is None or hsb is None else round(hsb - hsa, 1),
                 "contact_A": tca, "contact_B": tcb, "contact_diff": None if tca is None or tcb is None else round(tcb - tca, 1)}
    print(name, res[name], flush=True)
A_all, B_all = np.array(A_all), np.array(B_all)
sd = [r["stop_diff"] for r in res.values() if r["stop_diff"] is not None]; cd = [r["contact_diff"] for r in res.values() if r["contact_diff"] is not None]
summ = {"alpha_pooled": round(float(krippendorff.alpha(reliability_data=[A_all, B_all], level_of_measurement="ordinal")), 2),
        "exact_pooled": round(float((A_all == B_all).mean()), 2), "within1_pooled": round(float((abs(A_all - B_all) <= 1).mean()), 2),
        "stop_diff_median_abs": round(float(np.median(np.abs(sd))), 2), "stop_within_1s": f"{sum(abs(x) <= 1.0 for x in sd)}/{len(sd)}",
        "contact_diff_median_abs": round(float(np.median(np.abs(cd))), 2), "contact_within_0.5s": f"{sum(abs(x) <= 0.5 for x in cd)}/{len(cd)}"}
print(json.dumps(summ, ensure_ascii=False, indent=1))
json.dump({"summary": summ, "clips": res}, open(os.path.join(HERE, "raters.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
