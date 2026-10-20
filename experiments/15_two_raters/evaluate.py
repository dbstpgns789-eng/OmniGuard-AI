# 실험 15: 판단자 2명(A 세훈, B 채우)의 일치도와, 동결한 판단기(J09, J13, V1)의 채점. 기준은 PREREG.md(커밋 a12a1c0) 그대로
# V1은 14_foot_gate/run.py의 V1과 같은 규칙(손목이 위험한계 안인 사람이 있으면 정지, 3프레임 연속)을 바닥 보정 없이 다시 쓴 것
import os, sys, csv, json, numpy as np, cv2, krippendorff
from scipy.stats import chi2
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import cohen_kappa_score
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
for d in ("09_red_fit", "10_isafety_transfer"): sys.path.insert(0, os.path.join(X, d))
import transfer as T
from common import STAGES, DATA, KP_MIN

VIDEOS = ["slitter", "paper_rewinder", "night_paper", "conveyor_parcel", "bag_machine"]
SRC = {"A": DATA + "exp08/frames_for_review/{}/판단표.csv", "B": DATA + "판단표_채우_결과/{}/판단표.csv"}
EARLY, MERGE, ACTIVE_PRE = 2.0, 1.0, 5.0
P_RULE = list(T.p_red)
for k, v in (("near_warn", 1.0), ("near_slow", 0.5), ("hand_inside", 0.0), ("body_inside", 0.0)): P_RULE[T.NAMES.index(k)] = v
g = lambda x: round(round(x / 0.2) * 0.2, 1)

def read(path):  # 0.2초 격자 → 단계(빈칸은 앞 판단, 마지막 판단 뒤는 제외), 첫 "닿음 Y" 시각
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r["시각(초)"].strip()]
    seq = [(round(float(r["시각(초)"]), 1), r["단계(정상/주의/감속/정지)"].strip(), r["회전체에 닿음(Y/N/모름)"].strip().upper()) for r in rows]
    last = max(i for i, (_, v, _) in enumerate(seq) if v in STAGES); lab, cur = {}, None
    for i, (s, v, _) in enumerate(seq):
        if v in STAGES: cur = STAGES.index(v)
        if i <= last and cur is not None: lab[s] = cur
    yc = [s for s, _, c in seq if c == "Y"]
    return lab, (min(yc) if yc else None)

def onset(t, a, lvl=3):
    i = np.flatnonzero(a >= lvl); return None if len(i) == 0 else float(t[i[0]])

def episodes(t, on, gap):  # 정지 켜짐 구간(시작, 끝), 간격이 gap보다 짧으면 합침
    eps = []
    for i in np.flatnonzero(on):
        if eps and t[i] - eps[-1][1] <= gap + 1e-6: eps[-1][1] = t[i]
        else: eps.append([t[i], t[i]])
    return eps

def garwood(k, minutes):
    hi = chi2.ppf(0.975, 2 * k + 2) / 2; lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
    return round(lo / minutes, 2), round(hi / minutes, 2)

def score(t, out, gt_stop, normal, h0, tc, y4=None):
    """t·out·gt는 채점 프레임만. gt_stop: 정답 정지 / normal: 헛정지를 세는 프레임 / h0: 정답 첫 정지 / tc: 접촉"""
    end = tc if tc is not None else float(t[-1]); r = {"human_first_stop": h0}
    eps = episodes(t, out == 3, MERGE)
    if h0 is not None:
        w0 = h0 - EARLY; valid = [max(s, w0) for s, e in eps if s <= end and e >= w0]
        first = min(valid) if valid else None
        r.update(hit=first is not None, first_stop=first, lead_s=None if first is None or tc is None else round(tc - first, 2),
                 margin_vs_human_s=None if first is None else round(h0 - first, 2), early_false=sum(1 for s, e in eps if e < w0))
    r["stop_recall"] = round(float((out[gt_stop] == 3).mean()), 3) if gt_stop.any() else None
    mins = normal.sum() * float(np.median(np.diff(t))) / 60 if normal.sum() > 1 else 0
    k = len(episodes(t, (out == 3) & normal, MERGE)); r.update(false_stops=k, normal_min=round(mins, 2), false_per_min=None if mins == 0 else round(k / mins, 2), false_ci=None if mins == 0 else garwood(k, mins))
    if y4 is not None:
        d = out - y4; r.update(under=round(float((d < 0).mean()), 3), over=round(float((d > 0).mean()), 3),
                               cautious_ok=round(float(((d == 0) | (d == 1)).mean()), 3),
                               critical_miss=round(float((out[y4 == 3] <= 1).mean()), 3) if (y4 == 3).any() else None)
    return r

R, AGREE, pooled = {}, {}, ([], [])
for k in VIDEOS:
    labA, tcA = read(SRC["A"].format(k)); labB, tcB = read(SRC["B"].format(k)); tc = min([x for x in (tcA, tcB) if x is not None], default=None)
    # 1부: 일치도 (0.2초 격자, 두 사람 모두 판단이 있는 칸)
    grid = sorted(set(labA) & set(labB)); a = np.array([labA[s] for s in grid]); b = np.array([labB[s] for s in grid]); gt_ = np.array(grid)
    first_abn = min([s for s in grid if labA[s] > 0 or labB[s] > 0], default=grid[-1]); act = gt_ >= first_abn - ACTIVE_PRE
    aa, bb = a[act], b[act]; pooled[0].extend(aa); pooled[1].extend(bb)
    alpha = None if len(set(aa) | set(bb)) < 2 else round(float(krippendorff.alpha(reliability_data=np.vstack([aa, bb]).astype(float), level_of_measurement="ordinal", value_domain=[0, 1, 2, 3])), 3)
    oA, oB = onset(gt_, a), onset(gt_, b); wA, wB = onset(gt_, a, 1), onset(gt_, b, 1)
    AGREE[k] = {"grid_frames": len(grid), "active_frames": int(act.sum()), "alpha_ordinal": alpha,
                "kappa_linear": round(float(cohen_kappa_score(aa, bb, labels=[0, 1, 2, 3], weights="linear")), 3) if len(set(aa) | set(bb)) > 1 else None,
                "exact": round(float((aa == bb).mean()), 3), "within1": round(float((abs(aa - bb) <= 1).mean()), 3),
                "A_first_stop": oA, "B_first_stop": oB, "stop_onset_diff_B_minus_A": None if oA is None or oB is None else round(oB - oA, 2),
                "A_first_warn": wA, "B_first_warn": wB, "warn_onset_diff_B_minus_A": None if wA is None or wB is None else round(wB - wA, 2),
                "contact_A": tcA, "contact_B": tcB, "contact_used": tc,
                "confusion_A_rows_B_cols": [[int(((aa == i) & (bb == j)).sum()) for j in range(4)] for i in range(4)]}
    # 2부: 판단기 채점 (프레임 단위, 두 사람 모두 판단이 있는 프레임만)
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True))
    T.Z = json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    T.L = json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json")))["lines"]; F = T.features(P); t = F["t"]
    v1 = np.zeros(len(t), int)
    for i, (K, S) in enumerate(zip(P["kps"], P["sc"])):
        hs = F["hull_at"](i).astype(np.float32)
        if any(s[j] >= KP_MIN and cv2.pointPolygonTest(hs, (float(q[j][0]), float(q[j][1])), False) >= 0 for q, s in zip(K, S) for j in (9, 10)): v1[i] = 3
    v1 = np.array([3 if i >= 2 and (v1[i - 2:i + 1] == 3).all() else 0 for i in range(len(t))])
    outs = {"J09": T.judge(F, T.p_red)[0], "J13": T.judge(F, P_RULE)[0], "V1": v1}
    m = np.array([g(x) in labA and g(x) in labB for x in t]); tt = t[m]
    yA = np.array([labA[g(x)] for x in tt]); yB = np.array([labB[g(x)] for x in tt])
    outs = {n: o[m] for n, o in outs.items()}; outs["사람B(천장)"] = yB; outs["사람A(천장)"] = yA
    GT = {"A": (yA == 3, yA < 3, yA), "B": (yB == 3, yB < 3, yB), "AnB": ((yA == 3) & (yB == 3), (yA < 3) & (yB < 3), None),
          "AuB": ((yA == 3) | (yB == 3), (yA < 3) & (yB < 3), None),
          "contact": ((tt >= tc) if tc is not None else np.zeros(len(tt), bool), (tt < tc) if tc is not None else np.ones(len(tt), bool), None)}
    R[k] = {}
    for gname, (gs, nm, y4) in GT.items():
        h0 = float(tt[gs][0]) if gs.any() else None
        R[k][gname] = {n: score(tt, o, gs, nm, h0, tc, y4) for n, o in outs.items()
                       if not (n == "사람A(천장)" and gname in ("A", "AnB", "AuB")) and not (n == "사람B(천장)" and gname in ("B", "AnB", "AuB"))}
    print(k, "done", flush=True)

pa, pb = np.array(pooled[0]), np.array(pooled[1])
AGREE["_pooled"] = {"active_frames": len(pa), "alpha_ordinal": round(float(krippendorff.alpha(reliability_data=np.vstack([pa, pb]).astype(float), level_of_measurement="ordinal", value_domain=[0, 1, 2, 3])), 3),
                    "kappa_linear": round(float(cohen_kappa_score(pa, pb, labels=[0, 1, 2, 3], weights="linear")), 3),
                    "exact": round(float((pa == pb).mean()), 3), "within1": round(float((abs(pa - pb) <= 1).mean()), 3),
                    "median_abs_stop_onset_diff": float(np.median([abs(v["stop_onset_diff_B_minus_A"]) for kk, v in AGREE.items() if kk in VIDEOS and v["stop_onset_diff_B_minus_A"] is not None]))}
SUM = {}
for gname in ("A", "B", "AnB", "AuB", "contact"):
    for n in ("J09", "J13", "V1", "사람B(천장)", "사람A(천장)"):
        rs = [R[k][gname][n] for k in VIDEOS if n in R[k][gname] and R[k][gname][n].get("hit") is not None]
        if not rs: continue
        h = sum(r["hit"] for r in rs); ci = proportion_confint(h, len(rs), alpha=0.05, method="beta")
        fk = sum(R[k][gname][n]["false_stops"] for k in VIDEOS); fm = sum(R[k][gname][n]["normal_min"] for k in VIDEOS)
        SUM[f"{gname}|{n}"] = {"hits": f"{h}/{len(rs)}", "hit_ci": [round(ci[0], 2), round(ci[1], 2)],
                               "leads": [r["lead_s"] for r in rs], "margins": [r["margin_vs_human_s"] for r in rs],
                               "false_stops": fk, "normal_min": round(fm, 2), "false_per_min": round(fk / fm, 2) if fm else None, "false_ci": garwood(fk, fm) if fm else None,
                               "critical_miss": [R[k][gname][n].get("critical_miss") for k in VIDEOS]}
json.dump({"prereg_commit": "a12a1c0", "agreement": AGREE, "summary": SUM, "per_video": R}, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(AGREE, ensure_ascii=False, default=float)); print(json.dumps(SUM, ensure_ascii=False, default=float))
