# 실험 16: 동결 판단기(J09, J13)에 사람 윤곽(옷) 신호를 얹는다. 규칙·경계값은 README 설계 그대로
#   옷 앞섬(관절 거리 − 윤곽 거리) ≥ 0.15 이고 윤곽이 위험한계에 닿음 → 한 칸 올림
#   윤곽 면적 10% 이상이 위험한계 안 (0.2초 간격 3스텝 중 2) + 손목·팔꿈치가 감속 거리 안이거나 가려짐 → 정지
# 채점은 실험 15(evaluate.py)의 함수를 그대로 쓴다. 정답 A, B, A∩B, 그리고 B를 한 칸 올린 것
import os, sys, json, glob, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
src = open(os.path.join(X, "15_two_raters", "evaluate.py"), encoding="utf-8").read(); exec(src[:src.index("R, AGREE, pooled")])
from common import torso_len, onsets
SEG = DATA + "exp16/"; LEAD, OVER = 0.15, 0.10

def cloth_signals(P, F, S, near_slow):
    """프레임마다: up1(한 칸 올림 조건), ov(정지 경로의 윤곽 겹침 조건)"""
    n = len(P["t"]); up1, ov = np.zeros(n, bool), np.zeros(n, bool); w, h = [int(v) for v in P["wh"]]; sc_ = 0.25
    for i in range(min(n, len(S["masks"]))):
        hulls = F["hull_at"](i); hulls = hulls if isinstance(hulls, list) else [hulls]; hulls = [hh.astype(np.float32) for hh in hulls]
        sd = lambda q, tl: min(-cv2.pointPolygonTest(hh, (float(q[0]), float(q[1])), True) for hh in hulls) / tl
        zone = np.zeros((int(h * sc_) + 1, int(w * sc_) + 1), np.uint8)
        for hh in hulls: cv2.fillPoly(zone, [(hh * sc_).astype(np.int32)], 1)
        masks = [np.asarray(m, np.float32).reshape(-1, 2) for m in S["masks"][i]]; masks = [m for m in masks if len(m) >= 3]
        for k, s in zip(P["kps"][i], P["sc"][i]):
            vis = [j for j in range(17) if s[j] >= KP_MIN]
            if len(vis) < 3 or not masks: continue
            hit = [sum(cv2.pointPolygonTest(m, (float(k[j][0]), float(k[j][1])), False) >= 0 for j in vis) for m in masks]
            if max(hit) < 3: continue
            m = masks[int(np.argmax(hit))]; tl = torso_len(k, s) or 0.25 * h
            arm = [j for j in (5, 6, 7, 8, 9, 10) if s[j] >= KP_MIN]
            d_kp = min(sd(k[j], tl) for j in arm) if arm else np.inf
            d_mask = min(sd(q, tl) for q in m[:: max(1, len(m) // 60)])
            if d_mask <= 0 and d_kp - d_mask >= LEAD: up1[i] = True
            mm = np.zeros_like(zone); cv2.fillPoly(mm, [(m * sc_).astype(np.int32)], 1); area = mm.sum()
            if area and (mm & zone).sum() / area >= OVER:
                we = [j for j in (7, 8, 9, 10) if s[j] >= KP_MIN]
                if not we or min(sd(k[j], tl) for j in we) < near_slow: ov[i] = True
    return up1, ov

def apply(base, up1, ov, fps):
    out = np.where(up1, np.minimum(base + 1, 3), base); kk = max(1, int(round(0.2 * fps)))
    for i in range(len(out)):
        if sum(ov[j] for j in (i, i - kk, i - 2 * kk) if j >= 0) >= 2: out[i] = 3
    return out

def judges(k_zone, P):
    F = T.features(P) if k_zone is None else T.features(P, k_zone)
    return F, {"J09": (T.judge(F, T.p_red)[0], T.p_red[T.NAMES.index("near_slow")]), "J13": (T.judge(F, P_RULE)[0], P_RULE[T.NAMES.index("near_slow")])}

R = {}
for k in VIDEOS:
    labA, tcA = read(SRC["A"].format(k)); labB, tcB = read(SRC["B"].format(k)); tc = min([x for x in (tcA, tcB) if x is not None], default=None)
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True)); S = dict(np.load(SEG + f"{k}_seg.npz", allow_pickle=True))
    T.Z = json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    T.L = json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json")))["lines"]
    F, J = judges(None, P); t = F["t"]; fps = F["fps"]; outs, stat = {}, {}
    for n, (o, ns) in J.items():
        up1, ov = cloth_signals(P, F, S, ns); outs[n] = o; outs[n + "+옷"] = apply(o, up1, ov, fps); stat[n] = {"up1_frames": int(up1.sum()), "overlap_stop_frames": int(ov.sum())}
    m = np.array([g(x) in labA and g(x) in labB for x in t]); tt = t[m]
    yA = np.array([labA[g(x)] for x in tt]); yB = np.array([labB[g(x)] for x in tt]); yB1 = np.minimum(yB + 1, 3)
    GT = {"A": (yA == 3, yA < 3, yA), "B": (yB == 3, yB < 3, yB), "AnB": ((yA == 3) & (yB == 3), (yA < 3) & (yB < 3), None), "B+1": (yB1 == 3, yB1 < 3, yB1)}
    R[k] = {"cloth_stats": stat, "first_slow_or_more": {n: onset(tt, o[m], 2) for n, o in outs.items()}}
    for gname, (gs, nm, y4) in GT.items():
        h0 = float(tt[gs][0]) if gs.any() else None
        R[k][gname] = {n: score(tt, o[m], gs, nm, h0, tc, y4) for n, o in outs.items()}
    print(k, stat, flush=True)

# 프레스 평소 작업: 전부 정상. 헛정지·헛감속 켜짐 횟수
OBJ = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values()); T.L = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))["lines"]
press, total = {}, 0.0
for p in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    name = os.path.basename(p)[:-9]; P = dict(np.load(p, allow_pickle=True)); S = dict(np.load(SEG + f"{name}_seg.npz", allow_pickle=True))
    F, J = judges(OBJ, P); t = F["t"]; total += float(t[-1] - t[0]) + 1 / F["fps"]
    for n, (o, ns) in J.items():
        up1, ov = cloth_signals(P, F, S, ns)
        for nn, oo in ((n, o), (n + "+옷", apply(o, up1, ov, F["fps"]))):
            d = press.setdefault(nn, {"stop": 0, "slow_or_more": 0}); d["stop"] += len(onsets(t, oo, 3)); d["slow_or_more"] += len(onsets(t, (oo >= 2).astype(int) * 2, 2))
for d in press.values(): d["stop_per_min"] = round(d["stop"] / total * 60, 2); d["slow_per_min"] = round(d["slow_or_more"] / total * 60, 2)
R["_press_normal"] = {"sec": round(total, 1), **press}

SUM = {}
for gname in ("A", "B", "AnB", "B+1"):
    for n in ("J09", "J09+옷", "J13", "J13+옷"):
        rs = [R[k][gname][n] for k in VIDEOS if R[k][gname][n].get("hit") is not None]
        SUM[f"{gname}|{n}"] = {"hits": f"{sum(r['hit'] for r in rs)}/{len(rs)}", "first_stop": [R[k][gname][n].get("first_stop") for k in VIDEOS],
                               "margin_vs_human": [R[k][gname][n].get("margin_vs_human_s") for k in VIDEOS],
                               "stop_recall": [R[k][gname][n]["stop_recall"] for k in VIDEOS], "false_stops": sum(R[k][gname][n]["false_stops"] for k in VIDEOS)}
json.dump({"design": "README.md", "summary": SUM, "per_video": R}, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(R["_press_normal"], ensure_ascii=False)); print(json.dumps({k: SUM[k] for k in SUM if k.startswith("A|") or k.startswith("B+1|")}, ensure_ascii=False, default=float))
print("bag first slow+", R["bag_machine"]["first_slow_or_more"])
