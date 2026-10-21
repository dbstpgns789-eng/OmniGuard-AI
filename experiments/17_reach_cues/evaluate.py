# 실험 17: 판단기의 정지에 "손이 롤 안으로 들어갔다"는 증거(가림, 팔 단축)를 요구한다. 증거가 없으면 정지 → 감속
# 설계·경계값은 README 그대로: 팔 단축 r < 0.8, 가림 기억 1.0초, 단서 유지 0.4초, 옷 신호의 겹침 정지(실험 16)는 그대로 둔다
import os, sys, json, glob, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
src = open(os.path.join(X, "16_clothing", "evaluate.py"), encoding="utf-8").read(); exec(src[:src.index("R = {}")])
R_SHORT, OCC_MEM, HOLD = 0.8, 1.0, 0.4

def cues(P, F, near_slow):
    n = len(P["t"]); fps = F["fps"]; h = float(P["wh"][1]); occ, fore = np.zeros(n, bool), np.zeros(n, bool); last_in = -1e9
    ratios = [np.linalg.norm(k[w] - k[e]) / torso_len(k, s) for K, S in zip(P["kps"], P["sc"]) for k, s in zip(K, S)
              for w, e in ((9, 7), (10, 8)) if min(s[w], s[e]) >= KP_MIN and torso_len(k, s)]
    ref = float(np.percentile(ratios, 90)) if ratios else None
    for i, (K, S) in enumerate(zip(P["kps"], P["sc"])):
        hulls = F["hull_at"](i); hulls = [hh.astype(np.float32) for hh in (hulls if isinstance(hulls, list) else [hulls])]
        sd = lambda q, tl: min(-cv2.pointPolygonTest(hh, (float(q[0]), float(q[1])), True) for hh in hulls) / tl
        seen_in = False
        for k, s in zip(K, S):
            tl0 = torso_len(k, s); tl = tl0 or 0.25 * h
            for w, e in ((9, 7), (10, 8)):
                if s[w] >= KP_MIN and sd(k[w], tl) <= 0: seen_in = True
                if s[w] < KP_MIN and s[e] >= KP_MIN and sd(k[e], tl) < near_slow and P["t"][i] - last_in <= OCC_MEM: occ[i] = True
                if ref and tl0 and min(s[w], s[e]) >= KP_MIN and np.linalg.norm(k[w] - k[e]) / tl0 / ref < R_SHORT and sd(k[w], tl) <= 0: fore[i] = True
        if seen_in: last_in = P["t"][i]
    return occ, fore, ref

def gate(out, cue, keep, fps):
    hk = max(1, int(round(HOLD * fps))); recent = np.array([cue[max(0, i - hk):i + 1].any() for i in range(len(cue))])
    o = out.copy(); o[(o == 3) & ~recent & ~keep] = 2; return o

def ov_stop(ov, fps):  # 실험 16의 겹침 정지가 켜진 프레임
    kk = max(1, int(round(0.2 * fps))); return np.array([sum(ov[j] for j in (i, i - kk, i - 2 * kk) if j >= 0) >= 2 for i in range(len(ov))])

def variants(P, F, S, J):
    outs, info = {}, {}
    for n, (o, ns) in J.items():
        up1, ov = cloth_signals(P, F, S, ns); oc = apply(o, up1, ov, F["fps"]); occ, fore, ref = cues(P, F, ns); cue = occ | fore
        outs[n] = o; outs[n + "+옷"] = oc; outs[n + "+단서"] = gate(o, cue, np.zeros(len(o), bool), F["fps"]); outs[n + "+옷+단서"] = gate(oc, cue, ov_stop(ov, F["fps"]), F["fps"])
        info[n] = {"occ_frames": int(occ.sum()), "fore_frames": int(fore.sum()), "forearm_ref": None if ref is None else round(ref, 3)}
    return outs, info

R = {}
for k in VIDEOS:
    labA, tcA = read(SRC["A"].format(k)); labB, tcB = read(SRC["B"].format(k)); tc = min([x for x in (tcA, tcB) if x is not None], default=None)
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True)); S = dict(np.load(SEG + f"{k}_seg.npz", allow_pickle=True))
    T.Z = json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    T.L = json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json")))["lines"]
    F, J = judges(None, P); t = F["t"]; outs, info = variants(P, F, S, J)
    m = np.array([g(x) in labA and g(x) in labB for x in t]); tt = t[m]
    yA = np.array([labA[g(x)] for x in tt]); yB1 = np.minimum(np.array([labB[g(x)] for x in tt]) + 1, 3)
    R[k] = {"cue_stats": info}
    for gname, (gs, nm, y4) in {"A": (yA == 3, yA < 3, yA), "B+1": (yB1 == 3, yB1 < 3, yB1)}.items():
        h0 = float(tt[gs][0]) if gs.any() else None
        R[k][gname] = {n: score(tt, o[m], gs, nm, h0, tc, y4) for n, o in outs.items()}
    print(k, info, flush=True)

OBJ = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values()); T.L = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))["lines"]
press, total, pinfo = {}, 0.0, {}
for p in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    name = os.path.basename(p)[:-9]; P = dict(np.load(p, allow_pickle=True)); S = dict(np.load(SEG + f"{name}_seg.npz", allow_pickle=True))
    F, J = judges(OBJ, P); t = F["t"]; total += float(t[-1] - t[0]) + 1 / F["fps"]; outs, info = variants(P, F, S, J); pinfo[name] = info
    for nn, oo in outs.items():
        d = press.setdefault(nn, {"stop": 0, "slow_or_more": 0}); d["stop"] += len(onsets(t, oo, 3)); d["slow_or_more"] += len(onsets(t, (oo >= 2).astype(int) * 2, 2))
for d in press.values(): d["stop_per_min"] = round(d["stop"] / total * 60, 2); d["slow_per_min"] = round(d["slow_or_more"] / total * 60, 2)
R["_press_normal"] = {"sec": round(total, 1), **press, "cue_stats": pinfo}

SUM = {}
for gname in ("A", "B+1"):
    for n in [b + s for b in ("J09", "J13") for s in ("", "+옷", "+단서", "+옷+단서")]:
        rs = [R[k][gname][n] for k in VIDEOS if R[k][gname][n].get("hit") is not None]
        SUM[f"{gname}|{n}"] = {"hits": f"{sum(r['hit'] for r in rs)}/{len(rs)}", "first_stop": [None if R[k][gname][n].get("first_stop") is None else round(R[k][gname][n]["first_stop"], 2) for k in VIDEOS],
                               "stop_recall": [R[k][gname][n]["stop_recall"] for k in VIDEOS], "false_stops_in_accident_videos": sum(R[k][gname][n]["false_stops"] for k in VIDEOS)}
json.dump({"design": "README.md", "summary": SUM, "per_video": R}, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in R["_press_normal"].items() if k != "cue_stats"}, ensure_ascii=False))
for k, v in SUM.items():
    if k.startswith("A|"): print(k, v)
