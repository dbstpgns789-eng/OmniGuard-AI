# 실험 23 채점: "사람이 롤보다 앞에 서 있음"(깊이 순서)이면 정지를 감속으로 내린다. D0 = J13 + 깊이, D1 = C1 + 깊이
# 사용: python gate.py tune   /   python gate.py test
import os, sys, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); MODE = sys.argv[1]; sys.argv = ["combine.py", MODE]
src = open(os.path.join(HERE, "..", "22_combine", "combine.py"), encoding="utf-8").read(); exec(src[:src.index("names = [")])
DEP = DATA + "exp23/"; Q = 4  # 깊이 지도 축소 비율

def front_flags(P, F, S, Dz):
    """0.2초 간격마다 '앞에 서 있음' 판정 → 프레임마다 값"""
    t = F["t"]; raw = []
    for k, d in zip(Dz["t"], Dz["depth"]):
        d = d.astype(np.float32); i = int(np.argmin(np.abs(t - k))); hh, ww = d.shape
        hl = F["hull_at"](i); hl = hl if isinstance(hl, list) else [hl]; zone = np.zeros((hh, ww), np.uint8)
        for h_ in hl: cv2.fillPoly(zone, [(h_ / Q).astype(np.int32)], 1)
        masks = []
        for m in (S["masks"][i] if i < len(S["masks"]) else []):
            m = np.asarray(m, np.float32).reshape(-1, 2)
            if len(m) < 3: continue
            mm = np.zeros((hh, ww), np.uint8); cv2.fillPoly(mm, [(m / Q).astype(np.int32)], 1); masks.append(mm)
        over = [int((mm & zone).sum()) for mm in masks]
        if not masks or max(over) == 0: raw.append(False); continue
        person = masks[int(np.argmax(over))]; anyp = np.clip(sum(masks), 0, 1); roll = (zone == 1) & (anyp == 0)
        raw.append(bool(roll.sum() >= 20 and np.median(d[person == 1]) > np.percentile(d[roll], 90)))
    raw = np.array(raw); sm = np.array([raw[max(0, j - 4):j + 1].sum() >= 3 for j in range(len(raw))])
    idx = np.clip(np.searchsorted(Dz["t"], t + 1e-6) - 1, 0, len(raw) - 1); return sm[idx], raw

names = [n for n in sorted(os.listdir(LAB)) if os.path.isdir(LAB + n) and ((n not in TEST) if MODE == "tune" else (n in TEST))]
keep = ["J13", "C1", "D0", "D1"]; res = {}; nst = {c: [0, 0] for c in keep}
for name in names:
    P = dict(np.load(POSE + f"{name}_pose.npz", allow_pickle=True)); S = dict(np.load(SEG + f"{name}_seg.npz", allow_pickle=True)); Dz = dict(np.load(DEP + f"{name}_depth.npz"))
    lab, tc, hs = labels(name); zone, objects, floor, fronts = geometry(name); F = J.features(P, zone, floor, objects); t = F["t"]
    y = np.array([lab.get(round(round(v / 0.2) * 0.2, 1), -1) for v in t]); win = (t >= tc - 0.5) & (t <= tc)
    outs = candidates(P, F, S); front, raw = front_flags(P, F, S, Dz)
    for b, g in (("J13", "D0"), ("C1", "D1")): o = outs[b].copy(); o[(o == 3) & front] = 2; outs[g] = o
    r = {"normal_sec": float((y == 0).sum() / F["fps"]), "front_ratio": round(float(raw.mean()), 2), "depth_ms": round(float(np.median(Dz["ms"])), 0)}
    for c in keep:
        o = outs[c]; r[c] = {"hit": bool(win.any() and (o[win] == 3).all()), "false_stops": len(episodes(t, (o == 3) & (y == 0))),
                             "normal_stop_frac": round(float(((o == 3) & (y == 0)).sum() / max(1, (y == 0).sum())), 2)}
        nst[c][0] += int(((o == 3) & (y == 0)).sum()); nst[c][1] += int((y == 0).sum())
    res[name] = r; print(name, "앞 판정", r["front_ratio"], {c: (r[c]["hit"], r[c]["false_stops"], r[c]["normal_stop_frac"]) for c in keep}, flush=True)
mins = sum(r["normal_sec"] for r in res.values()) / 60
summ = {c: {"stop_rate": f"{sum(r[c]['hit'] for r in res.values())}/{len(res)}", "false_per_min": round(sum(r[c]["false_stops"] for r in res.values()) / mins, 2),
            "normal_stop_time": f"{nst[c][0] / max(1, nst[c][1]):.0%}"} for c in keep}
print(json.dumps(summ, ensure_ascii=False, indent=1))
json.dump({"mode": MODE, "summary": summ, "clips": res}, open(os.path.join(HERE, f"result_{MODE}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
