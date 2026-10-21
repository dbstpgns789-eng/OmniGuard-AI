# 실험 22: J13 위에 정지 유지(C1)·윤곽 보조(C2)·둘 다(C3)를 얹어 채점. 규칙과 고르는 기준은 README
# 사용: python combine.py tune          → 맞춤용 10개 화면에서 후보 3개 채점, 고르는 규칙으로 하나 선택
#       python combine.py test C1       → 고른 후보 하나만 시험용 10건에 (2차 시험, 한 번만)
import os, sys, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); MODE = sys.argv[1]; CAND = sys.argv[2] if len(sys.argv) > 2 else None; sys.argv = sys.argv[:1]  # evaluate.py는 화면 보정 켬으로 불러옴
src = open(os.path.join(HERE, "..", "21_first_test", "evaluate.py"), encoding="utf-8").read(); exec(src[:src.index("res = {}")])
sys.path.insert(0, os.path.join(X, "09_red_fit")); from common import KP_MIN
SEG = DATA + "exp22/"; NEAR, AWAY_S, OVER = 0.5, 1.0, 0.10  # J13 감속 거리, 근처를 벗어난 상태 유지 시간, 실험 16 겹침 비율

def signals(P, F, S):
    n = len(F["t"]); w, h = [int(v) for v in P["wh"]]; sc_ = 0.25; lost, inz = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(n):
        lost[i] = not any(int((np.asarray(s) >= KP_MIN).sum()) >= 3 for s in P["sc"][i])
        hl = F["hull_at"](i); hl = hl if isinstance(hl, list) else [hl]; zone = np.zeros((int(h * sc_) + 1, int(w * sc_) + 1), np.uint8)
        for hh in hl: cv2.fillPoly(zone, [(hh * sc_).astype(np.int32)], 1)
        for m in (S["masks"][i] if i < len(S["masks"]) else []):
            m = np.asarray(m, np.float32).reshape(-1, 2)
            if len(m) < 3: continue
            mm = np.zeros_like(zone); cv2.fillPoly(mm, [(m * sc_).astype(np.int32)], 1); a = mm.sum()
            if a and (mm & zone).sum() / a >= OVER: inz[i] = True; break
    near = lost | (np.nan_to_num(F["dh"], nan=9) < NEAR) | (np.nan_to_num(F["db"], nan=9) < NEAR)
    return lost, inz, near

def c2(base, lost, inz, fps):
    ov = lost & inz; k = max(1, int(round(0.2 * fps))); out = base.copy()
    for i in range(len(out)):
        if sum(ov[j] for j in (i, i - k, i - 2 * k) if j >= 0) >= 2: out[i] = 3
    return out

def c1(base, near, fps):
    out = base.copy(); on = False; away = 0
    for i in range(len(out)):
        if base[i] == 3: on, away = True, 0
        elif on:
            away = 0 if near[i] else away + 1
            if away >= int(AWAY_S * fps): on = False
        if on: out[i] = 3
    return out

def candidates(P, F, S):
    j13 = J.run("J13", P, F); lost, inz, near = signals(P, F, S); fps = F["fps"]
    return {"J13": j13, "C1": c1(j13, near, fps), "C2": c2(j13, lost, inz, fps), "C3": c1(c2(j13, lost, inz, fps), near, fps)}

names = [n for n in sorted(os.listdir(LAB)) if os.path.isdir(LAB + n) and ((n not in TEST) if MODE == "tune" else (n in TEST))]
keep = ["J13", "C1", "C2", "C3"] if MODE == "tune" else ["J13", CAND]
res = {}
for name in names:
    P = dict(np.load(POSE + f"{name}_pose.npz", allow_pickle=True)); S = dict(np.load(SEG + f"{name}_seg.npz", allow_pickle=True))
    lab, tc, hs = labels(name); zone, objects, floor, fronts = geometry(name); F = J.features(P, zone, floor, objects); t = F["t"]
    key = np.array([round(round(v / 0.2) * 0.2, 1) for v in t]); y = np.array([lab.get(k, -1) for k in key]); m = y >= 0
    win = (t >= tc - 0.5) & (t <= tc); outs = candidates(P, F, S); r = {"normal_sec": float((y == 0).sum() / F["fps"])}
    for c in keep:
        o = outs[c]; js = next((float(t[i]) for i in range(len(t)) if o[i] == 3 and m[i]), None)
        r[c] = {"hit": bool(win.any() and (o[win] == 3).all()), "false_stops": len(episodes(t, (o == 3) & (y == 0))),
                "first_stop_diff": None if js is None or hs is None else round(js - hs, 2)}
    res[name] = r; print(name, {c: (r[c]["hit"], r[c]["false_stops"]) for c in keep}, flush=True)
mins = sum(r["normal_sec"] for r in res.values()) / 60
summ = {c: {"stop_rate": f"{sum(r[c]['hit'] for r in res.values())}/{len(res)}", "hits": sum(r[c]["hit"] for r in res.values()),
            "false_stops": sum(r[c]["false_stops"] for r in res.values()), "false_per_min": round(sum(r[c]["false_stops"] for r in res.values()) / mins, 2)} for c in keep}
print(json.dumps(summ, ensure_ascii=False, indent=1))
if MODE == "tune":
    order = ["C1", "C2", "C3"]; pick = sorted(order, key=lambda c: (-summ[c]["hits"], summ[c]["false_per_min"], order.index(c)))[0]
    print("선택:", pick); summ["pick"] = pick
json.dump({"mode": MODE, "normal_min": round(mins, 2), "summary": summ, "clips": res}, open(os.path.join(HERE, f"result_{MODE}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
