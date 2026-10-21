# V5: 미터 단위 깊이 모델로 손목-회전체 3D 거리를 재서 판단한다 (실험 08과 같은 영상·포즈·위험한계·정답)
# 실행 전에 정한 기준: 3D 거리 < 0.15m 정지, < 0.5m 감속, < 1.0m 주의. 카메라 수평 시야각 70° 가정
# 손목이 위험한계에서 화면상 2몸통 이상 떨어진 프레임은 깊이를 구하지 않고 "멀다"로 둔다(계산 절약)
# 프레임별 결과는 캐시에 저장해서 중단돼도 이어서 한다
import json, os, sys, time
import numpy as np, cv2, torch
from PIL import Image
from transformers import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "C:/Users/User/Desktop/OmniGuard-data/"; EXP = DATA + "exp08/"
CASES = json.load(open(os.path.join(HERE, "cases.json"), encoding="utf-8"))
LABELS = json.load(open(os.path.join(HERE, "labels.json"), encoding="utf-8"))
KP_MIN, NEED, FOV_DEG, IN_W = 0.3, 3, 70.0, 640
TH = {3: 0.15, 2: 0.5, 1: 1.0}
torch.set_num_threads(16)
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", device="cpu")

def label_of(name, t):
    s = int(round(t))
    for k in ("Y", "N", "U"):
        if any(a <= s <= b for a, b in LABELS[name][k]): return k
    return "U"

def polys_at(c, wh, H):
    w, h = wh
    return [cv2.perspectiveTransform((np.array(p, np.float64) * [w, h]).reshape(-1, 1, 2), H).reshape(-1, 2) for p in c["rot"]]

def confirm(raw): return [i >= NEED - 1 and all(raw[i - NEED + 1:i + 1]) for i in range(len(raw))]

summary, lat = {}, []
for name, c in CASES.items():
    if sys.argv[1:] and name not in sys.argv[1:]: continue
    P = dict(np.load(EXP + f"{name}_pose.npz", allow_pickle=True)); t = P["t"]; w, h = P["wh"]
    cache_p = EXP + f"{name}_v5.json"; cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    s = IN_W / w; f = (IN_W / 2) / np.tan(np.radians(FOV_DEG / 2)); cx, cy = IN_W / 2, h * s / 2
    cap = cv2.VideoCapture(DATA + "raw_cctv/" + c["file"]); fps = cap.get(5)
    for i in range(len(t)):
        if str(i) in cache: continue
        polys = [p * s for p in polys_at(c, P["wh"], P["H"][i])]
        best = None
        cand = []
        for k, sc in zip(P["kps"][i], P["sc"][i]):
            k = k * s
            tl = np.linalg.norm((k[5] + k[6]) / 2 - (k[11] + k[12]) / 2) if min(sc[5], sc[6], sc[11], sc[12]) >= KP_MIN else 0.15 * h * s
            for j in (9, 10):
                if sc[j] < KP_MIN: continue
                sd = max(cv2.pointPolygonTest(p.astype(np.float32), (float(k[j][0]), float(k[j][1])), True) for p in polys)
                if sd > -2 * tl: cand.append((k, sc, j))
        if not cand: cache[str(i)] = None; continue
        cap.set(1, int(round(t[i] * fps))); ok, img = cap.read()
        if not ok: cache[str(i)] = None; continue
        img = cv2.resize(img, (IN_W, int(h * s)))
        t0 = time.perf_counter(); Z = np.array(pipe(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))["predicted_depth"].squeeze().float()); lat.append((time.perf_counter() - t0) * 1000)
        Z = cv2.resize(Z, (img.shape[1], img.shape[0]))
        mask = np.zeros(Z.shape, np.uint8)
        for p in polys: cv2.fillPoly(mask, [p.astype(np.int32)], 1)
        for k, sc in zip(P["kps"][i], P["sc"][i]):  # 사람 몸이 가린 부분은 회전체 깊이에서 뺀다
            pts = k[sc >= KP_MIN] * s
            if len(pts): x0, y0 = pts.min(0) - 12; x1, y1 = pts.max(0) + 12; mask[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = 0
        vv, uu = np.nonzero(mask[::3, ::3]); vv, uu = vv * 3, uu * 3
        if len(vv) < 20: cache[str(i)] = None; continue
        zr = Z[vv, uu]; R = np.stack([(uu - cx) * zr / f, (vv - cy) * zr / f, zr], 1)
        for k, sc, j in cand:
            x, y = int(np.clip(k[j][0], 2, Z.shape[1] - 3)), int(np.clip(k[j][1], 2, Z.shape[0] - 3))
            zw = float(np.median(Z[y - 2:y + 3, x - 2:x + 3])); W3 = np.array([(x - cx) * zw / f, (y - cy) * zw / f, zw])
            d = float(np.min(np.linalg.norm(R - W3, axis=1)))
            best = d if best is None else min(best, d)
        cache[str(i)] = best
        if i % 50 == 0: json.dump(cache, open(cache_p, "w")); print(name, i, len(t), round(best, 2) if best is not None else None, flush=True)
    json.dump(cache, open(cache_p, "w"))
    dist = [cache.get(str(i)) for i in range(len(t))]
    level = [0 if d is None else (3 if d < TH[3] else 2 if d < TH[2] else 1 if d < TH[1] else 0) for d in dist]
    labs = [label_of(name, x) for x in t]; stop = confirm([L == 3 for L in level])
    first_y = next((x for x, l in zip(t, labs) if l == "Y"), None)
    on = [i for i in range(len(stop)) if stop[i] and (i == 0 or not stop[i - 1])]
    dY = [d for d, l in zip(dist, labs) if l == "Y" and d is not None]; dN = [d for d, l in zip(dist, labs) if l == "N" and d is not None]
    summary[name] = {"stop_ratio_on_Y": round(float(np.mean([s_ for s_, l in zip(stop, labs) if l == "Y"])), 3) if "Y" in labs else None,
                     "false_stop_ratio_on_N": round(float(np.mean([s_ for s_, l in zip(stop, labs) if l == "N"])), 3) if "N" in labs else None,
                     "stop_onsets_total": len(on), "onsets_in_N": sum(labs[i] == "N" for i in on),
                     "onsets_in_last3s_before_contact": sum(1 for i in on if first_y and first_y - 3 <= t[i] < first_y),
                     "dist3d_median_Y": round(float(np.median(dY)), 2) if dY else None, "dist3d_median_N": round(float(np.median(dN)), 2) if dN else None,
                     "dist3d_p10_N": round(float(np.percentile(dN, 10)), 2) if dN else None, "depth_frames": sum(d is not None for d in dist)}
    print(name, json.dumps(summary[name], ensure_ascii=False), flush=True)
summary["depth_cpu_ms_median"] = round(float(np.median(lat)), 1) if lat else None
json.dump(summary, open(os.path.join(HERE, "result_v5.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
