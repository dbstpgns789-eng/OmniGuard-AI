# 실험 07-4 (V4): 깊이 모델로 2D 겹침 오정지를 거르는가
# 손목이 회전체와 겹친 프레임마다 Depth Anything V2 Small(CPU)로 상대 깊이를 구한다. 값이 클수록 카메라에 가깝다
# 실행 전에 정한 규칙: 손목 깊이가 주변 회전체 깊이보다 10% 넘게 크면(확실히 앞) 겹침을 무시한다
import time
import numpy as np, cv2, torch
from PIL import Image
from transformers import pipeline
from common import CASES, DATA, load_pose, rot_polys, torso, metrics, done, KP_MIN, frames

MARGIN = 0.10
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cpu")
res, lat = {}, []
for name, cfg in CASES.items():
    P = load_pose(name); t = P["t"]
    imgs = [img for _, img in frames(DATA + f"raw_cctv/{name}.mp4", cfg["t0"], cfg["t1"])]
    v1, v4, kept = [], [], 0
    for i, (kps, sc) in enumerate(zip(P["kps"], P["sc"])):
        polys = rot_polys(name, P, i); h, w = imgs[min(i, len(imgs) - 1)].shape[:2]
        hits = [(k, s, j) for k, s in zip(kps, sc) for j in (9, 10)
                if s[j] >= KP_MIN and max(cv2.pointPolygonTest(p.astype(np.float32), (float(k[j][0]), float(k[j][1])), False) for p in polys) >= 0]
        v1.append(bool(hits))
        if not hits: v4.append(False); continue
        t0 = time.perf_counter()
        depth = np.array(pipe(Image.fromarray(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)))["predicted_depth"].squeeze().float())
        lat.append((time.perf_counter() - t0) * 1000)
        depth = cv2.resize(depth, (w, h))
        mask = np.zeros((h, w), np.uint8)
        for p in polys: cv2.fillPoly(mask, [p.astype(np.int32)], 1)
        stop = False
        for k, s, j in hits:
            x, y = int(np.clip(k[j][0], 3, w - 4)), int(np.clip(k[j][1], 3, h - 4))
            wd = float(np.median(depth[y - 3:y + 4, x - 3:x + 4]))
            tl = torso(k, s) or 0.15 * h
            ring = np.zeros_like(mask); cv2.circle(ring, (x, y), int(1.5 * tl), 1, -1); cv2.circle(ring, (x, y), int(0.5 * tl), 0, -1)
            pts = [k[q] for q in range(17) if s[q] >= KP_MIN]  # 사람 몸 영역은 회전체 깊이에서 뺀다
            x0, y0 = np.min(pts, 0) - 0.1 * tl; x1, y1 = np.max(pts, 0) + 0.1 * tl
            ring[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = 0
            sel = depth[(ring > 0) & (mask > 0)]
            if sel.size < 30: sel = depth[mask > 0]
            if wd <= float(np.median(sel)) * (1 + MARGIN): stop = True
            else: kept += 1
        v4.append(stop)
    res[name] = {"V1_overlap": metrics(t, v1, cfg["tc"]), "V4_depth_gate": metrics(t, v4, cfg["tc"]), "overlap_frames": int(sum(v1)), "ignored_as_in_front": kept}
    print(name, res[name])
res["depth_cpu_ms"] = {"median": round(float(np.median(lat)), 1), "p95": round(float(np.percentile(lat, 95)), 1), "n": len(lat)}
res["margin"] = MARGIN
done("t04_depth_gate", res)
