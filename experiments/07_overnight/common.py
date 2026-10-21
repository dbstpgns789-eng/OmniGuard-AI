# 밤샘 실험 공통: 경로, 사례 정의(실험 05·06과 같은 값), 포즈 캐시 읽기, 간단한 추적, 측정
import json, os, time
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DATA = "C:/Users/User/Desktop/OmniGuard-data/"
CACHE = DATA + "cache/"
POSE06 = os.path.join(HERE, "..", "06_body_parts")
KP_MIN, NEED_STOP = 0.3, 3

CASES = {
    "rec_001918": dict(t0=4.0, t1=18.5, tc=15.0, ref_t=16.0, stabilize=False,
                       rot=[[(0.15, 0.28), (0.52, 0.20), (0.57, 0.50), (0.53, 0.70), (0.20, 0.54)]]),
    "rec_002103": dict(t0=7.0, t1=46.5, tc=37.0, ref_t=36.0, stabilize=True,
                       rot=[[(0.24, 0.18), (0.95, 0.25), (0.95, 0.36), (0.80, 0.40), (0.62, 0.48), (0.45, 0.45), (0.30, 0.40)],
                            [(0.78, 0.36), (1.0, 0.33), (1.0, 0.80), (0.80, 0.80), (0.76, 0.55)]]),
}

def outdir(task):
    d = os.path.join(OUT, task); os.makedirs(d, exist_ok=True); return d

def done(task, result):
    d = outdir(task)
    json.dump(result, open(os.path.join(d, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(d, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))

def load_pose(name):
    return dict(np.load(os.path.join(POSE06, f"{name}_pose.npz"), allow_pickle=True))

def rot_polys(name, P, i):
    w, h = P["wh"]; M = P["M"][i]
    return [np.c_[np.array(p) * [w, h], np.ones(len(p))] @ M.T for p in CASES[name]["rot"]]

def signed_dist(pt, polys):  # 양수: 다각형 안, 음수: 바깥(가장 가까운 다각형까지 거리)
    return max(cv2.pointPolygonTest(p.astype(np.float32), (float(pt[0]), float(pt[1])), True) for p in polys)

def torso(k, s):
    if min(s[5], s[6], s[11], s[12]) < KP_MIN: return None
    return float(np.linalg.norm((k[5] + k[6]) / 2 - (k[11] + k[12]) / 2)) or None

def center(k, s):
    pts = [k[j] for j in range(17) if s[j] >= KP_MIN]
    return np.mean(pts, axis=0) if pts else None

def track(P):
    """프레임 사이 사람을 몸 중심 거리로 이어 붙인다. 반환: 프레임마다 [사람 id] 목록"""
    ids, prev, nxt = [], [], 0
    for kps, sc in zip(P["kps"], P["sc"]):
        cur = []
        for k, s in zip(kps, sc):
            c = center(k, s); best, bd = None, 1e9
            for pid, pc, ps in prev:
                if c is not None and pc is not None:
                    d = np.linalg.norm(c - pc)
                    if d < bd and d < 0.15 * max(P["wh"]): best, bd = pid, d
            if best is None or best in [x[0] for x in cur]: best, nxt = nxt, nxt + 1
            cur.append((best, c, s))
        ids.append([x[0] for x in cur]); prev = cur
    return ids

def confirm(raw, need=NEED_STOP):
    return [i >= need - 1 and all(raw[i - need + 1:i + 1]) for i in range(len(raw))]

def metrics(t, raw, tc):
    stop = confirm(list(raw))
    pre = [s for s, x in zip(stop, t) if x < tc - 3]
    near = [s for s, x in zip(stop, t) if tc - 0.5 <= x <= tc + 0.5]
    first = next((x for s, x in zip(stop, t) if s and tc - 3 <= x <= tc), None)
    return {"pre_approach_stop_ratio": round(float(np.mean(pre)), 3) if pre else None,
            "stop_ratio_tc_pm0.5": round(float(np.mean(near)), 3) if near else None,
            "lead_s": None if first is None else round(float(tc - first), 2)}

def frames(path, t0, t1, step=1):
    cap = cv2.VideoCapture(path); fps = cap.get(5); cap.set(1, int(t0 * fps)); f = int(t0 * fps)
    while f < int(t1 * fps):
        ok, img = cap.read()
        if not ok: break
        if (f - int(t0 * fps)) % step == 0: yield f / fps, img
        f += 1
    cap.release()
