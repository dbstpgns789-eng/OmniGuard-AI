# 실험 08: 공장 사고 영상 5건에서 V1(겹침만)과 V3(손의 접근 움직임)을 프레임별 정답으로 채점하고, 데모 영상을 만든다
# 정답(labels.json)은 시스템 출력을 보기 전에 정했다. V3 기준은 실험 07과 같다(접근 1.2 몸통/초, 창 1초, TTC 0.5초)
# 측정: 닿음(Y) 프레임 중 정지 비율, 안 닿음(N) 프레임 중 정지 비율(오정지), 첫 닿음 전에 정지가 떴는지와 몇 초 전인지
import json, os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "C:/Users/User/Desktop/OmniGuard-data/"; EXP = DATA + "exp08/"
CASES = json.load(open(os.path.join(HERE, "cases.json"), encoding="utf-8"))
LABELS = json.load(open(os.path.join(HERE, "labels.json"), encoding="utf-8"))
KP_MIN, NEED, V_TH, WIN_S, TTC_TH, LAG_S = 0.3, 3, 1.2, 1.0, 0.5, 0.3
COLOR = {0: (90, 90, 90), 3: (0, 0, 255)}
FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22)

def label_of(name, t):
    s = int(round(t))
    for k in ("Y", "N", "U"):
        if any(a <= s <= b for a, b in LABELS[name][k]): return k
    return "U"

def polys_at(c, wh, H):
    w, h = wh
    return [cv2.perspectiveTransform((np.array(p, np.float64) * [w, h]).reshape(-1, 1, 2), H).reshape(-1, 2) for p in c["rot"]]

def sdist(pt, polys): return max(cv2.pointPolygonTest(p.astype(np.float32), (float(pt[0]), float(pt[1])), True) for p in polys)

def track(kps_all, sc_all, diag):
    ids, prev, nxt = [], [], 0
    for kps, sc in zip(kps_all, sc_all):
        cur = []
        for k, s in zip(kps, sc):
            g = s >= KP_MIN; c = k[g].mean(0) if g.any() else None; best, bd = None, 0.15 * diag
            for pid, pc in prev:
                if c is not None and pc is not None and np.linalg.norm(c - pc) < bd and pid not in [x[0] for x in cur]: best, bd = pid, np.linalg.norm(c - pc)
            if best is None: best, nxt = nxt, nxt + 1
            cur.append((best, c))
        ids.append([x[0] for x in cur]); prev = cur
    return ids

def confirm(raw): return [i >= NEED - 1 and all(raw[i - NEED + 1:i + 1]) for i in range(len(raw))]

def run(name):
    c = CASES[name]; P = dict(np.load(EXP + f"{name}_pose.npz", allow_pickle=True)); t = P["t"]; wh = P["wh"]
    fps = 1 / np.median(np.diff(t)); lag, win = max(1, int(LAG_S * fps)), int(WIN_S * fps)
    ids = track(P["kps"], P["sc"], float(np.hypot(*wh))); hist, torso_mem, v1, v3 = {}, {}, [], []
    for i, (kps, sc) in enumerate(zip(P["kps"], P["sc"])):
        polys = polys_at(c, wh, P["H"][i]); a = b = False
        for pid, k, s in zip(ids[i], kps, sc):
            ds = [sdist(k[j], polys) for j in (9, 10) if s[j] >= KP_MIN]
            if not ds: continue
            inside = max(ds) >= 0; a |= inside
            if min(s[5], s[6], s[11], s[12]) >= KP_MIN: torso_mem[pid] = np.linalg.norm((k[5] + k[6]) / 2 - (k[11] + k[12]) / 2)
            tl = torso_mem.get(pid)
            if not tl: continue
            d = -max(ds) / tl; h = hist.setdefault(pid, []); h.append((i, d))
            vs = [(h[x - lag][1] - h[x][1]) / ((h[x][0] - h[x - lag][0]) / fps) for x in range(lag, len(h)) if h[x][0] > i - win]
            vmax = max(vs) if vs else 0.0; vnow = vs[-1] if vs and h[-1][0] == i else 0.0
            if (inside and vmax >= V_TH) or (not inside and vnow >= V_TH and d / vnow < TTC_TH): b = True
        v1.append(a); v3.append(b)
    labs = [label_of(name, x) for x in t]; first_y = next((x for x, l in zip(t, labs) if l == "Y"), None)
    res = {}
    for key, raw in (("V1_overlap", v1), ("V3_motion", v3)):
        stop = confirm(raw)
        y = [s for s, l in zip(stop, labs) if l == "Y"]; n = [s for s, l in zip(stop, labs) if l == "N"]
        # 첫 닿음 직전 3초 안에서 처음 뜬 정지 (접근 단계에서 멈췄는가)
        pre = [x for s, x in zip(stop, t) if s and first_y is not None and first_y - 3 <= x < first_y]
        res[key] = {"stop_ratio_on_Y": round(float(np.mean(y)), 3) if y else None, "false_stop_ratio_on_N": round(float(np.mean(n)), 3) if n else None,
                    "stopped_within_3s_before_first_Y": bool(pre), "lead_s": round(float(first_y - pre[0]), 2) if pre else None}
    res["frames"] = {"Y": labs.count("Y"), "N": labs.count("N"), "U": labs.count("U")}; res["first_Y_t"] = first_y
    return res, t, labs, confirm(v1), confirm(v3), P

def render(name, t, labs, s1, s3, P):
    c = CASES[name]; cap = cv2.VideoCapture(DATA + "raw_cctv/" + c["file"]); fps = cap.get(5); cap.set(1, int(round(c["t0"] * fps)))
    w, h = P["wh"]; scale = 960 / w; W, Hh = 960, int(h * scale); n = len(t); wr = None
    LC = {"Y": (0, 0, 255), "N": (0, 180, 0), "U": (110, 110, 110)}
    for i in range(n):
        ok, img = cap.read()
        if not ok: break
        for p in polys_at(c, P["wh"], P["H"][i]): cv2.polylines(img, [p.astype(np.int32)], True, (0, 0, 255), max(2, int(w / 400)))
        for k, s in zip(P["kps"][i], P["sc"][i]):
            for j in range(17):
                if s[j] >= KP_MIN: cv2.circle(img, (int(k[j][0]), int(k[j][1])), max(3, int(w / 300)), (0, 255, 0) if j in (9, 10) else (255, 255, 0), -1)
        img = cv2.resize(img, (W, Hh)); lv = 3 if s3[i] else 0
        cv2.rectangle(img, (0, 0), (W, 40), COLOR[lv], -1)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text((10, 7), f"{'정지' if lv else '정상'}   t={t[i]:5.1f}s   정답: {dict(Y='닿음', N='안 닿음', U='판단 불가')[labs[i]]}", font=FONT, fill=(255, 255, 255))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR); bars = []
        for lab, arr in (("정답", [LC[x] for x in labs]), ("V1", [COLOR[3 if x else 0] for x in s1]), ("V3", [COLOR[3 if x else 0] for x in s3])):
            bar = np.full((22, W, 3), 25, np.uint8)
            for j in range(i + 1): x = 50 + int(j * (W - 50) / n); cv2.line(bar, (x, 2), (x, 19), arr[j], max(1, int((W - 50) / n) + 1))
            p2 = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)); ImageDraw.Draw(p2).text((4, 1), lab, font=ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 15), fill=(255, 255, 255))
            bars.append(cv2.cvtColor(np.array(p2), cv2.COLOR_RGB2BGR))
        frame = np.vstack([img] + bars)
        if wr is None: wr = cv2.VideoWriter(EXP + f"demo_{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        wr.write(frame)
    wr.release()

summary = {}
for name in (sys.argv[1:] or CASES):
    res, t, labs, s1, s3, P = run(name); summary[name] = res; print(name, json.dumps(res, ensure_ascii=False), flush=True)
    render(name, t, labs, s1, s3, P)
json.dump(summary, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
