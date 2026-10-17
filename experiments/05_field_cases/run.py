# 실험 05: 현장 사고 CCTV 사례 연구. 손이 회전체에 처음 닿는 시점(Tc) 전에 정지가 뜨는가
# 바꾸는 변수는 하나: V1 화면상 겹침만 / V2 발이 팔이 닿는 거리 안 + 화면상 겹침
# 포즈는 프레임마다 한 번만 추정하고 두 규칙에 같이 넣는다. 영상은 레포 밖에 있고 결과 영상도 레포 밖에 저장한다
import csv, json, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from rtmlib import Body

DATA = "C:/Users/User/Desktop/OmniGuard-data/"
# 좌표는 기준 프레임(ref_t)의 화면 비율. 회전체는 다각형 여러 개, 바닥선은 설비 아래 레일이 바닥과 만나는 선
CASES = {
    "rec_001918": dict(t0=4.0, t1=18.5, tc=15.0, ref_t=16.0, stabilize=False,
                       rot=[[(0.15, 0.28), (0.52, 0.20), (0.57, 0.50), (0.53, 0.70), (0.20, 0.54)]],
                       floor=[(0.18, 0.58), (0.52, 0.80)]),
    "rec_002103": dict(t0=7.0, t1=46.5, tc=37.0, ref_t=36.0, stabilize=True,
                       rot=[[(0.24, 0.18), (0.95, 0.25), (0.95, 0.36), (0.80, 0.40), (0.62, 0.48), (0.45, 0.45), (0.30, 0.40)],
                            [(0.78, 0.36), (1.0, 0.33), (1.0, 0.80), (0.80, 0.80), (0.76, 0.55)]],
                       floor=[(0.30, 0.48), (1.0, 0.88)]),
}
HEIGHT_M, FORESHORTEN, REACH_M, WARN_M = 1.6, 0.65, 1.0, 2.0
KP_MIN, NEED = 0.3, {1: 1, 2: 2, 3: 3}
NAME = {0: "반응 없음", 1: "주의", 2: "감속", 3: "정지"}
COLOR = {0: (90, 90, 90), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 255)}
FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22)

def in_any(pt, polys): return any(cv2.pointPolygonTest(p.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0 for p in polys)
def upright(k, s):
    if min(s[5], s[6], s[11], s[12]) < KP_MIN: return False
    v = (k[5] + k[6]) / 2 - (k[11] + k[12]) / 2
    return np.degrees(np.arctan2(abs(v[0]), -v[1])) < 15

def judge(k, s, rot, floor, px_per_m):
    hand = any(s[j] >= KP_MIN and in_any(k[j], rot) for j in (9, 10))
    v1 = 3 if hand else 0
    d = None
    if max(s[15], s[16]) >= KP_MIN and px_per_m:
        foot = np.mean([k[j] for j in (15, 16) if s[j] >= KP_MIN], axis=0)
        (x0, y0), (x1, y1) = floor
        line_y = y0 + (y1 - y0) * (foot[0] - x0) / (x1 - x0)
        d = max(0.0, (foot[1] - line_y) / px_per_m)
    if d is None: v2 = 0  # 발이 안 보이면 거리를 잴 수 없다
    elif d < REACH_M: v2 = 3 if hand else 2
    elif d < WARN_M: v2 = 1
    else: v2 = 0
    return v1, v2, d

def confirm(hist):
    return max([L for L in (3, 2, 1) if len(hist) >= NEED[L] and min(hist[-NEED[L]:]) >= L] or [0])

body = Body(mode="lightweight", backend="onnxruntime", device="cpu")
orb, bf = cv2.ORB_create(2000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
summary = {}
for name in (sys.argv[1:] or CASES):
    cfg = CASES[name]; cap = cv2.VideoCapture(DATA + f"raw_cctv/{name}.mp4"); fps = cap.get(5)
    cap.set(1, int(cfg["ref_t"] * fps)); ok, ref = cap.read(); h, w = ref.shape[:2]
    kr, dr = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    rot0 = [np.array(p) * [w, h] for p in cfg["rot"]]; floor0 = np.array(cfg["floor"]) * [w, h]
    cap.set(1, int(cfg["t0"] * fps))
    writer, rows, h1, h2, px_per_m, M = None, [], [], [], None, np.eye(2, 3)
    for f in range(int(cfg["t0"] * fps), int(cfg["t1"] * fps)):
        ok, img = cap.read()
        if not ok: break
        t = f / fps
        if cfg["stabilize"]:  # 기준 프레임 → 현재 프레임 이동량을 배경 특징점으로 재서 구역을 따라 옮긴다
            k, d = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            m = sorted(bf.match(dr, d), key=lambda x: x.distance)[:400] if d is not None else []
            if len(m) >= 20:
                A, inl = cv2.estimateAffinePartial2D(np.float32([kr[x.queryIdx].pt for x in m]), np.float32([k[x.trainIdx].pt for x in m]), ransacReprojThreshold=4)
                if A is not None and inl.sum() >= 15: M = A
        tr = lambda P: (np.c_[P, np.ones(len(P))] @ M.T)
        rot, floor = [tr(p) for p in rot0], tr(floor0)
        kps, sc = body(img)
        for kk, ss in zip(kps, sc):  # 축척: 몸통이 곧게 선 프레임의 키로 갱신하고, 숙이면 마지막 값을 쓴다
            if upright(kk, ss) and max(ss[15], ss[16]) >= KP_MIN and max(ss[:5]) >= KP_MIN:
                top = min(kk[j][1] for j in range(5) if ss[j] >= KP_MIN); foot = max(kk[j][1] for j in (15, 16) if ss[j] >= KP_MIN)
                px_per_m = (foot - top) * 1.1 / HEIGHT_M * FORESHORTEN
        per = [judge(kk, ss, rot, floor, px_per_m) for kk, ss in zip(kps, sc)]
        h1.append(max([p[0] for p in per] or [0])); h2.append(max([p[1] for p in per] or [0]))
        s1, s2 = confirm(h1), confirm(h2)
        dist = min([p[2] for p in per if p[2] is not None] or [None], key=lambda x: 99 if x is None else x)
        rows.append({"t": round(t, 3), "people": len(per), "v1": s1, "v2": s2, "foot_m": None if dist is None else round(dist, 2)})
        # 결과 영상: 회전체, 바닥선, 관절, V2 배너, 아래에 V1·V2 기록 막대와 Tc 표시
        vis = img.copy(); over = vis.copy()
        for p in rot: cv2.fillPoly(over, [p.astype(np.int32)], (0, 0, 255))
        vis = cv2.addWeighted(over, 0.2, vis, 0.8, 0)
        for p in rot: cv2.polylines(vis, [p.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.line(vis, tuple(floor[0].astype(int)), tuple(floor[1].astype(int)), (255, 255, 255), 2)
        for kk, ss in zip(kps, sc):
            for j in range(17):
                if ss[j] >= KP_MIN: cv2.circle(vis, tuple(int(v) for v in kk[j]), 6 if j in (9, 10, 15, 16) else 3, (0, 255, 0) if j in (9, 10) else (255, 0, 255) if j in (15, 16) else (255, 255, 0), -1)
        cv2.rectangle(vis, (0, 0), (w, 40), COLOR[s2], -1)
        pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); dr_ = ImageDraw.Draw(pil)
        dr_.text((8, 7), f"V2 {NAME[s2]}  발-설비 {'-' if dist is None else f'{dist:.1f}m'}  t={t:4.1f}s  (Tc={cfg['tc']}s)", font=FONT, fill=(255, 255, 255))
        stack, n_all = [cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)], int((cfg["t1"] - cfg["t0"]) * fps)
        for v in ("v1", "v2"):
            bar = np.full((24, w, 3), 30, np.uint8)
            for i, r in enumerate(rows):
                x = 40 + int(i * (w - 40) / n_all); cv2.line(bar, (x, 2), (x, 21), COLOR[r[v]], 2)
            xc = 40 + int((cfg["tc"] - cfg["t0"]) * fps * (w - 40) / n_all); cv2.line(bar, (xc, 0), (xc, 23), (255, 255, 255), 2)
            cv2.putText(bar, v.upper(), (4, 17), 0, 0.5, (255, 255, 255), 1); stack.append(bar)
        frame = np.vstack(stack)
        if writer is None: writer = cv2.VideoWriter(DATA + f"demo/{name}_exp05.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    writer.release()
    with open(f"{name}_frames.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    # 측정값
    tc = cfg["tc"]; res = {"Tc": tc, "frames": len(rows), "person_rate": round(np.mean([r["people"] > 0 for r in rows]), 3)}
    for v in ("v1", "v2"):
        idx_tc = max(i for i, r in enumerate(rows) if r["t"] <= tc)
        if rows[idx_tc][v] == 3:  # Tc에 정지 중이면, 그 정지가 연속으로 시작된 시점
            i = idx_tc
            while i > 0 and rows[i - 1][v] == 3: i -= 1
            lead = round(tc - rows[i]["t"], 2)
        else: lead = None
        pre = [r[v] == 3 for r in rows if r["t"] < tc - 3]
        res[v] = {"lead_s": lead, "pre_approach_stop_ratio": round(np.mean(pre), 3) if pre else None,
                  "levels": {NAME[L]: sum(r[v] == L for r in rows) for L in range(4)}}
    summary[name] = res; print(name, json.dumps(res, ensure_ascii=False))
json.dump(summary, open("result.json", "w"), ensure_ascii=False, indent=1)
