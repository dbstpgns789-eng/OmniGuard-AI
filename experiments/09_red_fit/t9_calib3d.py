# 09-9 (V6): 카메라 캘리브레이션으로 원통과 작업자를 3D로 복원하고, 손과 원통 표면 사이 실제 거리(m)로 판단한다
# 1) 사용자가 찍은 바닥 선 4개 → 소실점 2개 → 초점 거리와 카메라 기울기 (주점은 화면 중앙 가정)
# 2) 축척: 똑바로 선 프레임에서 발목~코 높이 = 1.5m(키 약 1.6m)로 카메라 높이를 정함
# 3) 원통: 축 방향 = 설비 방향 소실점, 축은 두 원판 중심을 지남. 남는 미지수(깊이) 하나는 사용자 판단에 맞춰 고름(이 영상에 맞춤)
# 4) 손 위치는 보수적으로: 손목 시선 위에서, 어깨로부터 팔 길이 0.7m 안에 있으면서 원통에 가장 가까운 점
# 화면 보정 행렬 H로 기준 프레임 좌표를 각 프레임으로 옮긴다(확대 없이 옆 이동만 있는 영상이라 카메라 내부값은 같다고 봄)
import os, json, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestClassifier
from common import *

ZONE = json.load(open(os.path.join(HERE, "zone_user.json"))); FLOOR = json.load(open(os.path.join(HERE, "floor_user.json")))
NOSE_H, ARM = 1.5, 0.7
P = load_pose("performance"); t = P["t"]; y = load_labels(t); w, h = P["wh"]; n_fr = len(t)

# 1) 카메라
Lp = np.array(FLOOR["lines"]) * [w, h]
hom = lambda p: np.array([p[0], p[1], 1.0])
vp = lambda a, b, c, d: (lambda v: v[:2] / v[2])(np.cross(np.cross(hom(a), hom(b)), np.cross(hom(c), hom(d))))
v1, v2 = vp(*Lp[0:4]), vp(*Lp[4:8]); c0 = np.array([w / 2, h / 2])
f = float(np.sqrt(-np.dot(v1 - c0, v2 - c0)))
K = np.array([[f, 0, c0[0]], [0, f, c0[1]], [0, 0, 1]]); Ki = np.linalg.inv(K)
r1 = Ki @ hom(v1); r1 /= np.linalg.norm(r1); r2 = Ki @ hom(v2); r2 -= r1 * np.dot(r1, r2); r2 /= np.linalg.norm(r2)
nrm = np.cross(r1, r2); nrm /= np.linalg.norm(nrm)
if np.dot(nrm, Ki @ hom((w / 2, h))) > 0: nrm = -nrm  # 바닥점을 향한 시선과 반대쪽(위쪽)을 향하게

def ray(px, i):  # i번째 프레임 픽셀 → 기준 프레임 픽셀 → 카메라 시선(단위 벡터)
    q = cv2.perspectiveTransform(np.array(px, np.float64).reshape(-1, 1, 2), np.linalg.inv(P["H"][i])).reshape(-1, 2)
    d = (Ki @ np.c_[q, np.ones(len(q))].T).T; return d / np.linalg.norm(d, axis=1, keepdims=True)

def project(X, i):  # 카메라 좌표 3D 점 → i번째 프레임 픽셀
    p = (K @ X.T).T; p = p[:, :2] / p[:, 2:3]
    return cv2.perspectiveTransform(p.reshape(-1, 1, 2), P["H"][i]).reshape(-1, 2)

def floor_pt(d, Hc): return -Hc * d / np.dot(nrm, d)  # 바닥(높이 0) 위 점, 카메라 높이 Hc

def closest_on_line(p0, u, d):  # 직선 p0+s*u 위에서 원점 시선 d에 가장 가까운 점의 s
    A = np.array([[np.dot(u, u), -np.dot(u, d)], [np.dot(u, d), -np.dot(d, d)]]); b = np.array([-np.dot(p0, u), -np.dot(p0, d)])
    return np.linalg.solve(A, b)[0]

# 2) 축척(Hc=1로 계산한 코 높이 비율 → 실제 카메라 높이)
ratios = []
for i in range(n_fr):
    for k, s in zip(P["kps"][i], P["sc"][i]):
        if min(s[0], s[15], s[16], s[5], s[6], s[11], s[12]) < KP_MIN: continue
        v = (k[5] + k[6]) / 2 - (k[11] + k[12]) / 2
        if np.degrees(np.arctan2(abs(v[0]), -v[1])) > 8: continue
        Xf = floor_pt(ray([(k[15] + k[16]) / 2], i)[0], 1.0); sh = closest_on_line(Xf, nrm, ray([k[0]], i)[0])
        if sh > 0: ratios.append(sh)
Hc = NOSE_H / float(np.median(ratios))

# 3) 원통: 두 원판 중심 시선 위의 점 a=λ1·d1, b=λ2·d2 이고 b-a ∥ r1. λ1 하나가 정해지면 나머지가 정해진다
cent = [np.mean(np.array(o) * [w, h], axis=0) for o in ZONE["outlines"]]
ref_i = int(np.argmin(np.abs(t - ZONE["drawn_on_t"])))
d1, d2 = (Ki @ hom(cent[0])), (Ki @ hom(cent[1])); d1 /= np.linalg.norm(d1); d2 /= np.linalg.norm(d2)
M = np.c_[d2, -r1]  # λ2·d2 - μ·r1 = λ1·d1
def axis_for(l1):
    sol = np.linalg.lstsq(M, l1 * d1, rcond=None)[0]; return l1 * d1, sol[0] * d2
radius_px = np.mean([np.sqrt(cv2.contourArea((np.array(o) * [w, h]).astype(np.float32)) / np.pi) for o in ZONE["outlines"]])

def seg_d(Xs, a, b):
    ab = b - a; u = np.clip(((Xs - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(Xs - (a + u[:, None] * ab), axis=1)

def hand_dist(i, a, b, R):  # 프레임 i에서 가장 위험한 사람의 손-원통 표면 최소 거리(보수적), 발-축 수평 거리
    best = (np.nan, np.nan)
    for k, s in zip(P["kps"][i], P["sc"][i]):
        if max(s[15], s[16]) < KP_MIN or max(s[5], s[6]) < KP_MIN: continue
        ank = np.mean([k[j] for j in (15, 16) if s[j] >= KP_MIN], axis=0); Xf = floor_pt(ray([ank], i)[0], Hc)
        shp = np.mean([k[j] for j in (5, 6) if s[j] >= KP_MIN], axis=0); Sh = Xf + closest_on_line(Xf, nrm, ray([shp], i)[0]) * nrm
        dmin = np.nan
        for j in (9, 10):
            if s[j] < KP_MIN: continue
            dw = ray([k[j]], i)[0]; tc = np.dot(Sh, dw); disc = tc ** 2 - (np.dot(Sh, Sh) - ARM ** 2)
            ts = np.linspace(tc - np.sqrt(disc), tc + np.sqrt(disc), 25) if disc > 0 else np.array([tc])
            dd = float(np.min(seg_d(ts[:, None] * dw, a, b)) - R); dmin = dd if np.isnan(dmin) else min(dmin, dd)
        if not np.isnan(dmin) and (np.isnan(best[0]) or dmin < best[0]): best = (dmin, None)
    return best

def stage(d, th): return 0 if np.isnan(d) else 3 if d < th[0] else 2 if d < th[1] else 1 if d < th[2] else 0

# 원통 깊이(λ1) 후보마다 거리를 구하고, 임계값까지 사람 판단에 맞춘다
lam_floor = np.linalg.norm(floor_pt(d1, Hc))  # 원판 중심 시선이 바닥에 닿는 거리 = 깊이 상한
cands = []
for l1 in np.linspace(0.5 * lam_floor, 0.99 * lam_floor, 25):
    a, b = axis_for(l1); R = radius_px * l1 / f; hgt = np.dot(nrm, (a + b) / 2) + Hc
    if hgt - R < 0.02: continue  # 원통이 바닥을 뚫으면 제외
    D = np.array([hand_dist(i, a, b, R)[0] for i in range(0, n_fr, 3)]); Y = y[::3]
    rng = np.random.default_rng(0); bf = (-1, None)
    for _ in range(3000):
        th = np.sort(rng.uniform(-0.2, 1.5, 3)); p = smooth(np.array([stage(d, th) for d in D]), 3)
        s_ = macro_f1(Y, p)[0]
        if s_ > bf[0]: bf = (s_, th)
    cands.append((bf[0], l1, R, hgt, bf[1])); print(f"depth {l1:6.2f}m R {R:.2f}m axis_h {hgt:.2f}m f1 {bf[0]:.3f}", flush=True)
score, l1, R, hgt, th = max(cands, key=lambda x: x[0]); a, b = axis_for(l1)
D = np.array([hand_dist(i, a, b, R)[0] for i in range(n_fr)]); pr = smooth(np.array([stage(d, th) for d in D]), 3)

fps = 1 / np.median(np.diff(t)); win, lag = int(fps), max(1, int(0.3 * fps)); Dff = D.copy()
for i in range(1, n_fr):
    if np.isnan(Dff[i]): Dff[i] = Dff[i - 1]
X = np.array([[Dff[i], np.nanmin(Dff[max(0, i - win + 1):i + 1]), np.nanmean(Dff[max(0, i - win + 1):i + 1]),
               (Dff[i - lag] - Dff[i]) * fps / lag if i >= lag else 0] for i in range(n_fr)], float); X = np.nan_to_num(X, nan=9.0)
m = y >= 0; mk = lambda: RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", random_state=0)
cv = np.full(n_fr, -1)
for blk in np.array_split(np.arange(n_fr), 5):
    tr = np.ones(n_fr, bool); tr[blk] = False; tr &= m; cv[blk] = mk().fit(X[tr], y[tr]).predict(X[blk])
pc = smooth(mk().fit(X[m], y[m]).predict(X), 5); cv = smooth(cv, 5)
dY = {STAGES[c]: round(float(np.nanmedian(D[(y == c)])), 3) for c in range(4)}
res = {"camera": {"focal_px": round(f, 1), "camera_height_m": round(Hc, 2), "vp_machine": v1.round(1).tolist(), "vp_perp": v2.round(1).tolist()},
       "cylinder": {"depth_m": round(l1, 2), "radius_m": round(R, 2), "axis_height_m": round(hgt, 2), "length_m": round(float(np.linalg.norm(b - a)), 2)},
       "rule_3d": {"th_stop_slow_warn_m": th.round(3).tolist(), "macro_f1": macro_f1(y, pr), "stop_onsets": onsets(t, pr, 3)},
       "clf_3d_fit_all": {"macro_f1": macro_f1(y, pc), "stop_onsets": onsets(t, pc, 3)},
       "clf_3d_cv": {"macro_f1": macro_f1(y, cv), "stop_onsets": onsets(t, cv, 3)},
       "median_hand_dist_m_by_human_stage": dY, "human_stop_onsets": onsets(t, np.where(y < 0, 0, y), 3)}
print(json.dumps(res, ensure_ascii=False, default=float), flush=True)
np.save(OUT + "/t9_rule3d.npy", pr); np.save(OUT + "/t9_clf3d.npy", pc); np.save(OUT + "/t9_dist3d.npy", D)

# 데모: 3D 원통 표면과 바깥 띠(반지름 R+임계값)를 원근 그대로 화면에 투영
COL = {0: (90, 90, 90), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 255), -1: (40, 40, 40)}
FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 26); SM = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 16)
u_ax = (b - a) / np.linalg.norm(b - a); e1 = np.cross(u_ax, nrm); e1 /= np.linalg.norm(e1); e2 = np.cross(u_ax, e1)
def tube(rad, i):
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False); ring = np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2
    pts = np.vstack([c + rad * ring for c in (a, b)]); return cv2.convexHull(project(pts, i).astype(np.float32)).astype(np.int32)
cap = cv2.VideoCapture(DATA + "raw_cctv/rec_002103.mp4"); fpsv = cap.get(5); cap.set(1, int(round(t[0] * fpsv))); wr = None
for i in range(n_fr):
    ok, img = cap.read()
    if not ok: break
    for lvl, rad in ((1, R + max(th[2], 0)), (2, R + max(th[1], 0)), (3, R + max(th[0], 0))):
        hl = tube(rad, i); ov = img.copy(); cv2.fillPoly(ov, [hl], COL[lvl]); img = cv2.addWeighted(ov, 0.15, img, 0.85, 0); cv2.polylines(img, [hl], True, COL[lvl], 2)
    hl = tube(R, i); ov = img.copy(); cv2.fillPoly(ov, [hl], (0, 200, 0)); img = cv2.addWeighted(ov, 0.25, img, 0.75, 0); cv2.polylines(img, [hl], True, (0, 200, 0), 2)
    pa = project(np.array([a, b]), i).astype(int); cv2.line(img, tuple(pa[0]), tuple(pa[1]), (255, 80, 0), 3)
    for k, s in zip(P["kps"][i], P["sc"][i]):
        for j in range(17):
            if s[j] >= KP_MIN: cv2.circle(img, (int(k[j][0]), int(k[j][1])), 6 if j in (9, 10) else 3, (0, 255, 0) if j in (9, 10) else (255, 255, 0), -1)
    lv = int(pc[i]); cv2.rectangle(img, (0, 0), (w, 48), COL[lv], -1)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); dtxt = "-" if np.isnan(D[i]) else f"{D[i] * 100:.0f}cm"
    ImageDraw.Draw(pil).text((10, 8), f"모델: {STAGES[lv]}  사람: {STAGES[y[i]] if y[i] >= 0 else '-'}  손-롤 {dtxt}  t={t[i]:4.1f}s", font=FONT, fill=(255, 255, 255))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR); bars = []
    for lab, arr in (("사람", y), ("3D규칙", pr), ("3D분류기", pc)):
        bar = np.full((24, w, 3), 25, np.uint8)
        for j in range(i + 1): x = 80 + int(j * (w - 80) / n_fr); cv2.line(bar, (x, 2), (x, 21), COL[int(arr[j])], max(1, (w - 80) // n_fr + 1))
        p2 = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)); ImageDraw.Draw(p2).text((4, 2), lab, font=SM, fill=(255, 255, 255))
        bars.append(cv2.cvtColor(np.array(p2), cv2.COLOR_RGB2BGR))
    frame = np.vstack([img] + bars)
    if wr is None: wr = cv2.VideoWriter(DATA + "exp09/demo_calib3d.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fpsv, (frame.shape[1], frame.shape[0]))
    wr.write(frame)
    if i in (int(n_fr * 0.07), int(n_fr * 0.35), int(n_fr * 0.6), int(n_fr * 0.75)): cv2.imwrite(DATA + f"exp09/demo_calib3d_{i}.jpg", frame)
wr.release()
done("t9_calib3d", res)
