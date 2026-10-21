# 실험 09 공통: 빨간 상의 편집본(red_crop)에서 사용자 판단(정상/주의/감속/정지)에 모델 출력을 맞춘다
# 위험 구역은 다각형 대신 원통(캡슐): 화면에서 롤의 중심축 선분과 반지름. 손목에서 축까지 거리 - 반지름 = 표면까지 거리
import csv, json, os, time
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
DATA = "C:/Users/User/Desktop/OmniGuard-data/"
EXP08 = DATA + "exp08/"
LABEL_CSV = DATA + "exp08/frames_for_review/red_crop/판단표.csv"
STAGES = ["정상", "주의", "감속", "정지"]
KP_MIN = 0.3

# 기준 프레임(36.0초) 화면 비율 좌표. 축은 롤 몸통 왼쪽 끝 → 오른쪽 플랜지 중심. 반지름은 화면 폭 비율
AXIS0 = np.array([[0.22, 0.28], [0.92, 0.50]])
RADIUS0 = 0.12

def done(task, result):
    d = os.path.join(OUT, task); os.makedirs(d, exist_ok=True)
    json.dump(result, open(os.path.join(d, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    open(os.path.join(d, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))

def load_pose(kind="lightweight"):
    path = EXP08 + "red_crop_pose.npz" if kind == "lightweight" else DATA + f"exp09/red_crop_pose_{kind}.npz"
    return dict(np.load(path, allow_pickle=True))

def load_labels(t):
    rows = list(csv.DictReader(open(LABEL_CSV, encoding="utf-8-sig")))
    lab = {round(float(r["시각(초)"]), 1): r["단계(정상/주의/감속/정지)"].strip() for r in rows}
    y = []
    for x in t:  # 0.2초 격자에서 가장 가까운 칸(±0.1초)의 판단
        k = round(round(x / 0.2) * 0.2, 1)
        y.append(STAGES.index(lab[k]) if lab.get(k) in STAGES else -1)
    return np.array(y)

def to_frame(P, i, pts):  # 기준 프레임 좌표(화면 비율) → i번째 프레임 픽셀
    w, h = P["wh"]; q = np.array(pts, np.float64) * [w, h]
    return cv2.perspectiveTransform(q.reshape(-1, 1, 2), P["H"][i]).reshape(-1, 2)

def seg_dist(p, a, b):
    ab = b - a; u = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + u * ab)))

def torso_len(k, s):
    if min(s[5], s[6], s[11], s[12]) < KP_MIN: return None
    return float(np.linalg.norm((k[5] + k[6]) / 2 - (k[11] + k[12]) / 2)) or None

def features(P, axis=AXIS0, radius=RADIUS0):
    """프레임마다 가장 위험한 사람 기준 특징. 거리는 몸통 길이 단위(표면까지, 안쪽이면 음수)"""
    w = P["wh"][0]; F = []; last_torso = None
    for i, (kps, sc) in enumerate(zip(P["kps"], P["sc"])):
        a, b = to_frame(P, i, axis); r = radius * w * np.sqrt(abs(np.linalg.det(P["H"][i][:2, :2])))  # 화면 보정의 확대 비율 반영
        best = None
        for k, s in zip(kps, sc):
            tl = torso_len(k, s) or last_torso or 0.25 * P["wh"][1]
            if torso_len(k, s): last_torso = tl
            def d(j): return (seg_dist(k[j], a, b) - r) / tl if s[j] >= KP_MIN else np.nan
            dw = np.nanmin([d(9), d(10)]) if max(s[9], s[10]) >= KP_MIN else np.nan
            de = np.nanmin([d(7), d(8)]) if max(s[7], s[8]) >= KP_MIN else np.nan
            dn = d(0)
            bend = np.nan
            if min(s[5], s[6], s[11], s[12]) >= KP_MIN:
                v = (k[5] + k[6]) / 2 - (k[11] + k[12]) / 2; bend = float(np.degrees(np.arctan2(abs(v[0]), -v[1])))
            f = {"d_wrist": dw, "d_elbow": de, "d_nose": dn, "bend": bend, "wrist_conf": float(max(s[9], s[10])), "n_kp": int((s >= KP_MIN).sum())}
            key = np.nanmin([dw, de, 99])
            if best is None or key < best[0]: best = (key, f)
        F.append(best[1] if best else {"d_wrist": np.nan, "d_elbow": np.nan, "d_nose": np.nan, "bend": np.nan, "wrist_conf": 0.0, "n_kp": 0})
    return F

def add_temporal(F, t, lag_s=0.3, win_s=1.0):
    fps = 1 / np.median(np.diff(t)); lag, win = max(1, int(lag_s * fps)), max(1, int(win_s * fps))
    dw = np.array([f["d_wrist"] for f in F], float)
    ff = np.copy(dw)  # 결측은 직전 값으로 채움
    for i in range(1, len(ff)):
        if np.isnan(ff[i]): ff[i] = ff[i - 1]
    for i, f in enumerate(F):
        f["v_approach"] = float((ff[i - lag] - ff[i]) / (lag / fps)) if i >= lag and not np.isnan(ff[i - lag]) and not np.isnan(ff[i]) else 0.0
        seg = ff[max(0, i - win + 1):i + 1]; seg = seg[~np.isnan(seg)]
        f["d_min_1s"] = float(seg.min()) if len(seg) else np.nan
        f["d_mean_1s"] = float(seg.mean()) if len(seg) else np.nan
    return F

def macro_f1(y, p):
    m = y >= 0; y, p = y[m], p[m]; fs = []
    for c in range(4):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        fs.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs)), [round(x, 3) for x in fs]

def onsets(t, p, level=3, after=None):
    on = [t[i] for i in range(len(p)) if p[i] >= level and (i == 0 or p[i - 1] < level)]
    return [round(float(x), 2) for x in on if after is None or x >= after]

def smooth(p, k):
    if k <= 1: return p
    out = np.copy(p)
    for i in range(len(p)): out[i] = int(np.median(p[max(0, i - k + 1):i + 1]))
    return out
