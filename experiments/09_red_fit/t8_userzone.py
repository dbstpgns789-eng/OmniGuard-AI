# 09-8: 사용자가 그린 양 끝 원판 두 개(zone_user.json)를 감싸서(볼록 껍질) 원통 위험한계를 만들고 고정한다
# 원통을 화면에 비추면 양 끝 타원을 감싸는 볼록 영역이 되므로, 두 원판만 그리면 원통 전체가 된다
# 관절에서 원통 표면까지 화면 거리(몸통 단위, 안쪽이면 음수)로 판단. 위치는 고정하고 띠 두께(임계값)만 사람 판단에 맞춘다
# 주의 띠는 몸통 1.5배를 넘지 않게 제한한다. 규칙과 분류기를 가벼운·큰 포즈 모델 두 가지로 돌리고 데모 영상을 만든다
import os, sys, json, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestClassifier
from common import *

Z = json.load(open(os.path.join(HERE, "zone_user.json")))
WARN_MAX = 1.5

def hull_at(P, i):
    pts = np.vstack([to_frame(P, i, o) for o in Z["outlines"]]).astype(np.float32)
    return cv2.convexHull(pts).reshape(-1, 2)

def feats(P):
    F, last_tl = [], None
    for i, (kps, sc) in enumerate(zip(P["kps"], P["sc"])):
        hull = hull_at(P, i); best = None
        for k, s in zip(kps, sc):
            tl = torso_len(k, s) or last_tl or 0.25 * P["wh"][1]
            if torso_len(k, s): last_tl = tl
            def d(j): return -cv2.pointPolygonTest(hull, (float(k[j][0]), float(k[j][1])), True) / tl if s[j] >= KP_MIN else np.nan
            dw = np.nanmin([d(9), d(10)]) if max(s[9], s[10]) >= KP_MIN else np.nan
            de = np.nanmin([d(7), d(8)]) if max(s[7], s[8]) >= KP_MIN else np.nan
            bend = np.nan
            if min(s[5], s[6], s[11], s[12]) >= KP_MIN:
                v = (k[5] + k[6]) / 2 - (k[11] + k[12]) / 2; bend = float(np.degrees(np.arctan2(abs(v[0]), -v[1])))
            f = {"d_wrist": dw, "d_elbow": de, "d_nose": d(0), "bend": bend, "wrist_conf": float(max(s[9], s[10])), "n_kp": int((s >= KP_MIN).sum())}
            key = np.nanmin([dw, de, 99])
            if best is None or key < best[0]: best = (key, f)
        F.append(best[1] if best else {"d_wrist": np.nan, "d_elbow": np.nan, "d_nose": np.nan, "bend": np.nan, "wrist_conf": 0.0, "n_kp": 0})
    return F

def rule(F, th, ue, k):
    d = np.array([np.nanmin([f["d_wrist"], f["d_elbow"]]) if ue else f["d_wrist"] for f in F], float)
    p = np.zeros(len(d), int); p[d < th[2]] = 1; p[d < th[1]] = 2; p[d < th[0]] = 3; p[np.isnan(d)] = 0
    return smooth(p, k)

KEYS = ["d_wrist", "d_elbow", "d_nose", "bend", "wrist_conf", "n_kp", "v_approach", "d_min_1s", "d_mean_1s"]
res = {"zone": Z}
os.makedirs(DATA + "exp09", exist_ok=True)
for kind in ("lightweight", "performance"):
    P = load_pose(kind); t = P["t"]; y = load_labels(t); F = add_temporal(feats(P), t)
    rng = np.random.default_rng(0); best = None
    for _ in range(20000):  # 띠 두께만 탐색
        a = np.sort(rng.uniform(-0.6, WARN_MAX, 3)); ue = bool(rng.integers(2)); k = int(rng.choice([1, 3, 5, 9, 15]))
        f1, _ = macro_f1(y, rule(F, a, ue, k))
        if best is None or f1 > best[0]: best = (f1, a, ue, k)
    pr = rule(F, best[1], best[2], best[3])
    X = np.nan_to_num(np.array([[f[kk] for kk in KEYS] for f in F], float), nan=9.0); m = y >= 0
    mk = lambda: RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", random_state=0)
    cv = np.full(len(y), -1)
    for b in np.array_split(np.arange(len(y)), 5):
        tr = np.ones(len(y), bool); tr[b] = False; tr &= m; cv[b] = mk().fit(X[tr], y[tr]).predict(X[b])
    clf = mk().fit(X[m], y[m]); pc = smooth(clf.predict(X), 5); cv = smooth(cv, 5)
    res[kind] = {"rule": {"macro_f1": macro_f1(y, pr), "stop_onsets": onsets(t, pr, 3), "th_stop_slow_warn": best[1].round(3).tolist(), "use_elbow": best[2], "smooth": best[3]},
                 "classifier_fit_all": {"macro_f1": macro_f1(y, pc), "stop_onsets": onsets(t, pc, 3)},
                 "classifier_cv": {"macro_f1": macro_f1(y, cv), "stop_onsets": onsets(t, cv, 3)},
                 "importance": dict(zip(KEYS, clf.feature_importances_.round(3).tolist())), "human_stop_onsets": onsets(t, np.where(y < 0, 0, y), 3)}
    print(kind, json.dumps(res[kind], ensure_ascii=False, default=float), flush=True)
    np.save(OUT + f"/t8_rule_{kind}.npy", pr); np.save(OUT + f"/t8_clf_{kind}.npy", pc)

    # 데모 영상: 원통(초록) + 바깥 띠(주의·감속·정지), 상단 분류기 판정, 하단 사람·규칙·분류기 막대
    COL = {0: (90, 90, 90), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 255), -1: (40, 40, 40)}
    FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 26); SM = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 16)
    tl_med = float(np.nanmedian([torso_len(k, s) or np.nan for ks, ss in zip(P["kps"], P["sc"]) for k, s in zip(ks, ss)]))
    th = best[1]; cap = cv2.VideoCapture(DATA + "raw_cctv/rec_002103.mp4"); fps = cap.get(5); cap.set(1, int(round(t[0] * fps)))
    w, h = P["wh"]; wr = None; n = len(t)
    for i in range(n):
        ok, img = cap.read()
        if not ok: break
        hull = hull_at(P, i).astype(np.int32); inside = np.zeros((h, w), np.uint8); cv2.fillPoly(inside, [hull], 255)
        dist = cv2.distanceTransform(255 - inside, cv2.DIST_L2, 3) / tl_med  # 원통 바깥 거리(몸통 단위)
        for lvl, lim in ((1, th[2]), (2, th[1]), (3, th[0])):
            band = ((dist < max(lim, 0)) & (inside == 0)).astype(np.uint8)
            if band.any():
                ov = img.copy(); ov[band > 0] = COL[lvl]; img = cv2.addWeighted(ov, 0.18, img, 0.82, 0)
                cs, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img, cs, -1, COL[lvl], 2)
        ov = img.copy(); cv2.fillPoly(ov, [hull], (0, 200, 0)); img = cv2.addWeighted(ov, 0.22, img, 0.78, 0); cv2.polylines(img, [hull], True, (0, 200, 0), 2)
        for k, s in zip(P["kps"][i], P["sc"][i]):
            for j in range(17):
                if s[j] >= KP_MIN: cv2.circle(img, (int(k[j][0]), int(k[j][1])), 6 if j in (9, 10) else 3, (0, 255, 0) if j in (9, 10) else (255, 255, 0), -1)
        lv = int(pc[i]); cv2.rectangle(img, (0, 0), (w, 48), COL[lv], -1)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text((10, 8), f"모델: {STAGES[lv]}   사람: {STAGES[y[i]] if y[i] >= 0 else '-'}   t={t[i]:4.1f}s", font=FONT, fill=(255, 255, 255))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR); bars = []
        for lab, arr in (("사람", y), ("규칙", pr), ("분류기", pc)):
            bar = np.full((24, w, 3), 25, np.uint8)
            for j in range(i + 1): x = 70 + int(j * (w - 70) / n); cv2.line(bar, (x, 2), (x, 21), COL[int(arr[j])], max(1, (w - 70) // n + 1))
            p2 = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)); ImageDraw.Draw(p2).text((4, 2), lab, font=SM, fill=(255, 255, 255))
            bars.append(cv2.cvtColor(np.array(p2), cv2.COLOR_RGB2BGR))
        frame = np.vstack([img] + bars)
        if wr is None: wr = cv2.VideoWriter(DATA + f"exp09/demo_userzone_{kind}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        wr.write(frame)
        if i in (int(n * 0.07), int(n * 0.6), int(n * 0.75)): cv2.imwrite(DATA + f"exp09/demo_userzone_{kind}_{i}.jpg", frame)
    wr.release()
done("t8_userzone", res)
