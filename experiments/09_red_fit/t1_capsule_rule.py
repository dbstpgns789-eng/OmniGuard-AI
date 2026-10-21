# 09-1: 원통 구역 + 규칙. 손목(또는 손목·팔꿈치)에서 원통 표면까지 거리(몸통 단위)를 세 임계값으로 나눠 단계를 낸다
# 원통 축 위치·반지름·임계값·평활 창을 무작위 탐색으로 사용자 판단에 맞춘다(과적합 허용, 사용자 동의)
# 정직한 비교를 위해 앞 절반(7~27초)으로 맞추고 뒤 절반(27~46.5초)으로 확인한 결과도 함께 낸다
import sys, numpy as np
from common import *

kind = sys.argv[1] if len(sys.argv) > 1 else "lightweight"
P = load_pose(kind); t = P["t"]; y = load_labels(t)
rng = np.random.default_rng(0)

def predict(F, th, use_elbow, k):
    d = np.array([np.nanmin([f["d_wrist"], f["d_elbow"]]) if use_elbow else f["d_wrist"] for f in F], float)
    p = np.zeros(len(d), int)
    p[d < th[2]] = 1; p[d < th[1]] = 2; p[d < th[0]] = 3
    p[np.isnan(d)] = 0
    return smooth(p, k)

feat_cache = {}
def feats(dx):
    key = tuple(np.round(dx, 3))
    if key not in feat_cache:
        axis = AXIS0 + dx[:4].reshape(2, 2); feat_cache[key] = features(P, axis, RADIUS0 + dx[4])
    return feat_cache[key]

train, test = (t < 27) & (y >= 0), (t >= 27) & (y >= 0)
best_all, best_tr = None, None
base = np.zeros(5)
cands = [base] + [np.r_[rng.uniform(-0.06, 0.06, 4), rng.uniform(-0.05, 0.05)] for _ in range(60)]
for ci, dx in enumerate(cands):
    F = feats(dx)
    for _ in range(150):
        a = np.sort(rng.uniform(-0.5, 3.0, 3)); th = (a[0], a[1], a[2])
        ue = bool(rng.integers(2)); k = int(rng.choice([1, 3, 5, 9, 15]))
        p = predict(F, th, ue, k)
        fa, _ = macro_f1(y, p); ftr, _ = macro_f1(np.where(train, y, -1), p)
        if best_all is None or fa > best_all[0]: best_all = (fa, dx, th, ue, k)
        if best_tr is None or ftr > best_tr[0]: best_tr = (ftr, dx, th, ue, k)
    if ci % 10 == 0: print(ci, round(best_all[0], 3), round(best_tr[0], 3), flush=True)

def report(b):
    fa, dx, th, ue, k = b; p = predict(feats(dx), th, ue, k)
    return {"macro_f1_all": macro_f1(y, p), "macro_f1_test_27s+": macro_f1(np.where(test, y, -1), p),
            "accuracy_all": round(float((p[y >= 0] == y[y >= 0]).mean()), 3),
            "stop_onsets": onsets(t, p, 3), "human_stop_onsets": onsets(t, np.where(y < 0, 0, y), 3),
            "params": {"axis": (AXIS0 + dx[:4].reshape(2, 2)).round(3).tolist(), "radius": round(RADIUS0 + dx[4], 3),
                       "th_stop_slow_warn": [round(x, 3) for x in th], "use_elbow": ue, "smooth_frames": k}}

np.save(OUT + f"/t1_pred_{kind}.npy", predict(feats(best_all[1]), best_all[2], best_all[3], best_all[4]))
res = {"pose": kind, "fit_all": report(best_all), "fit_on_first_half": report(best_tr),
       "baseline_untuned": report((0, np.zeros(5), (0.0, 0.5, 1.5), False, 3))}
print(res); done(f"t1_capsule_rule_{kind}", res)
