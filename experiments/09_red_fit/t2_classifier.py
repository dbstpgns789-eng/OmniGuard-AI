# 09-2: 규칙 대신 작은 분류기가 포즈 특징(원통까지 거리, 접근 속도, 숙임 등)으로 단계를 맞히게 한다
# 정직한 점수: 시간 순서대로 5구간으로 나눠 한 구간씩 빼고 학습·예측(구간 교차검증). 데모용: 전체로 학습(과적합 허용)
import sys, numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from common import *

kind = sys.argv[1] if len(sys.argv) > 1 else "lightweight"
P = load_pose(kind); t = P["t"]; y = load_labels(t)
F = add_temporal(features(P), t)
KEYS = ["d_wrist", "d_elbow", "d_nose", "bend", "wrist_conf", "n_kp", "v_approach", "d_min_1s", "d_mean_1s"]
X = np.array([[f[k] for k in KEYS] for f in F], float); X = np.nan_to_num(X, nan=9.0)
m = y >= 0
models = {"random_forest": lambda: RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", random_state=0),
          "grad_boost": lambda: HistGradientBoostingClassifier(max_depth=4, class_weight="balanced", random_state=0)}
res = {"pose": kind, "features": KEYS}
for name, mk in models.items():
    cv = np.full(len(y), -1); blocks = np.array_split(np.arange(len(y)), 5)
    for b in blocks:
        tr = np.ones(len(y), bool); tr[b] = False; tr &= m
        clf = mk().fit(X[tr], y[tr]); cv[b] = clf.predict(X[b])
    full = mk().fit(X[m], y[m]); pf = full.predict(X)
    for k in (1, 5, 9):
        res[f"{name}_smooth{k}"] = {"cv_macro_f1": macro_f1(y, smooth(cv, k)), "fit_all_macro_f1": macro_f1(y, smooth(pf, k)),
                                    "fit_all_stop_onsets": onsets(t, smooth(pf, k), 3), "cv_stop_onsets": onsets(t, smooth(cv, k), 3)}
    if name == "random_forest":
        res["rf_importance"] = dict(zip(KEYS, np.round(full.feature_importances_, 3).tolist()))
        np.save(OUT + f"/t2_pred_{kind}.npy", smooth(pf, 5)); np.save(OUT + f"/t2_cvpred_{kind}.npy", smooth(cv, 5))
res["human_stop_onsets"] = onsets(t, np.where(y < 0, 0, y), 3)
print(res); done(f"t2_classifier_{kind}", res)
