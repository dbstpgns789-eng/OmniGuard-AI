# 실험 26: A 이상 탐지(정상 동작만), B 약한 라벨로 사고를 늘린 MotionBERT 분류기. 설계는 README
import os, sys, json, glob, cv2, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.pipeline import make_pipeline
HERE = os.path.dirname(os.path.abspath(__file__)); X_ = os.path.join(HERE, "..")
src = open(os.path.join(X_, "24_temporal", "train_eval.py"), encoding="utf-8").read(); exec(src[:src.index("clips = {n: clip_data(n)")])
MB = DATA + "models/MotionBERT"; sys.path.insert(0, MB)
from lib.model.DSTformer import DSTformer
from lib.data.dataset_action import coco2h36m
from lib.utils.utils_data import crop_scale
P26 = DATA + "exp26/pose/"; EMB = DATA + "exp26/emb/"; os.makedirs(EMB, exist_ok=True); NF = 27
WEAK = {"A01": 34.9, "A03": 17.6, "A07": 14.3, "A08": 9.9, "B03": 15.5, "B04": 7.6, "B05": 71.3, "B06": 7.4, "B08": 5.0, "B09": 138.8,
        "B15": 15.4, "B21": 19.6, "B22": 18.4, "B31": 37.5, "B33": 27.0, "B35": 33.3, "B37": 11.6, "B39": 17.8}
enc = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, depth=5, num_heads=8, mlp_ratio=2, num_joints=17, maxlen=243)
sd = torch.load(MB + "/checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub/best_epoch.bin", map_location="cpu", weights_only=False)["model"]
enc.load_state_dict({k.replace("module.backbone.", ""): v for k, v in sd.items() if k.startswith("module.backbone.")}); enc.eval(); torch.set_num_threads(os.cpu_count())

def pick(P, i, hull=None):
    """i번째 프레임에서 고를 사람 (17, 3). 위험한계가 있으면 가장 가까운 사람, 없으면 가장 큰 사람"""
    best = None
    for k, s in zip(P["kps"][i], P["sc"][i]):
        k, s = np.asarray(k, np.float32), np.asarray(s, np.float32); vis = s >= KP_MIN
        if vis.sum() < 3: continue
        if hull is not None: key = min(-cv2.pointPolygonTest(h, (float(k[j][0]), float(k[j][1])), True) for j in np.flatnonzero(vis) for h in hull)
        else: key = -np.ptp(k[vis, 0]) * np.ptp(k[vis, 1])
        if best is None or key < best[0]: best = (key, k, s)
    out = np.zeros((17, 3), np.float32)
    if best: out[:, :2] = best[1]; out[:, 2] = best[2]
    return out

def embed(name, P, hull_at=None):
    """초당 10장으로 맞춘 뒤 0.2초 간격마다 최근 1초 → MotionBERT 512. 캐시"""
    path = EMB + f"{name}.npz"
    if os.path.exists(path): z = np.load(path); return z["k"], z["e"]
    t = P["t"]; keep = np.unique(np.searchsorted(t, np.arange(t[0], t[-1] + 1e-6, 0.1)).clip(0, len(t) - 1))  # 초당 10장
    seq = np.array([pick(P, i, None if hull_at is None else [h.astype(np.float32) for h in (lambda x: x if isinstance(x, list) else [x])(hull_at(i))]) for i in keep]); ts = t[keep]
    ks = np.round(np.arange(round(ts[0] / 0.2) * 0.2, ts[-1] + 1e-6, 0.2), 1); batch = []
    for k in ks:
        idx = np.flatnonzero((ts > k - 1.0 - 1e-6) & (ts <= k + 1e-6))
        w = np.zeros((NF, 17, 3), np.float32) if len(idx) == 0 else seq[idx[np.linspace(0, len(idx) - 1, NF).round().astype(int)]]
        batch.append(crop_scale(coco2h36m(w[None])[0]))
    E = []
    with torch.no_grad():
        for b in range(0, len(batch), 32): E.append(enc.get_representation(torch.tensor(np.array(batch[b:b + 32]), dtype=torch.float32)).mean(dim=(1, 2)).numpy())
    E = np.concatenate(E); np.savez(path, k=ks, e=E); print("embedded", name, len(ks), flush=True); return ks, E

# 시험·검증 영상 (위험한계 있음)
clips = {}
for n in sorted(x for x in os.listdir(LAB) if os.path.isdir(LAB + x)):
    P = dict(np.load(POSE + f"{n}_pose.npz", allow_pickle=True)); lab, tc, hs = labels(n); zone, objects, floor, fronts = geometry(n); F = J.features(P, zone, floor, objects)
    t = F["t"]; y = np.array([lab.get(round(round(v / 0.2) * 0.2, 1), -1) for v in t]); ks, E = embed(n, P, F["hull_at"])
    lost = np.array([not any(int((np.asarray(s) >= KP_MIN).sum()) >= 3 for s in sc) for sc in P["sc"]]); dall = np.nan_to_num(J.db_rule4(P, F, None), nan=9)
    gi = np.clip(np.searchsorted(ks, t + 1e-6) - 1, 0, len(ks) - 1); yk = np.array([lab.get(k, -1) for k in ks])
    clips[n] = dict(t=t, y=y, tc=tc, fps=F["fps"], gi=gi, ks=ks, E=E, yk=yk, near=lost | (dall < NEAR))
# 학습용 정상·약한 라벨 데이터
objs = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values()); floor = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))
press = []
for p in sorted(glob.glob(DATA + "exp20_stab/*_pose.npz")):
    if not os.path.basename(p)[0].isdigit(): continue
    P = dict(np.load(p, allow_pickle=True)); J.T.L = floor["lines"]; F = J.T.features(P, objs); press.append(embed("press_" + os.path.basename(p)[:-9], P, F["hull_at"])[1])
su = [embed(os.path.basename(p)[:-4], dict(np.load(p, allow_pickle=True))) [1] for p in sorted(glob.glob(P26 + "su_*.npz"))]
weak = {}
for k, tc in WEAK.items():
    ks, E = embed(f"weak_{k}", dict(np.load(P26 + f"weak_{k}.npz", allow_pickle=True))); weak[k] = (ks, E, tc)
print("data: clips", len(clips), "press", len(press), "su", len(su), "weak", len(weak), flush=True)
TUNE = [n for n in clips if n not in TEST]; TESTN = [n for n in clips if n in TEST]

def evaluate(cand_fn, names):
    """cand_fn(clip) → 0.2초 지점마다 정지 후보(bool). 위험한계 근처일 때만 정지 + 정지 유지"""
    tot = [0, 0, 0]; mins = 0.0; nf = 0; per = {}
    for n in names:
        c = clips[n]; cand = cand_fn(c)[c["gi"]]; base = np.where(cand & c["near"], 3, 0); o = hold(base, c["near"], c["fps"])
        win = (c["t"] >= c["tc"] - 0.5) & (c["t"] <= c["tc"]); hit = bool(win.any() and (o[win] == 3).all())
        fs = len(episodes(c["t"], (o == 3) & (c["y"] == 0))); ns = int(((o == 3) & (c["y"] == 0)).sum())
        tot[0] += hit; tot[1] += fs; tot[2] += ns; mins += (c["y"] == 0).sum() / c["fps"] / 60; nf += int((c["y"] == 0).sum())
        per[n] = {"hit": hit, "false_stops": fs, "normal_stop_frac": round(ns / max(1, (c["y"] == 0).sum()), 2)}
    return {"stop_rate": f"{tot[0]}/{len(names)}", "false_per_min": round(tot[1] / mins, 2), "normal_stop_time": f"{tot[2] / max(1, nf):.0%}"}, per

# A. 정상 동작만으로 이상 탐지
normal = np.concatenate(press + su + [E[ks < tc - 5] for ks, E, tc in weak.values()] + [clips[n]["E"][clips[n]["yk"] == 0] for n in TUNE])
pre = make_pipeline(StandardScaler(), PCA(32, random_state=0)).fit(normal); Z = pre.transform(normal); cov = LedoitWolf().fit(Z)
thr = float(np.percentile(cov.mahalanobis(Z), 99)); print("A 정상 표본", len(normal), "문턱", round(thr, 1), flush=True)
candA = lambda c: cov.mahalanobis(pre.transform(c["E"])) >= thr
resA = {"tune": evaluate(candA, TUNE), "test": evaluate(candA, TESTN)}; print("A 검증", resA["tune"][0], "| 시험", resA["test"][0], flush=True)

# B. 약한 라벨로 사고를 늘린 분류기
Xs, ys = [], []
for ks, E, tc in weak.values():
    m1 = (ks >= tc - 2.0) & (ks <= tc); m0 = ks < tc - 5; Xs += [E[m1], E[m0]]; ys += [np.ones(m1.sum()), np.zeros(m0.sum())]
for n in TUNE:
    c = clips[n]; m = c["yk"] >= 0; Xs.append(c["E"][m]); ys.append((c["yk"][m] == 2).astype(float))
for E in press + su: Xs.append(E); ys.append(np.zeros(len(E)))
clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)).fit(np.concatenate(Xs), np.concatenate(ys))
print("B 학습 표본", sum(len(x) for x in Xs), "정지", int(sum(y.sum() for y in ys)), flush=True)
candB = lambda c: clf.predict_proba(c["E"])[:, 1] >= 0.5
resB = {"test": evaluate(candB, TESTN)}; print("B 시험", resB["test"][0], flush=True)
json.dump({"A": resA, "B": resB, "A_threshold": thr}, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
