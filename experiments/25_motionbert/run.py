# 실험 25: MotionBERT(NTU60 행동 인식 미세조정 인코더, 고정) 동작 표현 + 기하 정보 → 로지스틱 회귀. 설계는 README
import os, sys, json, glob, cv2, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
HERE = os.path.dirname(os.path.abspath(__file__)); X_ = os.path.join(HERE, "..")
src = open(os.path.join(X_, "24_temporal", "train_eval.py"), encoding="utf-8").read(); exec(src[:src.index("clips = {n: clip_data(n)")])
MB = DATA + "models/MotionBERT"; sys.path.insert(0, MB)
from lib.model.DSTformer import DSTformer
from lib.data.dataset_action import coco2h36m
from lib.utils.utils_data import crop_scale
OUT = DATA + "exp25/"; os.makedirs(OUT, exist_ok=True); NF = 27
enc = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, depth=5, num_heads=8, mlp_ratio=2, num_joints=17, maxlen=243)
sd = torch.load(MB + "/checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub/best_epoch.bin", map_location="cpu", weights_only=False)["model"]
enc.load_state_dict({k.replace("module.backbone.", ""): v for k, v in sd.items() if k.startswith("module.backbone.")}); enc.eval(); torch.set_num_threads(os.cpu_count())

def nearest_person(P, F):
    """프레임마다 위험한계에 가장 가까운 사람의 관절 (17, 3: x, y, 신뢰도). 없으면 0"""
    seq = np.zeros((len(P["t"]), 17, 3), np.float32)
    for i, (K, S) in enumerate(zip(P["kps"], P["sc"])):
        hl = F["hull_at"](i); hl = [h.astype(np.float32) for h in (hl if isinstance(hl, list) else [hl])]; best = None
        for k, s in zip(K, S):
            vis = [j for j in range(17) if s[j] >= KP_MIN]
            if len(vis) < 3: continue
            d = min(-cv2.pointPolygonTest(h, (float(k[j][0]), float(k[j][1])), True) for j in vis for h in hl)
            if best is None or d < best[0]: best = (d, k, s)
        if best: seq[i, :, :2] = np.asarray(best[1], np.float32); seq[i, :, 2] = np.asarray(best[2], np.float32)
    return seq

def embed(name, P, F):
    """0.2초 간격마다 최근 1초를 27프레임으로 뽑아 MotionBERT 표현 512. 캐시"""
    path = OUT + f"{name}.npz"
    if os.path.exists(path): z = np.load(path); return z["k"], z["e"]
    seq = nearest_person(P, F); t = F["t"]; ks = np.arange(round(t[0] / 0.2) * 0.2, t[-1] + 1e-6, 0.2); batch = []
    for k in ks:
        idx = np.searchsorted(t, np.linspace(k - 1.0, k, NF)); idx = np.clip(idx, 0, len(t) - 1)
        w = seq[idx].copy(); w[(t[idx] < t[0] - 1e-6) | (t[idx] > k + 1e-6)] = 0
        batch.append(crop_scale(coco2h36m(w[None])[0]))
    E = []
    with torch.no_grad():
        for b in range(0, len(batch), 32):
            r = enc.get_representation(torch.tensor(np.array(batch[b:b + 32]), dtype=torch.float32)); E.append(r.mean(dim=(1, 2)).numpy())
    E = np.concatenate(E); np.savez(path, k=np.round(ks, 1), e=E); print("embedded", name, len(ks), flush=True); return np.round(ks, 1), E

def load(name, P, F, y=None, extra=None):
    M, near = frame_feats(P, F); W = windows(M, F["fps"]); ks, E = embed(name, P, F); t = F["t"]
    gi = np.clip(np.searchsorted(ks, t + 1e-6) - 1, 0, len(ks) - 1)  # 프레임마다 직전 0.2초 판단 지점
    return dict(G=W, ME=E[gi], t=t, near=near, fps=F["fps"], y=y, **(extra or {}))

clips = {}
for n in sorted(x for x in os.listdir(LAB) if os.path.isdir(LAB + x)):
    P = dict(np.load(POSE + f"{n}_pose.npz", allow_pickle=True)); lab, tc, hs = labels(n); zone, objects, floor, fronts = geometry(n); F = J.features(P, zone, floor, objects)
    y = np.array([lab.get(round(round(v / 0.2) * 0.2, 1), -1) for v in F["t"]]); clips[n] = load(n, P, F, y, {"tc": tc})
objs = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values())
floor = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json"))); press = []
for p in sorted(glob.glob(DATA + "exp20_stab/*_pose.npz")):
    if not os.path.basename(p)[0].isdigit(): continue
    P = dict(np.load(p, allow_pickle=True)); J.T.L = floor["lines"]; F = J.T.features(P, objs); press.append(load("press_" + os.path.basename(p)[:-9], P, F))
print("loaded", len(clips), len(press), flush=True)

def feats(c, kind): return c["G"] if kind == "G" else np.concatenate([c["ME"], c["G"]], 1)
def train(names, kind):
    Xs = [feats(clips[n], kind)[clips[n]["y"] >= 0][::2] for n in names] + [feats(c, kind)[::3] for c in press]
    ys = [(clips[n]["y"][clips[n]["y"] >= 0][::2] == 2).astype(int) for n in names] + [np.zeros(len(c["G"][::3]), int) for c in press]
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0)).fit(np.concatenate(Xs), np.concatenate(ys))

def score(model, names, kind):
    out, tot, mins, nf = {}, {"raw": [0, 0, 0], "hold": [0, 0, 0]}, 0.0, 0
    for n in names:
        c = clips[n]; base = np.where(model.predict_proba(feats(c, kind))[:, 1] >= 0.5, 3, 0); win = (c["t"] >= c["tc"] - 0.5) & (c["t"] <= c["tc"])
        mins += (c["y"] == 0).sum() / c["fps"] / 60; nf += int((c["y"] == 0).sum()); out[n] = {}
        for k2, o in (("raw", base), ("hold", hold(base, c["near"], c["fps"]))):
            hit = bool(win.any() and (o[win] == 3).all()); fs = len(episodes(c["t"], (o == 3) & (c["y"] == 0))); ns = int(((o == 3) & (c["y"] == 0)).sum())
            out[n][k2] = {"hit": hit, "false_stops": fs, "normal_stop_frac": round(ns / max(1, (c["y"] == 0).sum()), 2)}
            tot[k2][0] += hit; tot[k2][1] += fs; tot[k2][2] += ns
    return {k: {"stop_rate": f"{v[0]}/{len(names)}", "false_per_min": round(v[1] / mins, 2), "normal_stop_time": f"{v[2] / max(1, nf):.0%}"} for k, v in tot.items()}, out

TUNE = [n for n in clips if n not in TEST]; TESTN = [n for n in clips if n in TEST]
groups = [["A09_pipe", "A10_pipe_view2"]] + [[n] for n in TUNE if n not in ("A09_pipe", "A10_pipe_view2")]
res = {}
for kind in ("G", "M+G"):
    cv = {}
    for g in groups: cv.update(score(train([n for n in TUNE if n not in g], kind), g, kind)[1])
    cvh = {k: f"{sum(o[k]['hit'] for o in cv.values())}/{len(cv)}" for k in ("raw", "hold")}
    summ, out = score(train(TUNE, kind), TESTN, kind)
    res[kind] = {"cv_hits": cvh, "test": summ, "test_clips": out}; print(kind, "교차 검증", cvh, "| 시험", json.dumps(summ, ensure_ascii=False), flush=True)
json.dump(res, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
