# (1차 탐지 = 최종 판단기, 2차 확인 = 채우 모델: 둘 다 stop일 때만 정지도 함께 잰다)
# 채우가 보낸 두 모델(jev, vlm)의 프레임 판단(safe / stop)을 세훈 판단표로 채점하고, 같은 프레임에서 최종 판단기(포즈 품질 필터 + J13 + 정지 유지)와 비교
import os, sys, json, zipfile, csv, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); X_ = os.path.join(HERE, "..")
src = open(os.path.join(X_, "24_temporal", "train_eval.py"), encoding="utf-8").read(); exec(src[:src.index("clips = {n: clip_data(n)")])
_s = open(os.path.join(X_, "28_pose_filter", "run.py"), encoding="utf-8").read(); exec(_s[_s.index("BONES ="):_s.index("def judge_pair")])
sys.path.insert(0, os.path.join(X, "09_red_fit")); from common import torso_len

z = zipfile.ZipFile("C:/Users/User/Downloads/sorted.zip"); pred = {}
for n in z.namelist():
    p = n.split("/")
    if len(p) == 5 and p[4].endswith(".jpg"): pred.setdefault((p[1], p[2]), {})[round(float(p[4][:-5]), 1)] = int(p[3] == "stop")

def red_inputs():
    R9 = os.path.join(X, "09_red_fit"); P = dict(np.load(DATA + "exp09/red_crop_pose_performance.npz", allow_pickle=True))
    raw = {round(float(r["시각(초)"]), 1): r["내 판단(기존)"].strip() for r in csv.DictReader(open(DATA + "exp09/frames_humanlike/판단표_판단기비교.csv", encoding="utf-8-sig"))}
    raw.update({8.6: "정지", 22.6: "주의", 35.2: "정지"}); lab = {k: {"정상": 0, "주의": 1, "감속": 1, "정지": 2}[v] for k, v in raw.items() if v in ("정상", "주의", "감속", "정지")}
    return P, lab, None, json.load(open(os.path.join(R9, "zone_user.json"))), None, json.load(open(os.path.join(R9, "floor_user.json")))

res = {}
for vid in ("A04_carpet", "B38_mixer", "D09", "red_shirt"):
    if vid == "red_shirt": P, lab, tc, zone, objects, floor = red_inputs()
    else: P = dict(np.load(POSE + f"{vid}_pose.npz", allow_pickle=True)); lab, tc, hs = labels(vid); zone, objects, floor, fronts = geometry(vid)
    P = filt(P)[0]; F = J.features(P, zone, floor, objects); t = F["t"]; M, near = frame_feats(P, F); final = hold(J.run("J13", P, F), near, F["fps"]) == 3
    fin = {round(round(v / 0.2) * 0.2, 1): bool(final[i]) for i, v in enumerate(t)}
    F_ = {k: fin.get(k, False) for k in pred[("jev", vid)]}
    for model in ("jev", "vlm", "최종 판단기", "최종 판단기 AND vlm", "최종 판단기 AND jev"):
        pr = F_ if model == "최종 판단기" else {k: F_[k] and pred[(model.split()[-1], vid)].get(k, 0) for k in F_} if "AND" in model else pred[(model, vid)]
        ks = [k for k in pr if k in lab]; y = np.array([lab[k] == 2 for k in ks]); o = np.array([bool(pr[k]) for k in ks]); nrm = np.array([lab[k] == 0 for k in ks])
        tp, fp, fn = int((o & y).sum()), int((o & ~y).sum()), int((~o & y).sum())
        r = {"frames": len(ks), "acc": round(float((o == y).mean()), 2), "recall_stop": round(tp / max(1, tp + fn), 2), "precision_stop": round(tp / max(1, tp + fp), 2),
             "normal_stop_frac": round(float(o[nrm].mean()), 2) if nrm.any() else None}
        if tc is not None:
            w = [k for k in ks if tc - 0.5 <= k <= tc]; r["stop_before_contact"] = bool(w) and all(pr[k] for k in w)
        res[f"{vid}/{model}"] = r; print(vid, model, r, flush=True)
json.dump(res, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
