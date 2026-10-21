# 영상마다 보정 결과 확인: 초점 거리, 카메라 높이, 똑바로 선 프레임 수, 바닥 앞선 길이(m), 사람 발과 앞선 거리 분포
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(X, "09_red_fit")); from common import DATA
from calib import Floor, seg_dist
def sets():
    yield "red", dict(np.load(DATA + "exp09/red_crop_pose_performance.npz", allow_pickle=True)), json.load(open(X + "/09_red_fit/floor_user.json"))["lines"]
    for n in (2, 4, 7, 8): yield f"wide_{n}", dict(np.load(DATA + f"exp10/trim_{n:02d}_pose.npz", allow_pickle=True)), json.load(open(X + "/10_isafety_transfer/floor_user.json"))["lines"]
    for p in sorted(glob.glob(DATA + "exp11/*_pose.npz"))[:3]: yield "press_" + os.path.basename(p)[:-9], dict(np.load(p, allow_pickle=True)), json.load(open(X + "/11_press_normal/floor_user.json"))["lines"]
    for k in ("slitter", "paper_rewinder", "night_paper", "conveyor_parcel", "bag_machine"): yield k, dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True)), json.load(open(X + f"/12_accident_set/floor_{k}.json"))["lines"]
FR = lambda name: json.load(open(os.path.join(HERE, "front_" + ("wide" if name.startswith("wide") else "press" if name.startswith("press") else name) + ".json")))["lines"]
for name, P, L in sets():
    if "H" not in P: P["H"] = np.repeat(np.eye(3)[None], len(P["t"]), 0)
    fl = Floor(L, P).fit_scale(); segs = [fl.seg(l) for l in FR(name)]
    ds = [min(seg_dist(F, sg) for sg in segs) for i, (K, S) in enumerate(zip(P["kps"], P["sc"])) for k, s in zip(K, S) for F in [fl.foot(k, s, i)] if F is not None]
    hv = np.array(fl.hs_raw) if fl.hs_raw else np.array([np.nan])
    q = (np.percentile(hv, 75) - np.percentile(hv, 25)) / np.median(hv)
    fov = np.degrees(2 * np.arctan(P["wh"][0] / 2 / fl.f)) if fl.f == fl.f else np.nan
    print(f"{name:22s} FOV={fov:5.1f}deg 키추정 흔들림(IQR/중앙값)={q:.2f}", end=" ")
    print(f"f={fl.f:7.0f} Hc={fl.Hc:5.2f}m upright={fl.n_upright:4d} front_len={[round(float(np.linalg.norm(sg[1]-sg[0])),2) for sg in segs]} foot_d p10/50/90={np.round(np.percentile(ds,[10,50,90]),2) if ds else None}")
