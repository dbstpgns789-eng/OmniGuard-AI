# 실험 13: 경계값을 어느 영상에도 맞추지 않고 원칙대로 정해, 지금까지의 모든 영상에 똑같이 돌린다
# 원칙: 손이 위험한계 경계에 닿거나 넘으면 정지(0), 몸통 길이 0.5배 이내로 다가오면 감속, 1배 이내로 다가오면 주의. 몸(어깨·팔꿈치)도 경계에 닿으면 정지
# 걷기 인식·정지 유지·측정 품질 같은 판단 이유 규칙과 그 값은 09 그대로. 비교: 빨간 상의 영상에 맞춘 경계값(09 t13)
# 영상: 09 빨간 상의, 10 넓은 화면(iSafety 4클립), 11 프레스 공장 평소 작업(모두 정상), 12 사고 5건
import os, sys, csv, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
for d in ("09_red_fit", "10_isafety_transfer"): sys.path.insert(0, os.path.join(X, d))
import transfer as T
from common import STAGES, DATA, macro_f1, onsets

P_RED = list(T.p_red); P_RULE = list(T.p_red)
for k, v in (("near_warn", 1.0), ("near_slow", 0.5), ("hand_inside", 0.0), ("body_inside", 0.0)): P_RULE[T.NAMES.index(k)] = v
VARIANTS = {"red": P_RED, "rule": P_RULE}

def score(F, y):  # y=-1은 채점 제외
    r = {}
    for name, p in VARIANTS.items():
        out, _ = T.judge(F, p); t = F["t"]; m = y >= 0
        hs = next((t[i] for i in range(len(y)) if y[i] == 3), None); js = next((t[i] for i in range(len(out)) if out[i] == 3 and y[i] >= 0), None)
        r[name] = {"macro_f1": round(macro_f1(y[m], out[m])[0], 3) if (y[m] > 0).any() else None,
                   "human_first_stop": None if hs is None else round(float(hs), 2), "judge_first_stop": None if js is None else round(float(js), 2),
                   "stop_recall": round(float((out[y == 3] == 3).mean()), 3) if (y == 3).any() else None,
                   "false_stop_frames": int(((out == 3) & m & (y < 3)).sum()), "scored_frames": int(m.sum()),
                   "stop_onsets": len(onsets(t, np.where(m, out, 0), 3)), "sec": round(float(m.sum() / F["fps"]), 1)}
    return r

res = {}
# 09 빨간 상의 (t12와 같은 반영본 정답)
T.Z = json.load(open(os.path.join(X, "09_red_fit", "zone_user.json"))); T.L = json.load(open(os.path.join(X, "09_red_fit", "floor_user.json")))["lines"]
P = dict(np.load(DATA + "exp09/red_crop_pose_performance.npz", allow_pickle=True))
lab = {round(float(r["시각(초)"]), 1): r["내 판단(기존)"].strip() for r in csv.DictReader(open(DATA + "exp09/frames_humanlike/판단표_판단기비교.csv", encoding="utf-8-sig"))}
lab.update({8.6: "정지", 22.6: "주의", 35.2: "정지"}); lab[7.0] = ""
g = lambda x: lab.get(round(round(x / 0.2) * 0.2, 1), "")
res["09_red"] = score(T.features(P), np.array([STAGES.index(g(x)) if g(x) in STAGES else -1 for x in P["t"]]))
# 10 넓은 화면 (transfer.py가 불러올 때 이미 특징을 계산해 둠)
for n_, (P, F, y) in T.clips.items(): res[f"10_isafety_{n_}"] = score(F, y)
# 11 프레스 공장 평소 작업: 사람 판단은 전부 정상
OBJ = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values())
T.L = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))["lines"]
for path in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    P = dict(np.load(path, allow_pickle=True)); F = T.features(P, OBJ); res["11_" + os.path.basename(path)[:-9]] = score(F, np.zeros(len(F["t"]), int))
# 12 사고 5건 (12 evaluate.py와 같은 정답 규칙)
sys.path.insert(0, os.path.join(X, "12_accident_set"))
C = json.load(open(os.path.join(X, "12_accident_set", "clips.json"), encoding="utf-8"))
def labels12(k, t):
    rows = list(csv.DictReader(open(DATA + f"exp08/frames_for_review/{k}/판단표.csv", encoding="utf-8-sig")))
    seq = [(round(float(r["시각(초)"]), 1), r["단계(정상/주의/감속/정지)"].strip()) for r in rows if r["시각(초)"].strip()]
    last = max((i for i, (_, v) in enumerate(seq) if v in STAGES), default=-1); filled, cur = {}, ""
    for i, (s, v) in enumerate(seq):
        if v in STAGES: cur = v
        filled[s] = cur if i <= last else ""
    f = lambda x: filled.get(round(round(x / 0.2) * 0.2, 1), "")
    return np.array([STAGES.index(f(x)) if f(x) in STAGES else -1 for x in t])
for k in C:
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True))
    T.Z = json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    T.L = json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json")))["lines"]
    F = T.features(P); res["12_" + k] = score(F, labels12(k, F["t"]))

json.dump(res, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, r in res.items():
    print(k, " | ".join(f"{n}: F1 {v['macro_f1']} 사람정지 {v['human_first_stop']} 판단기정지 {v['judge_first_stop']} 정지유지 {v['stop_recall']} 헛정지프레임 {v['false_stop_frames']} 정지횟수 {v['stop_onsets']}" for n, v in r.items()))
