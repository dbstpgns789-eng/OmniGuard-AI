# 실험 10-2: 회전축 넘기 규칙. 팔꿈치·손이 회전축(두 원판 중심을 잇는 선)을 넘어 반대편에 있어야 "손이 들어감(정지)"
# 넘지 않았으면 원판 표면 근처 작업으로 보고 감속까지만. 두 영상(빨간 상의 확대 / 넓은 화면)을 함께 채점해 과적합인지 확인
#   A: 09 판단기 그대로   B: A + 회전축 규칙(경계값은 A와 같고 넘기 기준만 0)   C: 두 영상에 함께 다시 맞춤
import os, sys, csv, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); R09 = os.path.join(HERE, "..", "09_red_fit"); sys.path.insert(0, R09)
from common import STAGES, KP_MIN, torso_len, to_frame, macro_f1, onsets
import transfer as T  # features(), judge(), 넓은 화면 데이터(clips), 09 경계값(p_red)를 재사용 (불러올 때 transfer가 한 번 실행됨)

DATA = "C:/Users/User/Desktop/OmniGuard-data/"

def crossing(P, Z):
    """프레임마다: 가장 깊이 들어간 사람의 팔꿈치·손이 회전축 너머로 얼마나 갔는가(몸통 단위, +면 반대편)"""
    out = np.full(len(P["t"]), np.nan)
    cents = [np.mean(np.array(o), axis=0) for o in Z["outlines"]]
    for i, (kps, sc) in enumerate(zip(P["kps"], P["sc"])):
        a, b = to_frame(P, i, cents); u = b - a; nrm = np.array([-u[1], u[0]]) / (np.linalg.norm(u) + 1e-9)
        for k, s in zip(kps, sc):
            tl = torso_len(k, s)
            if not tl or min(s[11], s[12]) < KP_MIN: continue
            side = np.sign(np.dot((k[11] + k[12]) / 2 - a, nrm)) or 1.0  # 사람이 서 있는 쪽
            pts = [k[j] for j in (7, 8, 9, 10) if s[j] >= KP_MIN]
            for wj, ej, sj in ((9, 7, 5), (10, 8, 6)):
                if s[wj] < KP_MIN and s[ej] >= KP_MIN and s[sj] >= KP_MIN: pts.append(k[ej] + 0.9 * (k[ej] - k[sj]))
            if not pts: continue
            x = max(-side * np.dot(q - a, nrm) for q in pts) / tl  # 반대편으로 넘어간 깊이
            out[i] = x if np.isnan(out[i]) else max(out[i], x)
    return out

def judge_x(F, X, p, cross_th, gate_body):
    """T.judge와 같되, 정지 조건에 '회전축 넘음'을 요구하고 넘지 않았으면 감속까지만"""
    F2 = dict(F); dh = F["dh"].copy(); db = F["db"].copy(); inside, body_in = p[2], p[3]
    blocked = ~(X > cross_th)  # 넘지 않음(또는 모름)
    dh2 = np.where(blocked & (dh < inside), np.maximum(dh, inside + 1e-3), dh)  # 손 조건으로는 정지 못 하게
    db2 = np.where(blocked & (db < body_in), np.maximum(db, body_in + 1e-3), db) if gate_body else db
    F2["dh"], F2["db"] = dh2, db2
    return T.judge(F2, p)[0]

# 데이터 1: 빨간 상의 확대 영상(09)
P9 = dict(np.load(DATA + "exp09/red_crop_pose_performance.npz", allow_pickle=True))
Z9 = json.load(open(os.path.join(R09, "zone_user.json")))
lab9 = {round(float(r["시각(초)"]), 1): r["내 판단(기존)"].strip() for r in csv.DictReader(open(DATA + "exp09/frames_humanlike/판단표_판단기비교.csv", encoding="utf-8-sig"))}
lab9.update({8.6: "정지", 22.6: "주의", 35.2: "정지"}); lab9[7.0] = ""
Zsave, Lsave = T.Z, T.L
T.Z, T.L = Z9, json.load(open(os.path.join(R09, "floor_user.json")))["lines"]
F9 = T.features(P9); X9 = crossing(P9, Z9)
T.Z, T.L = Zsave, Lsave
y9 = np.array([STAGES.index(lab9.get(round(round(x / 0.2) * 0.2, 1), "")) if lab9.get(round(round(x / 0.2) * 0.2, 1), "") in STAGES else -1 for x in F9["t"]])
# 데이터 2: 넓은 화면(10)
wide = [(F, crossing(P, T.Z), y) for P, F, y in T.clips.values()]

def score(p, mode, cross_th=0.0, gate_body=False):
    o9 = T.judge(F9, p)[0] if mode == "A" else judge_x(F9, X9, p, cross_th, gate_body)
    ow = np.concatenate([T.judge(F, p)[0] if mode == "A" else judge_x(F, X, p, cross_th, gate_body) for F, X, _ in wide])
    yw = np.concatenate([y for _, _, y in wide])
    return macro_f1(y9, o9)[0], macro_f1(yw, ow)[0], o9, ow, yw

res = {}
fa9, faw, *_ = score(T.p_red, "A"); res["A_as_is"] = {"red": round(fa9, 3), "wide": round(faw, 3)}
for gb in (False, True):
    fb9, fbw, *_ = score(T.p_red, "B", 0.0, gb); res[f"B_cross_gate_body={gb}"] = {"red": round(fb9, 3), "wide": round(fbw, 3)}
rng = np.random.default_rng(4); best = None
for _ in range(2500):  # C: 두 영상 평균 일치도로 함께 맞춤
    q = np.array(T.p_red) + rng.normal(0, 0.2, 12) * [1, 1, 1, 1, 0.3, 0, 1, 1, 0.3, 0.2, 0.5, 1]; q[5] = rng.choice([0.3, 0.5, 1.0])
    ct = rng.uniform(-0.6, 0.6); gb = bool(rng.integers(2))
    f9, fw, *_ = score(list(q), "B", ct, gb)
    if best is None or (f9 + fw) / 2 > best[0]: best = ((f9 + fw) / 2, list(q), ct, gb, f9, fw)
res["C_joint_refit"] = {"red": round(best[4], 3), "wide": round(best[5], 3), "cross_th": round(best[2], 3), "gate_body": best[3]}
_, _, o9, ow, yw = score(best[1], "B", best[2], best[3])
per = lambda o, y: {STAGES[c]: {STAGES[k]: int((o[y == c] == k).sum()) for k in range(4)} for c in range(4)}
res["C_per_stage_red"] = per(o9, y9); res["C_per_stage_wide"] = per(ow, yw)
res["C_stop_onsets_red"] = onsets(F9["t"], o9, 3); res["human_stop_red"] = onsets(F9["t"], np.where(y9 < 0, 0, y9), 3)
off = 0
for (F, X, y), n_ in zip(wide, T.CLIPS):
    o = ow[off:off + len(y)]; off += len(y)
    res[f"C_trim_{n_:02d}"] = {"human_stop": onsets(F["t"], np.where(y < 0, 0, y), 3), "judge_stop": onsets(F["t"], o, 3)}
res["X_probe_trim02"] = {f"{s:.1f}": round(float(wide[0][1][int(np.argmin(abs(wide[0][0]['t'] - s)))]), 2) for s in (0.4, 1.6, 3.0, 4.6, 5.2, 6.0)}
print(json.dumps(res, ensure_ascii=False, default=float), flush=True)
json.dump(res, open(os.path.join(HERE, "result_joint.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
json.dump({"params": best[1], "cross_th": best[2], "gate_body": best[3]}, open(os.path.join(HERE, "params_joint.json"), "w"), indent=1, default=float)
