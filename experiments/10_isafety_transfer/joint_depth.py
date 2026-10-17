# 실험 10-3: 3D 포즈 깊이 규칙. 손(또는 팔꿈치)이 엉덩이보다 깊이 들어갔을 때(롤 쪽)만 손 조건으로 정지, 아니면 감속까지
# 깊이 dz = min(손목·팔꿈치 깊이) - 엉덩이 깊이 (RTMW3D z, 음수일수록 카메라에서 멀다 = 롤 쪽). 2프레임마다 계산한 값
# 두 영상(빨간 상의 확대 / 넓은 화면)을 함께 채점:  A 그대로  B 깊이 규칙(기준 몇 개 고정 비교)  C 두 영상에 함께 다시 맞춤
import os, sys, csv, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R09 = os.path.join(HERE, "..", "09_red_fit"); sys.path.insert(0, R09)
from common import STAGES, macro_f1, onsets
import transfer as T

DATA = "C:/Users/User/Desktop/OmniGuard-data/"

def depth_feature(zfile):
    D = np.load(zfile, allow_pickle=True); out = []
    for zs, k2s in zip(D["z"], D["k2"]):
        best = None
        for z, k2 in zip(zs, k2s):
            size = np.ptp(k2[:, 1])  # 화면에서 가장 크게 보이는 사람 = 주 작업자
            dz = min(z[7], z[8], z[9], z[10]) - (z[11] + z[12]) / 2
            if best is None or size > best[0]: best = (size, dz)
        out.append(np.nan if best is None else best[1])
    return np.array(out, float)

def judge_z(F, DZ, p, z_th, gate_body):
    F2 = dict(F); dh, db = F["dh"].copy(), F["db"].copy(); inside, body_in = p[2], p[3]
    blocked = ~(DZ < z_th)  # 손이 몸보다 충분히 안쪽이 아님(또는 모름)
    F2["dh"] = np.where(blocked & (dh < inside), inside + 1e-3, dh)
    if gate_body: F2["db"] = np.where(blocked & (db < body_in), body_in + 1e-3, db)
    return T.judge(F2, p)[0]

# 빨간 상의(09) 데이터
P9 = dict(np.load(DATA + "exp09/red_crop_pose_performance.npz", allow_pickle=True))
lab9 = {round(float(r["시각(초)"]), 1): r["내 판단(기존)"].strip() for r in csv.DictReader(open(DATA + "exp09/frames_humanlike/판단표_판단기비교.csv", encoding="utf-8-sig"))}
lab9.update({8.6: "정지", 22.6: "주의", 35.2: "정지"}); lab9[7.0] = ""
Zs, Ls = T.Z, T.L
T.Z, T.L = json.load(open(os.path.join(R09, "zone_user.json"))), json.load(open(os.path.join(R09, "floor_user.json")))["lines"]
F9 = T.features(P9); T.Z, T.L = Zs, Ls
y9 = np.array([STAGES.index(lab9.get(round(round(x / 0.2) * 0.2, 1), "")) if lab9.get(round(round(x / 0.2) * 0.2, 1), "") in STAGES else -1 for x in F9["t"]])
DZ9 = depth_feature(DATA + "exp10/red_crop_z.npz")
wide = [(F, depth_feature(DATA + f"exp10/trim_{n:02d}_z.npz"), y) for (P, F, y), n in zip(T.clips.values(), T.CLIPS)]
yw = np.concatenate([w[2] for w in wide])

def score(p, z_th=None, gb=False):
    j = (lambda F, D: T.judge(F, p)[0]) if z_th is None else (lambda F, D: judge_z(F, D, p, z_th, gb))
    o9 = j(F9, DZ9); ow = np.concatenate([j(F, D) for F, D, _ in wide])
    return macro_f1(y9, o9)[0], macro_f1(yw, ow)[0], o9, ow

# 사람 판단별 깊이 분포(먼저 확인): 정지 장면에서 손이 몸보다 안쪽인가
dist = {}
for name, DZ, y in (("red", DZ9, y9), ("wide", np.concatenate([w[1] for w in wide]), yw)):
    dist[name] = {STAGES[c]: [round(float(x), 3) for x in np.nanpercentile(DZ[y == c], [25, 50, 75])] if np.isfinite(DZ[y == c]).any() else None for c in range(4)}
res = {"dz_by_human_stage(25/50/75%)": dist}
a9, aw, *_ = score(T.p_red); res["A_as_is"] = {"red": round(a9, 3), "wide": round(aw, 3)}
for zt in (0.0, -0.05, -0.1, -0.15):
    for gb in (False, True):
        b9, bw, *_ = score(T.p_red, zt, gb); res[f"B_z<{zt}_gate_body={gb}"] = {"red": round(b9, 3), "wide": round(bw, 3)}
rng = np.random.default_rng(5); best = None
for _ in range(2500):
    q = np.array(T.p_red) + rng.normal(0, 0.2, 12) * [1, 1, 1, 1, 0.3, 0, 1, 1, 0.3, 0.2, 0.5, 1]; q[5] = rng.choice([0.3, 0.5, 1.0])
    zt = rng.uniform(-0.3, 0.1); gb = bool(rng.integers(2)); f9, fw, *_ = score(list(q), zt, gb)
    if best is None or (f9 + fw) / 2 > best[0]: best = ((f9 + fw) / 2, list(q), zt, gb, f9, fw)
res["C_joint_refit"] = {"red": round(best[4], 3), "wide": round(best[5], 3), "z_th": round(best[2], 3), "gate_body": best[3]}
_, _, o9, ow = score(best[1], best[2], best[3])
per = lambda o, y: {STAGES[c]: {STAGES[k]: int((o[y == c] == k).sum()) for k in range(4)} for c in range(4)}
res["C_per_stage_red"] = per(o9, y9); res["C_per_stage_wide"] = per(ow, yw)
res["C_stop_red"] = onsets(F9["t"], o9, 3); res["human_stop_red"] = onsets(F9["t"], np.where(y9 < 0, 0, y9), 3)
off = 0
for (F, D, y), n in zip(wide, T.CLIPS):
    o = ow[off:off + len(y)]; off += len(y); res[f"C_trim_{n:02d}"] = {"human_stop": onsets(F["t"], np.where(y < 0, 0, y), 3), "judge_stop": onsets(F["t"], o, 3)}
np.save(os.path.join(HERE, "pred_depth_red.npy"), o9); np.save(os.path.join(HERE, "pred_depth_wide.npy"), ow)
json.dump({"params": best[1], "z_th": best[2], "gate_body": best[3]}, open(os.path.join(HERE, "params_depth.json"), "w"), indent=1, default=float)
json.dump(res, open(os.path.join(HERE, "result_depth.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, default=float), flush=True)
