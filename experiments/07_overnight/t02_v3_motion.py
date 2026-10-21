# 실험 07-2 (V3): 접근 속도·TTC. 손목이 회전체와 겹쳐도, 최근에 회전체 쪽으로 빠르게 다가온 경우에만 정지한다
# 비교 대상 V1(겹침만). 포즈 결과는 실험 06 캐시를 그대로 쓴다(포즈 고정)
# 임계값은 실행 전에 정함: 접근 속도 1.2 몸통/초(몸통≈0.5m → 약 0.6m/s, Kingwu12 저장소 값), 확인 창 1.0초, TTC 0.5초
# 보조로 임계값을 바꿔 본 결과(민감도)도 남기지만, 주 결과는 위 값이다
import numpy as np
from common import CASES, load_pose, rot_polys, signed_dist, torso, track, metrics, done, KP_MIN

V_TH, WIN_S, TTC_TH, LAG_S = 1.2, 1.0, 0.5, 0.3

def run(name, v_th):
    P = load_pose(name); t = P["t"]; fps = 1 / np.median(np.diff(t)); lag = max(1, int(LAG_S * fps)); win = int(WIN_S * fps)
    ids = track(P); hist = {}  # 사람 id → [(프레임, 손목-회전체 거리/몸통)]
    v1, v3 = [], []
    for i, (kps, sc) in enumerate(zip(P["kps"], P["sc"])):
        polys = rot_polys(name, P, i); hit1 = hit3 = False
        for pid, k, s in zip(ids[i], kps, sc):
            tl = torso(k, s) or hist.get(("torso", pid))
            if tl: hist[("torso", pid)] = tl
            ds = [signed_dist(k[j], polys) for j in (9, 10) if s[j] >= KP_MIN]
            if not ds or not tl: continue
            d = -max(ds) / tl  # 바깥 거리(몸통 단위), 안이면 음수
            h = hist.setdefault(pid, []); h.append((i, d))
            inside = d <= 0
            # 접근 속도: LAG 프레임 전 거리 - 지금 거리 (줄어들면 양수)
            vs = [(h[a][1] - h[b][1]) / ((h[b][0] - h[a][0]) / fps) for b in range(len(h)) for a in [b - lag] if a >= 0 and h[b][0] > i - win]
            vmax = max(vs) if vs else 0.0
            v_now = vs[-1] if vs and h[-1][0] == i else 0.0
            if inside: hit1 = True
            if inside and vmax >= v_th: hit3 = True
            if not inside and v_now >= v_th and d / v_now < TTC_TH: hit3 = True
        v1.append(hit1); v3.append(hit3)
    tc = CASES[name]["tc"]
    return metrics(t, v1, tc), metrics(t, v3, tc)

res = {}
for name in CASES:
    m1, m3 = run(name, V_TH)
    res[name] = {"V1_overlap": m1, "V3_motion": m3,
                 "sensitivity": {str(v): run(name, v)[1] for v in (0.6, 0.9, 1.5, 2.0)}}
    print(name, res[name])
res["thresholds"] = {"v_torso_per_s": V_TH, "window_s": WIN_S, "ttc_s": TTC_TH, "lag_s": LAG_S}
done("t02_v3_motion", res)
