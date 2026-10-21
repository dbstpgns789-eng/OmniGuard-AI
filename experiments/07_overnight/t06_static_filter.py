# 실험 07-6: 추적으로 "움직이지 않는 사람"(설비를 사람으로 잘못 잡은 것)을 걸러내면 오정지가 줄어드는가
# 실행 전에 정한 규칙: 같은 id로 3초 이상 이어지고, 몸 중심의 이동 표준편차가 화면 폭의 1% 미만이면 정적 오검출로 보고 제외
# 비교: V1(겹침만, 손목) 원본 vs 정적 오검출 제외. 포즈 결과는 실험 06 캐시
import numpy as np, cv2
from common import CASES, load_pose, rot_polys, center, track, metrics, done, KP_MIN

MIN_S, STD_TH = 3.0, 0.01
res = {}
for name, cfg in CASES.items():
    P = load_pose(name); t = P["t"]; w = P["wh"][0]; ids = track(P)
    path = {}
    for i, (kps, sc) in enumerate(zip(P["kps"], P["sc"])):
        for pid, k, s in zip(ids[i], kps, sc):
            c = center(k, s)
            if c is not None: path.setdefault(pid, []).append((t[i], c))
    static = {pid for pid, p in path.items() if p[-1][0] - p[0][0] >= MIN_S and np.std([c for _, c in p], axis=0).max() / w < STD_TH}
    raw, raw_f = [], []
    for i, (kps, sc) in enumerate(zip(P["kps"], P["sc"])):
        polys = rot_polys(name, P, i); a = b = False
        for pid, k, s in zip(ids[i], kps, sc):
            hit = any(s[j] >= KP_MIN and max(cv2.pointPolygonTest(p.astype(np.float32), (float(k[j][0]), float(k[j][1])), False) for p in polys) >= 0 for j in (9, 10))
            a |= hit; b |= hit and pid not in static
        raw.append(a); raw_f.append(b)
    res[name] = {"tracks": len(path), "static_tracks": len(static), "V1": metrics(t, raw, cfg["tc"]), "V1_static_removed": metrics(t, raw_f, cfg["tc"])}
    print(name, res[name])
res["rule"] = {"min_s": MIN_S, "std_ratio": STD_TH}
done("t06_static_filter", res)
