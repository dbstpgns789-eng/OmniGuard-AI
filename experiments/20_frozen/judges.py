# 실험 20: 시험 영상을 보기 전에 판단기 3개를 고정한다 (2026-10-20). 이 파일의 값과 규칙은 시험 뒤에 바꾸지 않는다
# J13 원칙(실험 13) → RULE4 규칙 4(실험 19) → QUIET 조용한 안(실험 18). 특징 계산과 상태 기계는 실험 10 transfer.py 그대로
# 영상마다 필요한 입력: 포즈(P), 위험한계 윤곽(zone), 바닥선(floor, 걷는 방향용 2점 이상), 규칙 4는 위험한계마다 바닥 앞선(fronts)
import os, sys, numpy as np, cv2
X = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for d in ("09_red_fit", "10_isafety_transfer"): sys.path.insert(0, os.path.join(X, d))
import transfer as T
from common import KP_MIN, torso_len

NAMES = T.NAMES

def _p(**kw):
    p = list(T.p_red)  # 실험 09 t13 값(걷기 인식·유지·품질 규칙)
    for k, v in kw.items(): p[NAMES.index(k)] = v
    return p

PARAMS = {
    "J13": _p(near_warn=1.0, near_slow=0.5, hand_inside=0.0, body_inside=0.0),  # 손·몸이 경계에 닿으면 정지
    "RULE4": _p(hand_inside=-99.0, body_inside=0.0, near_slow=0.25, near_warn=0.5, hold_s=1.0),  # 정지는 아래 db_rule4로만
    "QUIET": _p(hand_inside=-0.7, body_inside=-99.0, near_slow=-0.45, near_warn=-0.2, hold_s=1.0),  # 손 안쪽 0.7, 몸만으로는 정지 안 함
}
LEGS, ARMS, DEEP = {13, 14, 15, 16}, {7, 8, 9, 10}, -0.7

def features(P, zone, floor, objects=None):
    T.Z = {"outlines": [o for o in zone["outlines"] if len(o) >= 10]}; T.L = floor["lines"]
    return T.features(P, objects)

def _side(p, a, b): return np.sign((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))

def db_rule4(P, F, fronts):
    """규칙 4: 17개 관절의 위험한계 거리(몸통 단위, 안쪽 음수) 중 최소. 발이 그 설비 앞선 바깥(사람 쪽)이면
    머리·어깨·엉덩이는 빼고 손·팔꿈치는 0.7 더 깊어야 경계로 본다. 무릎·발목은 항상 포함. 앞선이 없거나 발을 모르면 모든 관절(원칙)"""
    h = float(P["wh"][1]); n = len(P["t"]); db = np.full(n, np.nan); last = None
    fpx = [(np.array(l[0]) * P["wh"], np.array(l[1]) * P["wh"]) for l in (fronts or [])]
    for i, (K, S) in enumerate(zip(P["kps"], P["sc"])):
        hl = F["hull_at"](i); hl = [hh.astype(np.float32) for hh in (hl if isinstance(hl, list) else [hl])]; b = np.inf
        for k, s in zip(K, S):
            tl = torso_len(k, s) or last or 0.25 * h
            if torso_len(k, s): last = tl
            feet = [k[j] for j in (15, 16) if s[j] >= KP_MIN]; fp = np.mean(feet, 0) if feet else None
            for j in range(17):
                if s[j] < KP_MIN: continue
                ds = [-cv2.pointPolygonTest(hh, (float(k[j][0]), float(k[j][1])), True) / tl for hh in hl]; m = int(np.argmin(ds)); d = ds[m]
                outside = False
                if fp is not None and len(fpx) == len(hl):
                    a, bb = fpx[m]; outside = _side(fp, a, bb) != _side(hl[m].mean(0), a, bb)
                if j in LEGS or not outside: b = min(b, d)
                elif j in ARMS: b = min(b, d - DEEP)
        if b < np.inf: db[i] = b
    return db

def run(name, P, F, fronts=None):
    """판단기 하나를 돌려 프레임마다 단계(0 정상, 1 주의, 2 감속, 3 정지)를 돌려준다"""
    if name == "RULE4": F = dict(F); F["db"] = db_rule4(P, F, fronts)
    return T.judge(F, PARAMS[name])[0]
