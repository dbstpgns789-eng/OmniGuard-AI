# 실험 14: 영상마다 바닥 평면을 잡는다(실험 09-2 방식). 바닥 선 4개 → 소실점 2개 → 초점 거리·바닥 법선, 축척은 선 사람의 코 높이 1.55m(키 약 1.7m)
import numpy as np, cv2
KP_MIN = 0.3; NOSE_H = 1.55
hom = lambda p: np.array([p[0], p[1], 1.0])

class Floor:
    def __init__(self, lines, P, fov_deg=None):  # fov_deg를 주면 초점 거리를 소실점 대신 가정한 화각으로 정함(원근이 약한 화면용)
        self.P = P; w, h = P["wh"]; Lp = np.array(lines) * [w, h]
        vh = lambda a, b, c, d: np.cross(np.cross(hom(a), hom(b)), np.cross(hom(c), hom(d)))  # 동차 좌표(무한히 먼 소실점도 표현)
        V1, V2 = vh(*Lp[0:4]), vh(*Lp[4:8]); c0 = np.array([w / 2, h / 2])
        if fov_deg: self.f = w / 2 / np.tan(np.radians(fov_deg) / 2)
        else: self.f = float(np.sqrt(-np.dot(V1[:2] / V1[2] - c0, V2[:2] / V2[2] - c0)))
        K = np.array([[self.f, 0, c0[0]], [0, self.f, c0[1]], [0, 0, 1]]); self.Ki = np.linalg.inv(K)
        r1 = self.Ki @ V1; r1 /= np.linalg.norm(r1); r2 = self.Ki @ V2; r2 -= r1 * np.dot(r1, r2); r2 /= np.linalg.norm(r2)
        self.along = r1  # 설비 방향(바닥 위 단위 벡터)
        n = np.cross(r1, r2); n /= np.linalg.norm(n)
        if np.dot(n, self.Ki @ hom((w / 2, h))) > 0: n = -n  # 위쪽을 향하게
        self.n = n; self.Hc = 1.0; self.wh = (w, h)
    def ray(self, px, i):  # i번째 프레임 픽셀 → 기준 화면 → 시선
        q = cv2.perspectiveTransform(np.array(px, np.float64).reshape(-1, 1, 2), np.linalg.inv(self.P["H"][i])).reshape(-1, 2)
        d = (self.Ki @ np.c_[q, np.ones(len(q))].T).T; return d / np.linalg.norm(d, axis=1, keepdims=True)
    def ray_ref(self, ratio):  # 기준 화면 비율 → 시선
        q = np.array(ratio, float).reshape(-1, 2) * self.wh; d = (self.Ki @ np.c_[q, np.ones(len(q))].T).T; return d / np.linalg.norm(d, axis=1, keepdims=True)
    def on_floor(self, d): return -self.Hc * d / (d @ self.n)[:, None]  # 바닥 위 점(m, 카메라 좌표)
    def fit_scale(self):  # 똑바로 선 사람: 발 위 수직선에서 코 시선에 가장 가까운 점의 높이 = 1.55m
        hs = []
        for i, (K, S) in enumerate(zip(self.P["kps"], self.P["sc"])):
            for k, s in zip(K, S):
                if min(s[0], s[15], s[16], s[5], s[6], s[11], s[12]) < KP_MIN: continue
                sh, hp = (k[5] + k[6]) / 2, (k[11] + k[12]) / 2; v = sh - hp
                if abs(v[0]) > 0.25 * abs(v[1]): continue  # 몸통이 기울면 제외
                F = self.on_floor(self.ray([(k[15] + k[16]) / 2], i))[0]; dn = self.ray([k[0]], i)[0]
                A = np.array([[1, -np.dot(self.n, dn)], [np.dot(self.n, dn), -1]]); b = np.array([-np.dot(F, self.n), -np.dot(F, dn)])
                s_, _ = np.linalg.solve(A, b)
                if s_ > 0: hs.append(s_)
        self.n_upright = len(hs); self.hs_raw = hs
        if hs: self.Hc = NOSE_H / float(np.median(hs))
        return self
    def foot(self, k, s, i):  # 발목 중점(하나만 보이면 그것) → 바닥 점, 안 보이면 None
        a = [k[j] for j in (15, 16) if s[j] >= KP_MIN]
        return None if not a else self.on_floor(self.ray([np.mean(a, 0)], i))[0]
    def seg(self, line):  # 바닥 앞선(기준 화면 비율 2점) → 바닥 위 선분
        return self.on_floor(self.ray_ref(line))

def seg_dist(p, ab):
    a, b = ab; u = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1); return float(np.linalg.norm(p - (a + u * (b - a))))
