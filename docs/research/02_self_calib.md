# 02. 사람으로 카메라 자동 보정 (Aside, 2026-10-19)

결론: 사람만으로 하는 보정은 원근이 있는 영상에서는 쓸 만하다. 줌이 걸렸거나 거의 정면인 영상(우리가 실패한 3개)에서는 수학적으로 잘 풀리지 않으므로, 기계 치수 같은 알려진 길이를 함께 써야 한다. 바닥 평면을 고쳐도 "손 닿는 거리 안에서 정상 작업 중인 사람" 헛정지는 해결되지 않는다.

## A. 보행자 기반 보정 (고전 기하)
1. **Lv, Zhao, Nevatia 2006 (TPAMI)** https://pubmed.ncbi.nlm.nih.gov/16929736 — 걷는 사람의 다리가 모인 순간의 머리·발 선분으로 보정. 수평 소실점에는 바닥 보조선 2개가 필요. → 양 발목이 가까운 프레임만 쓰는 필터
2. **Foot-head homology (Krahnstoever ICCV 2005, Micusik CVPR 2010)** — 발→머리 대응을 평면 호몰로지로 보고 베이지안으로 보정. 수치 확인 못 함. → h(y) 직선 근사의 이론적 근거
3. **Liu, Collins, Liu 2011 (BMVC)** — 검출·추적 없이 키 분포 prior(평균의 90~110%)로 보정. 비교표상 Δf 347px. → 작업자 키 편차 흡수
4. **Brouwers et al. 2016 (ECCV-W)** — 보행자 트랙으로 수직 소실점 → 지평선 → f → 평균 키로 축척. 평균 오차 1.95%(시뮬), 실제 최대 3.7%
5. **Huang et al. 2016 (CVPR)** https://openaccess.thecvf.com/content_cvpr_2016/papers/Huang_Camera_Calibration_From_CVPR_2016_paper.pdf — 직선으로 걷는 한 사람의 발끝 궤적만으로 소실점 3개 복원. 바닥 선 불필요. 발끝 키포인트 필요(Halpe26)
6. **ESTHER 2019 (IEEE Access)** — 수직 소실점 mean-shift, 지평선 Laplace 회귀, "사람별 추정 키의 분산 최소화"로 최적화. Δf 121.5px. → 검증 지표: 같은 사람의 추정 키가 화면 위치와 무관하게 일정한가
7. **Fei et al. 2021 (Amazon)** https://arxiv.org/abs/2106.10335 — 발목 중점·어깨 중점만으로 f·바닥 평면·사람 간 거리를 선형 방정식으로 직접 풂. 노이즈 1px: f 6%, 바닥 거리 2.4% / 5px: f 28%, 거리 12.4%. 사람 5명이면 f 오차 21%, 100명이면 5%. 실제 MEVA f 오차 평균 23~25%. → RTMPose 17점으로 바로 가능, 여러 프레임을 모을수록 좋아짐

## B. 소실점 하나 / 약원근
8. **Criminisi 2000 (IJCV), Single View Metrology** — 소실선 + 수직 소실점만 있으면 f 없이 길이비·높이비·카메라 위치, 오차 전파까지. → 줌 영상은 f 추정을 우회
9. **원근 맵 h(y)=a·y+b (Shi 2019, Zhang 2015)** https://arxiv.org/html/1807.01989v3 — 키 1.75m 고정, 여러 위치의 사람 키로 선형회귀. h(y)=0인 행이 지평선, h(y)/H가 그 행의 px/m. **기울기 a≈0이면 약원근이라는 진단도 겸함** → 가장 싸고 견고

## C. 학습 기반 단일 영상 보정 (교차 검증용)
10. **ScaleNet (ECCV 2020)** https://github.com/Jerrypiglet/ScaleNet — 설치가 무거워 비추천
11. **SPEC (ICCV 2021)** https://github.com/mkocabas/SPEC — 학습 화각 70~130°라 좁은 화각 CCTV는 분포 밖일 가능성
12. **Perspective Fields (CVPR 2023)** https://github.com/jinlinyi/PerspectiveFields — 잘린 화면에도 대응. 육안 검증용
13. **GeoCalib (ECCV 2024)** https://github.com/cvg/GeoCalib — 여러 프레임을 같은 카메라로 묶어 보정. pitch 중앙값 오차 0.87~1.94°, FoV 3~4.5°. 이미지당 약 100ms. 주점 중심 가정
14. **lab-camera-dynamic-calibrator** https://github.com/flodelaplace/lab-camera-dynamic-calibrator — RTMPose Halpe26 발 키포인트로 바닥 평면, 머리-발 벡터로 수직축

## 코드 (Aside 제공, roll 0·주점 중심 가정, 일부 식은 Aside가 유도해 검증 안 됨)
```python
import numpy as np, cv2
# P: N x 4 [xf, yf, xs, ys], 똑바로 서고 다리를 모은 프레임만
def fit_h_of_y(P, tol=0.08, iters=500, seed=0):
    yf, h = P[:,1], P[:,1]-P[:,3]
    rng, best = np.random.default_rng(seed), None
    for _ in range(iters):
        i, j = rng.choice(len(yf), 2, replace=False)
        if abs(yf[i]-yf[j]) < 5: continue
        a = (h[i]-h[j])/(yf[i]-yf[j]); b = h[i]-a*yf[i]
        inl = np.abs(h-(a*yf+b)) < tol*h
        if best is None or inl.sum() > best.sum(): best = inl
    a, b = np.polyfit(yf[best], h[best], 1)
    return a, b, best                        # 지평선 행 y_hor = -b/a
def camera_from_people(y_hor, y_v, cx, cy):
    f2 = (cy - y_hor)*(y_v - cy)
    if f2 <= 0: return None                  # 퇴화 → 약원근 경로
    f = np.sqrt(f2); th = np.arctan2(cy - y_hor, f); return f, th
def cam_height(yf, yt, y_hor, y_v, H):       # Criminisi cross-ratio
    r = (abs(yf-y_hor)*abs(yt-y_v))/(abs(yt-y_hor)*abs(yf-y_v)); return r*H/(r-1)
def ground_H(f, th, Hc, cx, cy):              # 바닥(m) → 화면
    K = np.array([[f,0,cx],[0,f,cy],[0,0,1.]]); s, c = np.sin(th), np.cos(th)
    return K @ np.array([[1,0,0],[0,-s,c*Hc],[0,c,s*Hc]])
```

## 약원근 경로와 경고 (Aside 유도)
- 약원근: 가로 길이 s·X, 사람 높이 s·H·cosθ, 바닥 깊이 s·Y·sinθ. 롤 길이(사양서)로 s, 서 있는 사람 픽셀 높이로 cosθ
- 가장 간단한 것: 바닥 테이프·기계 받침대 실측 4점으로 getPerspectiveTransform (원근·약원근 모두 동작)
- **깊이 해상도 경고:** s=100px/m, θ=15°면 바닥 깊이 26px/m. 발목 노이즈 5~10px → 깊이 오차 ±0.2~0.4m. **0.5~0.9m와 1.0m를 가르는 발 게이트는 노이즈 수준** → 실험 14에서 발 게이트가 효과 없었던 이유 중 하나일 수 있음

## 검증 체크
1. 추정 카메라로 역산한 사람 키의 화면 위치별 변동계수 5~8% 이하
2. h = a·y + c·x + b에서 c가 크면 roll·왜곡
3. 지평선은 모든 머리 위, 수직 소실점은 화면 아래, f²≤0이면 약원근
4. 롤 길이·기계 폭을 바닥에서 재서 사양 대비 10% 이내
5. 걷는 속도가 가로·깊이 방향 모두 1.0~1.6m/s
6. 0.5m 격자를 영상에 겹쳐 바닥 줄눈과 비교
7. GeoCalib pitch와 5° 이상 차이 나면 표시

## 1주 계획 (Aside 제안)
1~2일: 8개 영상의 h(y) 피팅, 기울기로 원근/약원근 분류, 모든 거리를 "그 위치의 몸 크기 단위"(d_px/h(y))로 정규화 → 영상 간 전이 문제 감소. 3일: 원근 영상은 ground homography. 4일: 약원근 3개는 기계 치수 4점 또는 롤 길이 + 사람 키. 5일: 검증 표. 6~7일: 규칙을 미터 특징(발의 위험선 방향 속도, 진행 방향-기계 각도, 체류 시간, 손이 맞물림부 법선 방향으로 들어온 깊이)으로 다시 쓰고 영상 하나씩 빼는 교차검증.

## Claude 메모
- 러닝머신 촬영에서는 바닥 테이프 4점으로 보정하면 이 문제 전체를 피할 수 있다(촬영 계획에 반영할 것).
- "d_px / h(y)" 정규화는 지금의 "몸통 길이 단위"보다 위치에 덜 흔들리는 단위라 실험 13 판단기에 바로 넣어 볼 만하다.
