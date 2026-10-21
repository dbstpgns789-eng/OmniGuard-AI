# 12. 단안 깊이 추정 오픈소스 조사 (2026-10-20, Aside)

실험 07(Depth Anything V2 Small, 상대)·08(DA V2 Metric-Indoor-Small)에서 손-롤 거리 닿음 0.03m / 안 닿음 0.05m로 구분 실패, CPU 1초/장. 그 뒤 무엇을 다시 시도할 만한지 조사.

결론부터 말하면, 지금 구조로는 단안 미터 깊이로 손과 롤 사이 2cm 차이를 가르기 어렵습니다. 최상위 모델도 실내 오차가 약 10~20cm라서, 0.03m와 0.05m 차이는 오차 안에 묻힙니다. 카메라 2대로 손 위치를 삼각측량하는 쪽이 훨씬 유망합니다.

조사는 4갈래로 나눠 병렬로 했고, GitHub 스타 수와 라이선스 파일은 제가 GitHub API로 직접 확인했습니다. 표기는 [원문] = 논문·README·모델 카드, [2차] = 다른 논문 인용이나 블로그, "확인 못 함" = 이번 조사에서 근거를 못 찾음입니다.

## 1) 주요 모델 기본 정보

| 모델 | 깊이 종류 | 크기(파라미터) | 논문 (arXiv, 1판 날짜, 학회) | GitHub (★, 10/20 기준) | 라이선스 [원문] | HF ID |
|---|---|---|---|---|---|---|
| Depth Anything V2 | 상대 (Hypersim·VKITTI로 미세조정한 미터 버전 따로 있음) | S 24.8M / B 97.5M / L 335.3M | 2406.09414, 2024-06-13, NeurIPS 2024 | DepthAnything/Depth-Anything-V2 (9,002) | 코드 Apache-2.0. 가중치: S는 Apache, B·L은 CC-BY-NC-4.0 | depth-anything/Depth-Anything-V2-{Small,Base,Large}(-hf), ...-Metric-Indoor-Small-hf 등 |
| Depth Anything 3 | 다시점 기반. METRIC-LARGE는 미터 (초점거리 × 출력 / 300) | S 0.08B / B 0.12B / L 0.35B / G 1.15B | 2511.10647, 2025-11-13, ICLR 2026 | ByteDance-Seed/Depth-Anything-3 (6,530) | 코드 Apache-2.0. S·B·Metric-L·Mono-L은 Apache, L·G·Nested는 CC-BY-NC | depth-anything/DA3-SMALL, DA3METRIC-LARGE 등 |
| Video Depth Anything | 상대, 미터 버전 추가(2025-08) | S 28.4M / B 113.1M / L 381.8M | 2501.12375, 2025-01-21, CVPR 2025 Highlight | DepthAnything/Video-Depth-Anything (2,279) | S는 Apache, B·L은 CC-BY-NC | depth-anything/(Metric-)Video-Depth-Anything-Small |
| Depth Pro (Apple) | 미터 + 초점거리 추정 | 504M | 2410.02073, 2024-10-02, ICLR 2025 | apple/ml-depth-pro (5,757) | Apple 자체 라이선스 (HF 태그 apple-amlr) | apple/DepthPro(-hf) |
| UniDepth V2 | 미터 + 카메라 + 신뢰도 | S 34.2M / B 114.4M / L 353.8M | 2502.20110, 2025-02-27, TPAMI (DOI 기준) | lpiccinelli-eth/UniDepth (1,273) | **CC-BY-NC-4.0** | lpiccinelli/unidepth-v2-vits14 등 |
| Metric3D v2 | 미터 + 표면 법선 | ViT-S/L/giant2. L 411.9M [2차] | 2404.15506, TPAMI 2024 | YvanYin/Metric3D (2,330) | 코드 BSD-2. 가중치 확인 못 함 | JUGGHM/Metric3D, onnx-community/metric3d-vit-small |
| MoGe / MoGe-2 | MoGe-1은 상대 점군, MoGe-2는 미터 점군·깊이 | MoGe-2: S 35M / B 104M / L 326M | 2410.19115 (CVPR 2025) / 2507.02546 (2025-07-03, 학회 확인 못 함) | microsoft/MoGe (3,072) | MIT (가중치 포함) | Ruicheng/moge-2-vits-normal 등 (-onnx 버전 있음) |
| Marigold | 상대 (디퓨전 기반) | 약 866M (HF 기준) | 2312.02145, 2023-12-04, CVPR 2024 Oral | prs-eth/Marigold (3,242) | 코드 Apache. v1-1 가중치는 RAIL++-M | prs-eth/marigold-depth-v1-1 |
| ZoeDepth | 미터 | 약 345M | 2302.12288, 2023-02-23 | isl-org/ZoeDepth (2,836, 유지보수 중단) | MIT | Intel/zoedepth-nyu-kitti |
| DepthCrafter | 상대, 영상용 | 확인 못 함 | 2409.02095, 2024-09-03, CVPR 2025 Highlight | Tencent/DepthCrafter (1,593) | **연구용만, 상업 사용 금지** | tencent/DepthCrafter |
| VGGT (Meta) | 다시점 (카메라·깊이·점군) | 1.26B | 2503.11651, 2025-03-14, CVPR 2025 | facebookresearch/vggt (14,536) | VGGT-1B 가중치는 CC-BY-NC. Commercial 버전은 신청제 | facebook/VGGT-1B(-Commercial) |
| (추가) UniK3D | 미터, 모든 카메라 형식 | 358.8M | 2503.16591 | lpiccinelli-eth/UniK3D (757) | CC-BY-NC-SA-4.0 | lpiccinelli/unik3d-vitl |

라이선스 주의:
- 지금 쓰고 있는 DA V2 Metric-Indoor-Small-hf 저장소에는 라이선스 태그가 없습니다. README 기준으로는 Small이 Apache-2.0입니다.
- DA V2 Metric-VKITTI-Large와 DA3-LARGE-1.1은 HF 태그와 README 라이선스가 서로 다릅니다. 실제로 쓰기 전에 직접 확인이 필요합니다.

## 1) 보고된 정확도와 속도 ([원문], 수치는 원 표 그대로)

| 모델 | NYU AbsRel / δ1 | KITTI AbsRel / δ1 | 조건 | 속도 (GPU) |
|---|---|---|---|---|
| DA V2-S (상대) | 0.053 / 0.973 | 0.078 / 0.936 | 학습에 안 쓴 데이터(제로샷) | V100 60ms |
| DA V2-S (미터) | 0.073 / 0.961 | 0.053 / 0.973 | NYU·KITTI로 미세조정 | 위와 같음 |
| DA V2-L (상대) | 0.045 / 0.979 | 0.074 / 0.946 | 제로샷 | V100 213ms |
| DA3 (상대) | δ1 97.4 | δ1 95.3 | 제로샷. AbsRel 확인 못 함 | A100 기준 Small 160 FPS |
| VDA-S-Metric | δ1 0.850 | δ1 0.877 | README 수치 | A100 9.1ms |
| Depth Pro | δ1 0.953 [2차, VDA README] | δ1 0.822 [2차] | 원 논문엔 NYU·KITTI 수치 없음 | V100 341ms |
| UniDepthV2-L | 4.68% / 98.8 | 3.73% / 98.9 | 미세조정 | A6000 S 23ms, L 65ms |
| Metric3D v2 ViT-L | 0.047 / 0.989 | 0.044 / 0.985 | 미세조정 | A6000 134ms [2차] |
| MoGe-2 (미터 깊이) | 6.46% / 96.9 | 8.64% / 93.7 | 제로샷, 실제 카메라 내부값 사용 | A100 29ms (FP16) |
| Marigold v1.0 | 5.5% / 96.4 | 9.9% / 91.6 | 제로샷, 앙상블 10회 | 1070ms [2차] |
| ZoeDepth-N | 0.075 / 0.955 | 0.057 (NK 버전) | 미세조정 | 확인 못 함 |
| DepthCrafter | 확인 못 함 | 0.104 / 0.896 | 영상 | A100 약 2.1 FPS |

CPU 속도는 6개 주요 모델 모두 공식 수치를 찾지 못했습니다. 유일한 CPU 계열 수치는 4번의 OpenVINO 노트북입니다.

## 2) 가까운 거리·작은 물체에서의 미터 오차

- **Depth Pro 논문 (2410.02073) [원문]:** iBims-1 실내 데이터셋의 평면 오차가 Depth Pro, DA V2, Metric3D v2 모두 0.095m, ZoeDepth는 0.103m입니다. 최상위 모델도 약 9.5cm 수준입니다.
- **Depth Anything V1 metric README [원문]:**
  - NYU로 미세조정한 경우 RMSE 0.206m, ZoeDepth는 0.282m입니다.
  - 학습에 안 쓴 데이터에서는 iBims AbsRel 0.150, SUN RGB-D 0.500까지 나빠집니다.
- **환산 (제 계산):** AbsRel 5%면 3m 거리에서 평균 약 15cm 오차입니다.
- **DA V2를 LiDAR와 비교한 연구 (2025-11, ResearchGate) [2차]:** 평균 오차 0.32~0.66m입니다.
- **손-물체 접촉 기준의 cm 단위 평가는 찾지 못했습니다.** 단안 깊이를 HO3D·DexYCB·ARCTIC 같은 손-물체 데이터셋에서 cm로 평가한 자료는 확인 못 함입니다.

즉 지금 측정값(닿음 0.03m, 안 닿음 0.05m)의 차이는 모델 오차보다 한 자릿수 작습니다. 구분이 안 된 건 예상된 결과입니다.

## 3) 프레임마다 깊이가 흔들리는 문제

**VDA 논문 Table 1 [원문]:** ScanNet에서 프레임 간 흔들림 지표(TAE, 낮을수록 좋음)를 비교했습니다.

| 모델 | TAE |
|---|---|
| VDA-L | 0.570 |
| DepthCrafter | 0.639 |
| VDA-S | 0.703 |
| ChronoDepth | 1.022 |
| DA V2-L (단일 이미지용) | 1.140 |

**흔들림을 줄인 모델들:**
- **Video Depth Anything-S:** Apache-2.0, 28.4M. 흔들림이 가장 적은 경량 모델이고, 실시간 스트리밍 모드와 미터 버전도 있습니다. 다만 스트리밍 모드는 성능이 떨어집니다(ScanNet δ1 0.926 → 0.836) [원문].
- **FlashDepth:** arXiv 2504.07093, ICCV 2025, Apache-2.0 [원문].
- **RollingDepth:** CVPR 2025, 코드 Apache [원문].
- **DepthCrafter:** 상업 사용 금지라 제외 대상입니다.

흔들림을 줄여도 절대 정확도 문제는 그대로 남습니다.

## 4) CPU·엣지용 경량 변환본

- **OpenVINO 공식 노트북 (DA V2-Small, INT8 양자화)** [원문]: https://docs.openvino.ai/2024/notebooks/depth-anything-v2-with-output.html
  - 320×180 입력에서 원본 10.7 FPS, INT8 13.2 FPS입니다.
  - 어떤 하드웨어에서 측정했는지는 확인 못 함입니다.
  - 지금 1초/장보다 크게 빨라질 여지가 보입니다.
- **onnx-community/depth-anything-v2-small(-ONNX)** [원문]: Apache-2.0, fp16·q4 양자화 버전이 있습니다.
- **fabio-sim/Depth-Anything-ONNX** [원문]: Apache-2.0, ★446. 보고된 속도는 GPU(RTX4080) 13ms뿐입니다.
- **기타 ONNX 변환본** [원문]:
  - onnx-community/metric3d-vit-small (cc0)
  - onnx-community/DepthPro-ONNX (Apple 라이선스, 504M이라 CPU에는 무거움)
  - Ruicheng/moge-2-*-onnx
- **Distill-Any-Depth** [원문]: MIT, 24.8M, 상대 깊이만 지원합니다.
- **확인 못 함:** Lite-Mono, FastDepth, MiDaS small의 CPU 수치.

## 5) 산업안전에 단안 깊이를 쓴 사례와 한계

- **Ding & Luo, 건설현장 사람-장비 근접 경고** (arXiv 2305.17931, 2023-05. 저널판은 *Eng. Appl. of AI* 2025-11) [원문]
  - 단안 3D 검출로 거리를 4단계 위험 등급으로 분류합니다. 20m 이내 AP 75%입니다.
  - **한계:** 거리가 멀수록 정확도가 떨어지고, cm 단위가 아니라 등급 분류에 그칩니다.
- **MiDaS를 Raspberry Pi에서 돌려 안전구역 침입 경보** (IEEE Xplore 11307624, 2025-10) [원문 초록]
  - "대략적인 미터 거리" 수준이라고만 밝힙니다.
- **KineDepth** (arXiv 2409.19490) [원문]
  - 크기를 아는 로봇 형상으로 단안 깊이의 축척을 보정해 오차를 22.1% 줄였습니다.
  - **한계:** 기준이 되는 물체가 필요합니다.
- **사람-로봇 협업 안전 연구는 대부분 깊이 센서를 씁니다.** 예: Kinect v2 (Procedia CIRP 2018), RGB-D 기반 (arXiv 2511.19094) [원문].
- **손-기계 접촉이나 끼임 판정에 단안 깊이를 쓴 논문은 찾지 못했습니다.**
- **관련 표준 (NIST, PMC5117641) [원문]:** ISO/TS 15066의 속도·간격 감시 방식, ISO 13855의 접근 속도 2000mm/s 같은 기준이 있습니다. 센서 정확도 요구치는 확인 못 함입니다.

## 6) 카메라 2대 vs 단안

**StereoBench (CVPRW 2026) [원문]:**
- **불리한 점:** 실제 데이터에서는 보정 오차 때문에 스테레오 정확도가 떨어집니다. FoundationStereo의 AbsRel이 합성 데이터 7%에서 실제 데이터 22%로 나빠집니다.
- **유리한 점:** 실제 축척은 스테레오가 더 정확하고, 영상 전체에서 축척이 일정하게 유지되는 것도 스테레오·다시점뿐이었습니다. 축척 비율(1이 완벽) 기준 CREStereo 1.10, UniDepthV2 1.30입니다.
- **시사점:** 카메라 보정을 정밀하게 하는 게 전제 조건입니다.

**오차 공식:** 깊이 오차 ≈ Z² × (시차 오차) / (초점거리 × 기준선 길이). 출처는 Gallup et al., CVPR 2008 [원문].
- 제 계산 예시: Z=1.5m, f=1000px, 기준선 0.3m, 시차 오차 1px이면 약 7.5mm입니다. 2cm 구분이 가능한 수준입니다.
- 단, 이 공식은 두 카메라를 나란히 둔 경우 기준입니다. 넓은 각도로 둔 CCTV 2대는 삼각측량 오차로 따로 봐야 합니다.

**손 키포인트 삼각측량:** MediaPipe를 카메라 2대로 삼각측량하면 단안 3D보다 유의하게 정확했고, 두 카메라 사이 각도는 약 90°가 최적이었습니다 (PMC11644880) [원문].

**쓸 만한 오픈소스:**

| 도구 | 라이선스 | 비고 |
|---|---|---|
| FoundationStereo (★2,957) | **비상업용만** (NVIDIA, LICENSE 3.3조) [원문] | GPU 기준 무거움 |
| Fast-FoundationStereo (★1,545, CVPR 2026) | GitHub는 비상업용만. NGC 배포판은 상업 사용 가능 [원문] | 14.6M. CPU 속도 확인 못 함 |
| CREStereo ONNX (ibaiGorordo) | 코드 MIT, 모델 Apache | onnxruntime CPU 실행 가능 [원문] |
| RAFT-Stereo | MIT | 빠른 경로는 CUDA 필요 |
| IGEV++ | MIT | |
| aniposelib / Anipose (★470) | BSD-2 (문서 표기 불일치, 확인 필요) | 다시점 삼각측량 |
| Pose2Sim (★851) | BSD-3 | 손 지원은 아직 할 일 목록에 있음 |
| stereohand | MIT | ChArUco 보정 + MediaPipe + 삼각측량 예제 |

## 다시 시도해 볼 가치가 있는 것 상위 3개

1. **카메라 2대 + 손 키포인트 삼각측량 + 롤을 3D로 미리 측정 (가장 추천)**
   - **방법:** OpenCV `stereoCalibrate`와 ChArUco 보드로 두 카메라를 보정합니다. 손 키포인트를 2D로 찾고, `triangulatePoints`나 aniposelib으로 3D 좌표를 구합니다. 롤 축과 반지름은 고정이니 설치할 때 한 번만 3D로 측정해 둡니다. 판정은 "손끝에서 롤 축까지 거리 − 반지름 < 임계값"으로 합니다.
   - **이유:**
     - 이론 오차가 mm~1cm대로 2cm 구분이 가능한 범위입니다.
     - 깊이 신경망이 필요 없어 CPU에서도 빠릅니다.
     - StereoBench에서도 실제 축척은 스테레오·다시점이 우세했습니다.
   - **조건:** 정밀한 보정이 전제입니다. 시연 때 두 카메라 사이 각도는 60~90°로 두는 게 좋습니다.
   - **제 판단:** 천장 카메라처럼 사람이 작게 잡히면 MediaPipe 손 검출이 약할 수 있습니다. 손목·손끝 키포인트를 쓰는 RTMPose 계열도 함께 시험해 보세요.

2. **CREStereo ONNX를 손-롤 주변 영역만 잘라서 CPU로 실행**
   - **이유:** 키포인트가 불안정할 때 손 영역의 전체 깊이맵으로 보완할 수 있습니다. 라이선스도 MIT/Apache라 깨끗합니다.
   - **조건:** 작은 영역만 처리하면 CPU 부담이 줄 것으로 예상하지만, CPU 속도 수치는 확인 못 했으니 직접 재 봐야 합니다.
   - **참고:** FoundationStereo 계열이 더 정확할 수 있지만 GitHub판은 비상업용만 허용됩니다.

3. **단안을 유지해야 한다면 Video Depth Anything-Small을 OpenVINO INT8로 돌리고, 롤 크기로 축척 보정**
   - **용도:** 실제 공장 CCTV가 1대뿐인 경우의 대안입니다.
   - **이유:**
     - Apache-2.0 경량 모델 중 프레임 간 흔들림이 가장 적습니다(TAE 0.703, DA V2-L은 1.140).
     - OpenVINO INT8로 CPU 속도를 개선할 여지가 있습니다.
     - 미터 값을 그대로 믿지 말고, 크기를 아는 롤로 축척을 맞추는 방식(KineDepth 아이디어)을 씁니다. 그다음 손과 롤 표면의 상대적 앞뒤 관계를 여러 프레임에 걸쳐 판정합니다.
   - **한계:** 근거상 cm 단위 신뢰성은 기대하기 어렵습니다. 보조 신호로 쓰는 게 맞습니다.
   - **비교 대상:** MoGe-2 ViT-S(35M, MIT, ONNX 있음)도 같이 시험해 볼 만합니다.

손-롤 접촉에 단안 깊이를 쓴 선행 논문은 찾지 못했습니다. 그래서 1번을 주 방식으로 하고, 단안은 보조로 두는 구성을 권합니다.
