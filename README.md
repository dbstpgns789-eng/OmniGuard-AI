# OmniGuard AI

공장 CCTV로 돌아가는 롤러·회전축에 몸이 끼이기 전에 먼저 알아채고, 진짜 위험만 관제실에 알리는 AI 안전 관제입니다.
컴퓨터비전(ITE4052) 최종 팀 프로젝트 · 2026-10-22 성과 발표

- 기획서: [docs/PR-FAQ.md](docs/PR-FAQ.md) · 용어: [CONTEXT.md](CONTEXT.md) · 판단 기준(정답지 규칙): [docs/판단기준.md](docs/판단기준.md)
- 팀: 윤세훈 (PM), 이채우 · 팀활동 보고서: [1일 차](docs/TeamReport_20261015.MD) · [2일 차](docs/TeamReport_Day2.md) · [3일 차](docs/TeamReport_Day3.md)

## 프로젝트 성과

### 무엇을 만들었나

```
공장 CCTV ─▶ 사람·관절 17개 ─▶ 포즈 품질 필터 ─▶ 위험한계까지 거리 ─▶ 원칙 규칙 + 정지 유지 ─▶ 정상 / 주의 / 정지
            (RTMPose, 학습 없음)  (가짜 사람 제거)    (몸통 길이 단위)       (1차 판단, 놓치지 않는 쪽)        │
                                                                                                       ▼ 정지일 때만
                                                                         VLM 2차 확인 (Qwen) ─▶ 관제실 알림 ─▶ 사후 리포트
                                                                         (괜히 멈춘 장면을 거름)
```

- 안전관리자가 CCTV 화면에 위험한 회전체(**위험한계**)를 한 번 그리면 설치가 끝납니다.
- 사전학습 모델만 씁니다. 회전체 끼임 사고 영상은 공개된 것이 거의 없어서(조사 결과 닿기 직전 장면은 1건), 새로 학습하지 않았습니다.
- 정답은 사람이 0.2초마다 적은 **판단표**입니다. "사고를 막으려면 지금 무엇을 해야 하나"를 정상·주의·정지로 적었습니다.

### 결과: 처음 보는 사고 10건 (판단기를 고정한 뒤 한 번만 채점)

| 방법 | 정지율 | 1분당 헛정지 | 평소 멈춘 시간 | 근거 |
|---|---|---|---|---|
| 기준선: 위험한계 근처면 무조건 정지 + 정지 유지 | 10/10 | - | 75% | [3일 차 보고서](docs/TeamReport_Day3.md) |
| 원칙 규칙 J13 | 5/10 | 6.7 | 54% | [실험 21](experiments/21_first_test) |
| J13 + 정지 유지 | 9/10 | 7.6 | 67% | [실험 22](experiments/22_combine) |
| **최종 판단기: 포즈 품질 필터 + J13 + 정지 유지** | **8/10** | **5.7** | **64%** | [실험 28](experiments/28_pose_filter) |
| 시간 모델(그래디언트 부스팅) + 정지 유지 | 7/10 | 4.8 | 62% | [실험 24](experiments/24_temporal) |
| MotionBERT (사고 9건으로 학습) | 6/10 | 6.7 | 62% | [실험 25](experiments/25_motionbert) |
| MotionBERT + 약한 라벨로 늘린 사고 27건 | **10/10** | 8.6 | 67% | [실험 26](experiments/26_more_data) |
| 이상 탐지 (정상 동작만 학습) | 3/10 | 0.95 | 2% | [실험 26](experiments/26_more_data) |

- **정지율:** 닿기 0.5초 전부터 닿을 때까지 정지가 계속 켜져 있었던 사고의 수입니다.
- **헛정지:** 판단표가 "정상"인 구간에서 정지가 켜진 횟수와 시간입니다. 정지율은 근처에서 무조건 멈추기만 해도 10/10이 나오므로, 실제 성능은 "정지율을 지키면서 평소 멈춘 시간(75%)을 얼마나 줄이는가"로 봅니다.

**VLM 2차 확인** ([실험 29](experiments/29_chaewoo_eval), [after_decisions](experiments/after_decisions)): 1차 판단기가 정지를 건 장면만 VLM이 safe/stop으로 다시 판정했습니다. Qwen3.8-27B는 롤 앞에서 일하다 겹쳐 보여 생긴 헛정지 53장면 중 30장면을 걸러 냈고, 진짜 정지 20장면은 하나도 풀지 않았습니다.

**처리 시간:** 관절 모델은 가벼운 버전에 OpenVINO와 탐지 간격을 적용해 176ms에서 17ms로 줄였습니다([실험 07](experiments/07_overnight)). 다만 야간 작업자를 찾으려면 큰 버전이 필요하고, 큰 버전은 CPU에서 프레임당 약 0.9초가 걸립니다. VLM은 imajev-4b가 0.13초, Qwen이 0.87초(A100)입니다.

### 알게 된 것

1. **규칙은 다듬을수록 처음 보는 사고에서 떨어졌습니다.** 개발 데이터에서 헛정지를 9번에서 1번으로 줄인 규칙 4는 시험에서 10/20, 그리드 서치 336가지 중 가장 조용한 조합은 2/20에 그쳤습니다. 가장 단순한 원칙 규칙이 13/20으로 가장 잘 버텼습니다([실험 21](experiments/21_first_test)).
2. **병목은 모델이 아니라 데이터 양이었습니다.** MotionBERT는 사고 9건으로는 과적합했지만(교차 검증 10/10, 시험 6/10), 약한 라벨로 사고를 27건으로 늘리자 하나도 놓치지 않으면서 평소 멈춘 시간을 기준선보다 줄였습니다.
3. **헛정지의 원인은 카메라 한 대의 2D 겹침입니다.** 화면에서는 롤 앞에 서 있는 사람과 롤에 손을 넣는 사람이 똑같이 겹쳐 보입니다. 발 위치, 자세, 팔 길이, 사람 윤곽, 단안 깊이로 풀어 보았지만 모두 실패했습니다([실험 14](experiments/14_foot_gate)~[23](experiments/23_depth_order)). 그래서 1차 판단은 놓치지 않는 쪽을 맡기고, 헛정지는 VLM 2차 확인과 카메라 추가로 줄이는 구조를 택했습니다.
4. **정답을 누가 정하느냐가 결론을 바꿉니다.** Claude가 정한 "접촉 시각" 정답으로 낸 결론은 사람 판단표로 다시 채점하자 뒤집혔습니다([실험 07·08](experiments/08_field_demo)). 판단자 2명의 정답지는 안내문 한 줄 때문에 한 칸씩 어긋났고, 이를 찾아 보정했습니다([실험 15](experiments/15_two_raters)).

## 팀과 역할

| 이름 | 역할 | 담당 |
|---|---|---|
| **윤세훈** | PM, 레포 관리, 판단자 A | 기획(PR/FAQ), 판단 기준과 정답지, 위험한계 그리기, 데이터 수집, 규칙 판단기·학습 모델 실험(01~28), 시험 설계와 채점, 데모 영상, 3D 방향 탐색, 발표 대본 |
| **이채우** | VLM 담당, 판단자 B | VLM 판정 실험(imajev-4b, Qwen3.8-27B), VLM 2차 확인 실험, 처리 시간 측정, 정답지(판단자 B), 팀활동 보고서, 발표자료 |
| 공통 | | 정답지 작성, 결과 영상 검토, 실험 결과 정리, 발표 준비 |

### 윤세훈 담당 작업

| 분야 | 한 일 | 결과 | 위치 |
|---|---|---|---|
| 문제 정의 | 범위를 회전 롤 맞물림부 끼임으로 좁히고, 정상·주의·정지 판단 기준을 정함 | 판단 기준 문서, 시연 시나리오 | [PR-FAQ](docs/PR-FAQ.md), [판단기준](docs/판단기준.md) |
| 데이터 | 공개 데이터셋 조사, 국가별 사고 영상 수집, 사고 19건 0.2초 판단표 작성, 영상마다 위험한계 그리기 | 개발용·맞춤용·시험용 분리, 결과를 보기 전에 분할을 커밋 | [videos](docs/videos/종합.md), [research](docs/research/종합.md), [실험 20](experiments/20_frozen) |
| 모델 선정 | RF-DETR·YOLOX, RTMPose 크기별, MediaPipe Hands, Depth Anything V2, YOLO11n-seg 비교 | 사람 탐지 98/100 프레임, 처리 시간 176ms → 17ms | [실험 01](experiments/01_zero_shot_check), [07](experiments/07_overnight), [결정근거](docs/결정근거.md) |
| 규칙 판단기 | 장면마다 "왜 위험/안전한가"를 규칙으로 옮김, 그리드 서치 336가지, 헛정지 원인 관절 분석 | 개발 데이터 헛정지 9번 → 1번 (사고 4/4 유지) | [실험 09](experiments/09_red_fit), [18](experiments/18_occlusion_gate), [19](experiments/19_remove_false) |
| 검증 | 판단기 3개를 고정하고 처음 보는 사고에 한 번만 채점, 판단자 2명 일치도 | 과적합 확인(규칙 4: 개발 4/4 → 시험 10/20) | [실험 15](experiments/15_two_raters), [21](experiments/21_first_test) |
| 개선 | 정지 유지, 사람 윤곽 보조, 단안 깊이 앞뒤 판정, 포즈 품질 필터 | 최종 판단기 8/10, 헛정지 7.6 → 5.7/분 | [실험 22](experiments/22_combine), [23](experiments/23_depth_order), [28](experiments/28_pose_filter) |
| 학습 모델 | 그래디언트 부스팅, 로지스틱 회귀, MotionBERT, 약한 라벨로 데이터 늘리기, 이상 탐지 | MotionBERT + 사고 27건 10/10, 평소 정지 67% | [실험 24](experiments/24_temporal), [25](experiments/25_motionbert), [26](experiments/26_more_data) |
| 통합·발표 | VLM 결과를 판단표로 채점해 1차·2차 구조로 묶음, 데모 영상 렌더링 | 1차 판단기 + 2차 VLM 구조 | [실험 27](experiments/27_demo), [29](experiments/29_chaewoo_eval), [발표용 정리](docs/실험정리_발표용.md) |

### 이채우 담당 작업

| 분야 | 한 일 | 결과 | 위치 |
|---|---|---|---|
| VLM 1차 판정 | 사고 영상 5개 374장을 정상·주의·정지로 판정 | imajev 장면당 0.13초·판단 보류 25%, Qwen 0.87초·실제 정지의 79%를 잡음 | [jev](experiments/jev), [vlm](experiments/vlm) |
| VLM 2차 확인 | 1차 정지 장면 373장을 safe/stop으로 다시 판정 | Qwen이 롤 앞 헛정지 53장면 중 30장면을 거르고 진짜 정지는 모두 유지 | [after_decisions](experiments/after_decisions) |
| 정답지 | 판단자 B로 사고 영상 판단표 작성 | 두 판단자 일치도 측정에 사용 | [실험 15](experiments/15_two_raters) |

## 사용 가이드

### 레포 구조

```
OmniGuard-AI/
├─ README.md            이 문서
├─ requirements.txt     판단기·채점·데모에 필요한 패키지
├─ CONTEXT.md           용어집
├─ tools/draw_zone.py   위험한계·바닥 앞선 그리기 도구
├─ docs/                기획서, 판단 기준, 팀활동 보고서, 조사, 영상 수집 기록, 발표용 실험 정리
└─ experiments/
   ├─ 01~29_*/          실험마다 README(질문·설계·결과)와 코드, result.json
   ├─ jev/, vlm/        VLM 1차 판정 Colab 노트북과 입력 frames.zip, 결과
   └─ after_decisions/  VLM 2차 확인 Colab 노트북과 결과
```

실험 폴더마다 `README.md`가 있고, 설계와 채점 기준은 결과를 보기 전에 커밋했습니다. 결과만 보려면 각 폴더의 `README.md`와 `result.json`을 열면 됩니다.

### 1. 바로 돌려 보기: VLM 판정 (Colab, 데이터 포함)

입력 이미지 `frames.zip`(374장)이 레포에 들어 있어서, 클론만 하면 바로 실행할 수 있습니다. GPU는 imajev-4b가 A100, Qwen3.8-27B가 A100 High-RAM입니다.

1. 레포를 클론합니다.
   ```bash
   git clone https://github.com/dbstpgns789-eng/OmniGuard-AI.git
   cd OmniGuard-AI/experiments/vlm      # imajev는 experiments/jev
   ```
2. Colab에서 `notebooks/OmniGuard_Qwen38_Colab.ipynb`를 열고, `data/frames.zip`을 `/content/frames.zip`으로 올린 뒤 전체 실행합니다.
3. 결과 CSV(`omniguard_qwen38_results.csv`)를 받아 `results/qwen38/`에 둡니다. 열은 `frame, stage, elapsed_ms, ...`입니다.

터미널에서 Colab CLI로 실행하는 명령과 업로드가 끊길 때의 대처는 [vlm/docs/Run_Colab.md](experiments/vlm/docs/Run_Colab.md), [jev/docs/Run_Colab.md](experiments/jev/docs/Run_Colab.md)에 있습니다. 2차 확인(safe/stop)은 [after_decisions/docs/Run_Colab.md](experiments/after_decisions/docs/Run_Colab.md)를 따르며, 입력 `stop_frames.zip`은 레포에 없으므로 팀에 요청해야 합니다.

### 2. 판단기 돌리기 (CPU, 영상 데이터 필요)

사고 영상과 판단표는 레포에 올리지 않았습니다. 같은 구조로 영상을 준비하면 최종 판단기를 재현할 수 있습니다.

**설치** (Python 3.13, GPU 없이 확인)

```bash
pip install -r requirements.txt
```

RTMPose 모델 가중치는 `rtmlib`이 처음 실행할 때 자동으로 내려받습니다.

**데이터 폴더 준비**

코드는 데이터 폴더를 `C:/Users/User/Desktop/OmniGuard-data/`로 가정합니다. 다른 위치를 쓰려면 각 스크립트 맨 위의 `DATA = ...` 한 줄을 바꿉니다(`experiments/21_first_test/prepare.py`, `evaluate.py` 등).

```
OmniGuard-data/
├─ collected/A/A04.mp4 ...           원본 영상
└─ 세훈_작업/판단표_사고19/<영상 이름>/
   ├─ frames/0433.0s.jpg ...          0.2초 간격 프레임
   ├─ 위험한계.json                    {"frame": 기준 프레임, "outlines": [[x, y], ...]}  좌표는 0~1 비율
   ├─ 앞선.json                        바닥 앞선: 위험한 롤 아래, 사람이 서는 쪽 바닥에 설비 방향으로 그은 선
   └─ 판단표.csv                       시각(초), 파일, 단계(정상/주의/정지), 회전체에 닿음(Y/N/모름), 메모
```

- **위험한계:** 롤 양 끝 원판의 윤곽입니다. 몸 어디든 들어가면 위험한 영역을 뜻합니다. [그리기 도구](tools/draw_zone.py)로 그립니다(Windows, 맑은 고딕 글꼴 사용).
  ```bash
  python tools/draw_zone.py OmniGuard-data/세훈_작업/판단표_사고19/A04_carpet
  ```
  Z 키를 누른 뒤 원판 테두리를 왼쪽 클릭으로 10점 이상 찍고, 더블클릭으로 원판 하나를 완성합니다. 롤이 여러 개면 N 키로 다음 롤로 넘어갑니다. F 키를 누르면 바닥 앞선(클릭 2번)을 그립니다. 오른쪽 클릭은 되돌리기, ←/→ 키는 화면 바꾸기, ESC는 끝입니다. 클릭할 때마다 `위험한계.json`·`앞선.json`·`그린화면.jpg`에 자동 저장됩니다.
- **판단표:** 판단이 바뀌는 줄만 적으면 됩니다. 빈칸은 바로 위 판단을 이어 받습니다. 적는 기준은 [docs/판단기준.md](docs/판단기준.md)입니다.

**실행 순서**

```bash
# 1) 관절 추출 + 화면 보정: 영상마다 <영상>_pose.npz 생성 (CPU에서 프레임당 약 0.9초)
python experiments/21_first_test/prepare.py

# 2) 최종 판단기(포즈 품질 필터 + J13 + 정지 유지) 채점: 정지율, 1분당 헛정지, 평소 멈춘 시간
#    개발 데이터 확인용 포즈(exp20_stab/)도 읽으므로, 먼저 experiments/20_frozen/stabilize_dev.py를 실행해 둡니다
python experiments/28_pose_filter/run.py        # 결과: experiments/28_pose_filter/result.json

# 3) 데모 영상: 위험한계, 관절, 판단기 상태, 판단기·정답 시간 막대를 덧그림
python experiments/27_demo/render.py video A04_carpet
python experiments/27_demo/render.py still A04_carpet 441.2    # 한 장면만
```

판단기 정의는 [experiments/20_frozen/judges.py](experiments/20_frozen/judges.py)에 있고, 시험 전에 고정했습니다. 거리 단위는 그 사람의 몸통 길이이며, 위험한계 안쪽이 음수입니다.

| 판단기 | 정지 조건 | 정지 해제 |
|---|---|---|
| J13 원칙 | 손 또는 어깨·팔꿈치가 위험한계 경계에 닿음 | 0.3초 유지 |
| + 정지 유지 | 한 번 정지하면 사람이 위험한계 근처(몸통 0.5배 안)에 있거나 포즈를 놓친 동안 계속 정지 | 근처를 벗어난 상태가 1초 이어지면 |
| + 포즈 품질 필터 | 입력 단계에서 팔다리 한 마디가 몸통의 1.5배보다 긴 관절, 보이는 관절 6개 미만, 평균 신뢰도 0.4 미만인 사람을 버림 | |

### 다른 실험 재현

- 각 실험 폴더의 `README.md` 맨 위나 스크립트 첫 줄 주석에 사용법이 있습니다. 예: `python experiments/22_combine/combine.py tune`
- 실험 25·26(MotionBERT)은 [walterzhu/MotionBERT](https://github.com/Walter0807/MotionBERT) 저장소를 `OmniGuard-data/models/MotionBERT/`에 받고, NTU60 행동 인식 미세조정 가중치를 `checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub/best_epoch.bin`에 두어야 합니다.

## 한계와 다음 단계

- 카메라 한 대로는 손끝과 롤 표면의 거리를 알 수 없어서, 잘 잡는 방법은 모두 평소에도 64% 이상 멈췄습니다.
- 다음 단계는 세 가지입니다. ① 지금 있는 CCTV에 VLM 2차 확인을 붙여 헛정지 알림을 절반 이하로 줄입니다. ② 맞물림부에 카메라를 한두 대 더 두고 손 관절을 삼각측량해, 손끝에서 롤까지의 거리를 cm 단위로 잽니다. ③ 실제 공장에서 설비 정지 신호(PLC)와 연결합니다.
- 자세한 근거와 로드맵: [docs/실험정리_발표용.md](docs/실험정리_발표용.md), 시도별 기록: [docs/시도기록.md](docs/시도기록.md)

## 데이터 출처

- [Voxel51/Safe_and_Unsafe_Behaviours](https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours) (CC BY 4.0, Önal & Dandıl, Data in Brief 2024): 프레스 평소 작업(헛정지 시험). 실행할 때 내려받습니다.
- iSafetyBench: 빨간 상의 사고(개발용)
- 웹 사고 영상과 유튜브 사고 채널 화면 녹화: 시험용 사고 19건. 목록은 [docs/videos/manifest.csv](docs/videos/manifest.csv)이며, 영상 원본은 레포에 올리지 않습니다.
