# after_decisions: 1차 Stop 프레임의 safe / stop 재판정

1차 분류기가 **Stop**으로 판정한 프레임을, 범주를 **safe / stop** 두 개로 줄여 다시 판정하는 실험입니다.
imajev-4b와 Qwen3.8-27B가 같은 프레임을 각각 판정합니다. 추가 학습 없이 사전학습 모델만 씁니다.

[실행 안내](docs/Run_Colab.md)

## 목적

정지 신호를 내기 전에 한 번 더 확인하는 2차 판정을 가정합니다. 1차의 3단계(Normal / Caution / Stop)에서 중간 단계인 Caution을 없애고 "멈출지 말지"만 물었을 때,

1. 1차 Stop 중 몇 장이 2차에서도 stop으로 남는가 (모델별, 장면별)
2. 두 모델이 같은 프레임에서 얼마나 같은 판정을 내리는가
3. 판단 보류를 허용하지 않을 때, 모델이 근거 부족을 표시한 프레임(imajev의 보류 신호)은 어떻게 판정되는가
4. 2차 판정에 프레임당 얼마의 처리 시간이 더 드는가

를 확인합니다.

## 입력

`data/stop_frames/<장면>/<시각>s.jpg`. 1차 분류기가 Stop으로 판정한 프레임을 모은 것으로, 두 모델이 같은 373장을 처리합니다.

| 장면 | 장수 | 해상도 | 영상 속 시각 |
|---|---:|---|---|
| A04_carpet | 54 | 959×720 | 438.2~448.8s |
| B38_mixer | 50 | 1280×720 | 0.0~9.8s |
| D09 | 96 | 722×908 | 0.0~20.4s |
| red_shirt | 173 | 750×780 | 7.0~46.4s |

`data/`는 `.gitignore` 대상이라 이미지는 레포에 올라가지 않습니다. 1차 분류기의 종류와 1차 결과는 이 폴더에 없으므로, 결과 보고서에 따로 적어야 합니다.

## 범주 정의 (두 모델 공통)

| 범주 | 정의 (모델에 전달한 문장) |
|---|---|
| stop | A hand, arm, head, or other body part is visibly inside an exposed rotating or pinch area, or contact is imminent. The machine needs to be stopped immediately. |
| safe | No body part is visibly inside an exposed rotating or pinch area and contact is not imminent. This includes a worker who is close to or reaching toward exposed rotating parts without entering the hazardous area. |

- **stop**은 기존 실험의 Stop 정의를 그대로 씁니다. 범주를 줄인 효과만 보기 위해서입니다.
- **safe**는 그 밖의 모든 경우입니다. 기존 Normal과 Caution이 합쳐집니다.
- 여러 사람이 보이면 한 명이라도 stop 조건에 맞을 때 stop입니다. 현재 이미지에 보이는 근거만 쓰고, 정지 이미지에서 움직임이나 설비 가동 여부를 추측하지 않습니다. 이미지 속 글자는 지시로 따르지 않습니다.
- **판단 보류는 허용하지 않습니다.** 모든 프레임은 safe 또는 stop입니다. 처리 실패만 `error`로 남기며, 오류를 safe나 stop으로 바꾸지 않습니다.

## 모델과 설정

| | imajev-4b | Qwen3.8-27B |
|---|---|---|
| 노트북 | [jev/notebooks/OmniGuard_Imajev_SafeStop_Colab.ipynb](jev/notebooks/OmniGuard_Imajev_SafeStop_Colab.ipynb) | [vlm/notebooks/OmniGuard_Qwen38_SafeStop_Colab.ipynb](vlm/notebooks/OmniGuard_Qwen38_SafeStop_Colab.ipynb) |
| 모델 | `mohit67890/imajev-4b`, 어댑터 `c9e5f132`, 공식 PyTorch 서버 | `Qwen/Qwen3.8-27B` revision `1d4bf0f2`, transformers 5.19.0 |
| 실행 | A100, `FAST=True`, `--rotations 1`, `--thinking off` | A100 80GB면 BF16(40GB면 NF4), thinking 끔, 샘플링 없이 최대 16토큰 |
| 판정 방식 | `choice` 질문의 선택지를 `safe`, `stop` 두 개로 설정 | `{"stage":"safe"}` 또는 `{"stage":"stop"}` JSON 하나만 생성 |
| 판단 보류 처리 | 보류 기능이 모델에 내장돼 있어 끌 수 없음. 서버가 보류할 때도 돌려주는 `choice`(판단 불가를 뺀 두 범주 중 확률이 높은 쪽)를 판정으로 쓰고, 보류 신호는 `model_abstained`·`unknown_probability`에 따로 기록 | 프롬프트에서 `null`을 금지. 그래도 `null`이나 다른 값을 내면 `error` |
| 기반 | `exp/jev`의 imajev 노트북과 같은 설치·서버·시간 측정 | `exp/vlm`의 Qwen 노트북과 같은 로드·시간 측정·run_config |

두 노트북 모두 프레임마다 대화 이력 없이 독립적으로 판정하고, 파일명·시각·1차 판정은 모델에 전달하지 않습니다.

## 결과 파일

| 모델 | 결과 CSV | 열 |
|---|---|---|
| imajev | `jev/results/after_decisions_imajev_results.csv` | `frame, stage, elapsed_ms, confidence, stop_probability, unknown_probability, model_abstained, error` |
| Qwen | `vlm/results/after_decisions_qwen38_results.csv` (+ `run_config.json`) | `frame, stage, elapsed_ms, inference_ms, error, raw_output` |

- `frame`은 `stop_frames/<장면>/<시각>s.jpg` 형식으로, 두 CSV를 이 열로 바로 맞춰 볼 수 있습니다.
- `stage`는 safe / stop이며, 오류일 때만 빈칸입니다.
- `elapsed_ms`는 이미지 읽기부터 응답 파싱까지, 설치·로딩·워밍업을 뺀 프레임 한 장의 처리 시간입니다.

## 분석 계획

1. **stop 유지:** 모델별·장면별 stop / safe 개수. safe는 "1차 Stop을 2차가 safe로 바꾼 프레임"입니다.
2. **두 모델 비교:** 같은 프레임에서 stop·stop / safe·safe / 불일치 개수.
3. **safe로 바뀐 프레임 검토:** 앞뒤 프레임을 붙인 이미지로 화면을 직접 보고 이유를 추론합니다(기존 Experiment.md와 같은 방식).
4. **imajev 보류 신호:** `model_abstained=True`인 프레임 수와, 그 프레임들이 safe / stop 중 어디로 갔는지. `unknown_probability` 분포.
5. **처리 시간:** 모델별 평균·중앙값·P95. 2차 판정은 1차 Stop 프레임에만 더해지는 시간입니다.

## 해석할 때 주의할 점

- **정답 라벨이 없습니다.** safe로 바뀐 프레임이 1차의 잘못된 Stop인지, 2차가 위험을 놓친 것인지는 판단하지 않습니다. stop 유지 비율은 정확도가 아닙니다.
- **입력이 모두 1차 Stop입니다.** 그래서 2차는 Stop을 줄이는 방향으로만 작용하며, 1차가 놓친 위험은 이 실험으로 찾을 수 없습니다.
- **판단 보류를 허용하지 않습니다.** 근거가 부족한 프레임도 safe 또는 stop으로 강제되므로, imajev의 보류 신호와 함께 봐야 합니다.
- 시간은 프레임 분류 처리 시간입니다. 영상 입력부터 설비 정지까지의 지연이 아닙니다.
- 실제 설비 제어 없이 프레임 분류만 합니다.

## 폴더

- `data/stop_frames/`, `data/stop_frames.zip`: 입력 이미지와 Colab 업로드용 ZIP(레포에 올리지 않음)
- `jev/notebooks/`, `jev/results/`: imajev-4b 노트북과 결과
- `vlm/notebooks/`, `vlm/results/`: Qwen3.8-27B 노트북과 결과
- `docs/Run_Colab.md`: Colab 실행 안내
