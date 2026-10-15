# OmniGuard AI

기계를 켠 채 손을 넣는 순간만 골라 멈추는 카메라.
컴퓨터비전(ITE4052) 최종 팀 프로젝트 · 2026-10-22 PoC 시연

- 기획서: [docs/PR-FAQ.md](docs/PR-FAQ.md)
- 팀: 윤세훈 (PM), 팀원 1명 (추가 예정)

## 무엇을 만드나

설비 옆 카메라가 사람을 찾고(RF-DETR), 관절 좌표를 뽑고(RTMPose), 아래 세 조건이 모두 겹칠 때만 경고 → 감속 → 정지합니다.

1. 설비가 가동 중
2. 정비 동작(숙임, 팔 뻗음, 머무름)
3. 손목이 위험 구역 안

정지 판단은 규칙 엔진이 하고, 사건 이후의 조사와 보고는 에이전트가 합니다. 자세한 구조는 PR/FAQ 부록 A.

## 실험

| 폴더 | 질문 | 결과 요약 |
|---|---|---|
| [experiments/01_zero_shot_check](experiments/01_zero_shot_check) | 학습 없이 공개 모델을 공장 CCTV에 돌리면 무엇이 되나 | 사람 탐지 98/100 프레임, 포즈 특징 분류기로 개입/통행 구별 73% |
| [experiments/02_zone_rule](experiments/02_zone_rule) | 학습 없이 "손목이 구역 안" 규칙만으로 개입/통행을 구별할 수 있나 | 최고 66.7%. 클립 라벨과 다른 작업자, 2D 깊이 겹침이 원인 ([README](experiments/02_zone_rule/README.md)) |

## 데이터

- [Voxel51/Safe_and_Unsafe_Behaviours](https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours) (CC BY 4.0, Önal & Dandıl, Data in Brief 2024). 실험 코드가 실행할 때 내려받으며, 레포에는 포함하지 않습니다.
- 사고 참고 영상 캡처와 연기 촬영 원본은 레포에 올리지 않습니다.

## 실행

```bash
pip install rfdetr rtmlib onnxruntime huggingface_hub opencv-python-headless scikit-learn
cd experiments/02_zone_rule
python step3_rule.py
```
