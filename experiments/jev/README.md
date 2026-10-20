# OmniGuard

산업 현장 프레임을 **Normal / Caution / Stop**으로 분류하는 Colab 실험입니다.

[실행 안내](docs/Run_Colab.md) · [입력 데이터](data/data.zip)

| 모델 | 실행 노트북 | 실험 요약 | 결과 CSV |
|---|---|---|---|
| Basal | [노트북](notebooks/OmniGuard_Basal_Colab.ipynb) | [요약](results/basal/OmniGuard_Basal_Experiment.md) | [CSV](results/basal/omniguard_results.csv) |
| imajev-4b | [노트북](notebooks/OmniGuard_Imajev_Colab.ipynb) | [요약](results/imajev/OmniGuard_ImaJEV_Experiment.md) | [CSV](results/imajev/omniguard_imajev_results.csv) |

- `notebooks/`: 실행용 노트북
- `data/`: 원본 이미지 371장과 Guide.txt가 담긴 ZIP
- `results/<모델>/`: 결과 CSV, 실행 결과 노트북, 보고서와 이미지
- `docs/`: 실행 안내

원본 이미지는 `data/data.zip`에만 보관합니다. 결과 CSV의 `frame`은 ZIP 내부 `data/` 기준 경로입니다.
