# OmniGuard

산업 현장 프레임을 **Normal / Caution / Stop**으로 분류하는 Colab 실험입니다.

[실행 안내](docs/Run_Colab.md) · [입력 데이터](data/frames.zip)

| 모델 | 실행 노트북 | 실험 요약 | 결과 CSV |
|---|---|---|---|
| imajev-4b | [노트북](notebooks/OmniGuard_Imajev_Colab.ipynb) | [요약](results/OmniGuard_ImaJEV_Experiment.md) | [CSV](results/omniguard_imajev_results.csv) |

- `notebooks/`: 실행용 노트북
- `data/`: 입력 ZIP `frames.zip`(5개 장면, 374장)
- `results/`: 결과 CSV, 실행 결과 노트북, 보고서, `images/`
- `docs/`: 실행 안내

원본 이미지는 `data/frames.zip`에만 보관합니다. 결과 CSV의 `frame`은 `frames/<장면>/<시각>.jpg` 형식입니다.
Basal 모델 실험과 이전 입력 `data.zip`(371장)의 결과는 커밋 `b466b10`에 있습니다.
