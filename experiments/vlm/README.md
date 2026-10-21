# OmniGuard VLM

범용 비전·언어 모델(VLM)로 산업 현장 프레임을 **Normal / Caution / Stop**으로 분류하는 Colab 실험입니다. 추가 학습 없이 프레임 한 장씩 판정합니다.

[실행 안내](docs/Run_Colab.md) · [입력 데이터](data/frames.zip)

| 모델 | 실행 노트북 | 실험 요약 | 결과 CSV | 실행 설정 |
|---|---|---|---|---|
| Qwen3.8-27B | [노트북](notebooks/OmniGuard_Qwen38_Colab.ipynb) | [요약](results/qwen38/OmniGuard_Qwen38_Experiment.md) | [CSV](results/qwen38/omniguard_qwen38_results.csv) | [run_config](results/qwen38/run_config.json) |

- `notebooks/`: 실행용 노트북
- `data/`: 입력 ZIP `frames.zip`(5개 장면, 374장). jev 실험과 같은 파일입니다.
- `results/<모델>/`: 결과 CSV, run_config.json, 실행 결과 노트북, 보고서와 `images/`
- `docs/`: 실행 안내

원본 이미지는 `data/frames.zip`에만 보관합니다. 결과 CSV의 `frame`은 `frames/<장면>/<시각>.jpg` 형식입니다.
`results/qwen38/`의 결과는 374장 입력으로 실행한 것입니다. 검은 제목 화면 4장을 포함한 직전 378장 실행의 기록은 `OmniGuard_Qwen38_Colab_output_378.ipynb`에 남겨 두었습니다.
