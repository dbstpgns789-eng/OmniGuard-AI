# Colab 실행

[Colab CLI](https://github.com/googlecolab/google-colab-cli)를 설치·인증한 WSL / Linux / macOS에서 실행합니다.
아래 명령의 기준 위치는 **프로젝트 루트**입니다. 입력은 `data/frames.zip`(5개 장면, 374장)을 사용합니다.
노트북은 이를 `/content/data`에 풀며, 결과 CSV의 `frame`은 `frames/<장면>/<시각>.jpg` 형식입니다(vlm 실험과 동일).

## imajev-4b

모델: `mohit67890/imajev-4b`. **24GB 이상 GPU와 고메모리 RAM을 권장합니다.**

```bash
colab new -s omniguard-imajev --gpu A100 --high-mem
colab upload -s omniguard-imajev data/frames.zip /content/frames.zip
colab exec -s omniguard-imajev -f notebooks/OmniGuard_Imajev_Colab.ipynb --timeout 7200
colab download -s omniguard-imajev /content/omniguard_imajev_results.csv results/omniguard_imajev_results.csv
mv notebooks/OmniGuard_Imajev_Colab_output.ipynb results/
colab stop -s omniguard-imajev
```

`_output.ipynb`는 CLI가 `notebooks/`에 생성하며, `mv`로 `results/`에 보관합니다. `download`와 `mv`는 기존 결과를 덮어씁니다.
현재 실행 코드는 `notebooks/`, 마지막 실행의 코드·출력은 `results/OmniGuard_Imajev_Colab_output.ipynb`에서 확인합니다.
실행이 실패해도 `colab stop -s omniguard-imajev`로 세션을 종료하세요.

## 설정과 결과

- `MAX_FRAMES`: 기본 `None`은 전체, `3`은 소량 테스트.
- `FAST`: 기본 `True`. 최적화 전 비교는 `False`로 실행합니다.
- `elapsed_ms`: 프레임별 전체 처리 시간. 설치·모델 로딩·워밍업은 제외합니다.
- 판단 보류는 `abstained=True`, `stage` 빈칸으로 기록합니다. 처리 실패는 `error`로 구분합니다.
- `confidence`의 정의·보정 여부는 확인하지 못했습니다. 정확도로 해석하지 않습니다.

로그: `/content/imajev_server.log`.
