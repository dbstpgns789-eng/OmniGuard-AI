# Colab 실행

[Colab CLI](https://github.com/googlecolab/google-colab-cli)를 설치·인증한 WSL / Linux / macOS에서 실행합니다.
아래 명령의 기준 위치는 **프로젝트 루트**입니다. 입력은 `data/data.zip`을 사용합니다.

## Basal

모델: `Remek/basal-omni-1.0-ultra`. **80GB급 GPU가 필요합니다.**

```bash
mkdir -p results/basal
colab new -s omniguard --gpu H100 --high-mem
colab upload -s omniguard data/data.zip /content/data.zip
colab exec -s omniguard -f notebooks/OmniGuard_Basal_Colab.ipynb --timeout 7200
colab download -s omniguard /content/omniguard_results.csv results/basal/omniguard_results.csv
mv notebooks/OmniGuard_Basal_Colab_output.ipynb results/basal/
colab stop -s omniguard
```

## imajev-4b

모델: `mohit67890/imajev-4b`. **24GB 이상 GPU와 고메모리 RAM을 권장합니다.**

```bash
mkdir -p results/imajev
colab new -s omniguard-imajev --gpu H100 --high-mem
colab upload -s omniguard-imajev data/data.zip /content/data.zip
colab exec -s omniguard-imajev -f notebooks/OmniGuard_Imajev_Colab.ipynb --timeout 7200
colab download -s omniguard-imajev /content/omniguard_imajev_results.csv results/imajev/omniguard_imajev_results.csv
mv notebooks/OmniGuard_Imajev_Colab_output.ipynb results/imajev/
colab stop -s omniguard-imajev
```

`_output.ipynb`는 CLI가 `notebooks/`에 생성하며, `mv`로 모델별 결과 폴더에 보관합니다.
현재 실행 코드는 `notebooks/`, 과거 코드·출력은 `results/<모델>/*_output.ipynb`에서 확인합니다.
현재 결과 CSV의 시간 열은 `elapsed_ms` 하나이며, 기존 실행 기록에는 `server_ms`도 남아 있습니다.
실행이 실패해도 해당 세션을 `colab stop -s <세션명>`으로 종료하세요.

## 설정과 결과

- `MAX_FRAMES`: 기본 `None`은 전체, `3`은 소량 테스트.
- Basal의 `DETAIL`: 기본 `high`. `default`로 바꾸면 이미지 토큰 수를 줄입니다.
- imajev의 `FAST`: 기본 `True`. 최적화 전 비교는 `False`로 실행합니다.
- `elapsed_ms`: 프레임별 전체 처리 시간. 설치·모델 로딩·워밍업은 제외합니다.
- imajev의 판단 보류는 `abstained=True`, `stage` 빈칸으로 기록합니다. 처리 실패는 `error`로 구분합니다.
- 두 모델의 `confidence`는 정의가 다릅니다. 같은 임계값으로 비교하지 않습니다.

로그: `/content/basal_server.log`, `/content/imajev_server.log`.
