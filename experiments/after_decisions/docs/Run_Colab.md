# Colab 실행: after_decisions

[Colab CLI](https://github.com/googlecolab/google-colab-cli)를 설치·인증한 WSL / Linux / macOS에서 실행합니다.
아래 명령의 기준 위치는 `experiments/after_decisions/`입니다. 실험 설계는 [README](../README.md)를 보세요.

## 1. 입력 ZIP

두 모델이 같은 `data/stop_frames.zip`(373장, 약 48MB)을 씁니다. 이미 만들어 두었으므로 그대로 올리면 됩니다.
이미지를 바꿨을 때만 아래 명령으로 다시 만듭니다.

```bash
(cd data && rm -f stop_frames.zip && python3 -m zipfile -c stop_frames.zip stop_frames/)
```

## 2. imajev-4b

**24GB 이상 GPU와 고메모리 RAM을 권장합니다.**

```bash
mkdir -p jev/results
colab new -s ad-imajev --gpu A100 --high-mem
colab upload -s ad-imajev data/stop_frames.zip /content/stop_frames.zip
colab exec -s ad-imajev -f jev/notebooks/OmniGuard_Imajev_SafeStop_Colab.ipynb --timeout 7200
colab download -s ad-imajev /content/after_decisions_imajev_results.csv jev/results/after_decisions_imajev_results.csv
mv jev/notebooks/OmniGuard_Imajev_SafeStop_Colab_output.ipynb jev/results/
colab stop -s ad-imajev
```

서버 로그: `/content/imajev_server.log`.

## 3. Qwen3.8-27B

BF16 가중치만 약 56GB입니다. A100 80GB면 BF16 그대로, 40GB면 4-bit NF4로 실행하며 어느 쪽인지 `run_config.json`에 기록됩니다.

```bash
mkdir -p vlm/results
colab new -s ad-qwen38 --gpu A100 --high-mem
colab upload -s ad-qwen38 data/stop_frames.zip /content/stop_frames.zip

# HF 토큰(선택). 노트북이 읽은 뒤 런타임에서 지웁니다.
read -rs -p "HF token: " HF_TOKEN; echo
(umask 077; printf '%s' "$HF_TOKEN" > /tmp/hf_token)
colab upload -s ad-qwen38 /tmp/hf_token /content/hf_token
rm -f /tmp/hf_token; unset HF_TOKEN

colab exec -s ad-qwen38 -f vlm/notebooks/OmniGuard_Qwen38_SafeStop_Colab.ipynb --timeout 7200
colab download -s ad-qwen38 /content/after_decisions_qwen38_results.csv vlm/results/after_decisions_qwen38_results.csv
colab download -s ad-qwen38 /content/run_config.json vlm/results/run_config.json
mv vlm/notebooks/OmniGuard_Qwen38_SafeStop_Colab_output.ipynb vlm/results/
colab stop -s ad-qwen38
```

토큰은 `colab exec --env HF_TOKEN=...`으로 넘기지 마세요. CLI가 셀 코드 앞에 토큰을 붙여 `~/.config/colab-cli/history/<세션>.jsonl`에 평문으로 남깁니다.
첫 실행은 모델 다운로드(약 56GB) 시간이 포함됩니다.

## 공통

- `MAX_FRAMES`: 노트북 첫 코드 셀. 기본 `None`은 전체, `3`은 소량 실행입니다.
- `_output.ipynb`는 CLI가 `notebooks/`에 만들며, `mv`로 `results/`에 보관합니다. `download`와 `mv`는 기존 결과를 덮어씁니다.
- 실행이 실패해도 `colab stop -s <세션명>`으로 세션을 종료하세요.
- 업로드가 `SSLEOFError`로 끊기면 `experiments/vlm/docs/Run_Colab.md`(`exp/vlm`)의 조각 업로드 방법을 쓰세요. 현재 ZIP은 약 48MB로, 비슷한 크기(47.6MB)는 한 번에 올라갔습니다.
