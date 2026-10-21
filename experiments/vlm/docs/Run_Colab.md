# Colab 실행: Qwen3.8-27B

[Colab CLI](https://github.com/googlecolab/google-colab-cli)를 설치·인증한 WSL / Linux / macOS에서 실행합니다.
아래 명령의 기준 위치는 `experiments/vlm/`입니다. 입력은 `data/frames.zip`(5개 장면, 374장, 해상도 468×347~1221×597)입니다. jev 실험의 `frames.zip`과 같은 파일이며, 영상 사이의 검은 제목 화면은 뺐습니다.

모델: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (revision `1d4bf0f2`). 실행 환경은 **A100 (High-RAM)** 입니다.
BF16 가중치만 약 56GB이므로 노트북이 VRAM을 보고 정밀도를 고릅니다. A100 80GB면 BF16 그대로, A100 40GB면 4-bit NF4 양자화이며, 어느 쪽인지 run_config.json의 `gpu`·`precision`에 기록됩니다.

```bash
mkdir -p results/qwen38
colab new -s omniguard-qwen38 --gpu A100 --high-mem
colab upload -s omniguard-qwen38 data/frames.zip /content/frames.zip
colab exec -s omniguard-qwen38 -f notebooks/OmniGuard_Qwen38_Colab.ipynb --timeout 7200
colab download -s omniguard-qwen38 /content/omniguard_qwen38_results.csv results/qwen38/omniguard_qwen38_results.csv
colab download -s omniguard-qwen38 /content/run_config.json results/qwen38/run_config.json
mv notebooks/OmniGuard_Qwen38_Colab_output.ipynb results/qwen38/
colab stop -s omniguard-qwen38
```

위 `download`·`mv`는 `results/qwen38/`의 기존 결과를 덮어씁니다. 남겨 둘 결과가 있으면 먼저 다른 이름으로 옮기세요.

토큰은 `colab exec --env HF_TOKEN=...`으로 넘기지 마세요. CLI가 셀 코드 앞에 토큰을 붙여 `~/.config/colab-cli/history/<세션>.jsonl`에 평문으로 남깁니다.
실행이 실패해도 `colab stop -s omniguard-qwen38`으로 세션을 종료하세요. 첫 실행은 모델 다운로드(약 56GB) 시간이 포함됩니다.

### 업로드가 `SSLEOFError`로 끊길 때

Colab CLI(0.7.5)의 `upload`는 파일 전체를 base64 JSON 요청 하나로 보냅니다. 현재 `frames.zip`(47.6MB)은 그대로 올라가지만,
이전 버전(92.7MB)은 `SSLEOFError`로 끊겼습니다. 파일이 커서 끊기면 20MB 조각으로 올린 뒤 런타임에서 합칩니다.

```bash
mkdir -p /tmp/frames_parts && split -b 20M -d data/frames.zip /tmp/frames_parts/frames.zip.part
for f in /tmp/frames_parts/frames.zip.part*; do
  colab upload -s omniguard-qwen38 "$f" "/content/$(basename "$f")" || break
done
colab exec -s omniguard-qwen38 <<'EOF'
import glob, hashlib, os
parts = sorted(glob.glob("/content/frames.zip.part*"))
with open("/content/frames.zip", "wb") as out:
    for p in parts:
        out.write(open(p, "rb").read())
print(len(parts), os.path.getsize("/content/frames.zip"),
      hashlib.sha256(open("/content/frames.zip", "rb").read()).hexdigest())
for p in parts:
    os.remove(p)
EOF
sha256sum data/frames.zip   # 위 출력의 해시와 같아야 합니다
```

## 노트북 흐름

1. GPU·VRAM 확인 → 정밀도 선택(`QUANTIZATION="auto"`), ZIP 해제, 장면·시각 순 정렬
2. transformers 5.19.0, accelerate 1.15.0, flash-linear-attention 0.5.2 설치(NF4일 때 bitsandbytes 0.50.2)
3. 모델 1회 로드. 모든 가중치가 `cuda:0`에 있는지 확인(CPU offload 없음)
4. run_config.json 저장
5. 장면별 첫 프레임으로 워밍업 → 서로 다른 장면 3장으로 입력·파싱·시간 확인. 오류가 있으면 중단
6. 전체 프레임 batch_size=1 순차 실행, 프레임마다 CSV 기록
7. 입력 수·행 수·중복·누락·오류 확인, 단계별 개수와 시간 요약(run_config.json의 `summary`에도 저장)

## 설정

- `MAX_FRAMES`: 기본 `None`은 전체, `3`은 소량 실행
- `QUANTIZATION`: `auto` / `none`(BF16) / `nf4`
- `IMAGE_MAX_PIXELS`: 기본 `None`은 모델 기본 전처리(사실상 원본 해상도). 현재 프레임은 장면별 이미지 토큰 165~722개(D01 468×347 ~ D03 1221×597)입니다. 값을 정하면(예: `640 * 360`) 그보다 큰 프레임만 줄어듭니다.
- 기존 `/content/omniguard_qwen38_results.csv`가 있으면 frame 열을 이번 입력 목록과 비교해 출력하고, `.prev.csv`로 보관한 뒤 새로 씁니다.

## 결과 CSV 열

| 열 | 의미 |
|---|---|
| `frame` | `DATA_DIR`(`/content/data`) 기준 경로, 예: `frames/D04/0012.2s.jpg` |
| `stage` | Normal / Caution / Stop. 판단 보류·오류면 빈칸 |
| `elapsed_ms` | 이미지 읽기 → 전처리 → 추론 → 파싱(CUDA 동기화). 설치·로딩·워밍업·CSV 쓰기 제외 |
| `server_ms`, `confidence`, `unknown_probability` | 이 방식에서는 제공되지 않아 빈칸 |
| `abstained` | 모델이 `{"stage":null}`이면 True, 정상 분류면 False, 오류면 빈칸 |
| `error` | 추론·파싱 실패 원인. Normal로 대체하지 않음 |
| `raw_output` | 모델이 생성한 원문 |
| `inference_ms` | `model.generate` 구간만 |

- 시간은 프레임 한 장의 분류 처리 시간입니다. 카메라 입력부터 설비 정지까지의 지연이 아닙니다.
- 정답 라벨이 없으므로 단계별 개수는 모델 출력 분포이며, 정확도나 Stop 미탐률이 아닙니다.
- 실제 설비 제어 없이 프레임 분류만 합니다.
