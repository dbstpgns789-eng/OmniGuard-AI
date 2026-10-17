# 설치: 뒤 실험에 필요한 패키지. 하나가 실패해도 나머지는 계속 설치하고, 결과에 남긴다
import subprocess, sys, importlib
from common import done

PKGS = {"openvino": "openvino", "rfdetr": "rfdetr", "supervision": "supervision", "mediapipe": "mediapipe",
        "qwen_vl_utils": "qwen-vl-utils", "accelerate": "accelerate"}
res = {}
for mod, pip in PKGS.items():
    try:
        importlib.import_module(mod); res[mod] = "already"; continue
    except Exception: pass
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip], capture_output=True, text=True, timeout=1800)
    try:
        importlib.invalidate_caches(); importlib.import_module(mod); res[mod] = "installed"
    except Exception as e:
        res[mod] = f"FAILED: {type(e).__name__} {r.stderr[-300:]}"
print(res); done("t01_setup", res)
