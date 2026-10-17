# 준비: VLM 평가용 핵심 프레임과 정답. Colab(T4)과 CPU VLM 실험이 같은 프레임을 쓴다
# 정답(실행 전에 정함): Tc 이후 = "닿음(YES)", Tc-3초 이전 = "안 닿음(NO)", 그 사이(접근 구간)는 "접근"으로 따로 본다
import json, os, zipfile, shutil
import cv2
from common import CASES, DATA, HERE, frames, done

KF = DATA + "keyframes/"; os.makedirs(KF, exist_ok=True)
labels = []
for name, cfg in CASES.items():
    for t, img in frames(DATA + f"raw_cctv/{name}.mp4", cfg["t0"], cfg["t1"]):
        if round(t * 30) % 15: continue  # 0.5초 간격
        lab = "YES" if t >= cfg["tc"] else "NO" if t < cfg["tc"] - 3 else "APPROACH"
        fn = f"{name}_{t:05.1f}.jpg"; cv2.imwrite(KF + fn, img); labels.append({"file": fn, "video": name, "t": round(t, 2), "label": lab})
json.dump(labels, open(KF + "labels.json", "w"), indent=1)
# Colab에 올릴 묶음: 핵심 프레임 + 지연 측정용 영상 2개 + Colab 스크립트
pack = DATA + "colab_pack.zip"
with zipfile.ZipFile(pack, "w", zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir(KF): z.write(KF + f, "keyframes/" + f)
    for name in CASES: z.write(DATA + f"raw_cctv/{name}.mp4", f"clips/{name}.mp4")
    for s in ("colab_vlm.py", "colab_gpu_latency.py"): z.write(os.path.join(HERE, s), s)
done("t08_keyframes", {"n": len(labels), "by_label": {l: sum(x["label"] == l for x in labels) for l in ("YES", "NO", "APPROACH")}, "pack": pack})
