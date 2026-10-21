# 실험 07-3b: t03 재측정. t03은 변형을 차례로 재서, 측정 중 CPU 부하(Aside 브라우저 등)가 변형마다 달랐을 수 있다
# (같은 A1을 실험 04에서는 149ms, t03에서는 464ms로 쟀다). 그래서 변형을 번갈아 잰다: 한 바퀴에 변형마다 100프레임, 3바퀴
# RF-DETR은 CPU에서 1초가 넘어 이미 제외 근거가 충분하므로 뺀다. 측정 직전 CPU 사용률도 남긴다
import json, os, time
import numpy as np
from rtmlib import Body
from common import DATA, frames, done, outdir

IMGS = [img for _, img in frames(DATA + "raw_cctv/rec_002103.mp4", 30.0, 40.0)][:100]
VARIANTS = {"A1_lightweight_ort": (dict(mode="lightweight", backend="onnxruntime"), 1),
            "B2_lightweight_ort_det_every5": (dict(mode="lightweight", backend="onnxruntime"), 5),
            "B1_lightweight_openvino": (dict(mode="lightweight", backend="openvino"), 1),
            "B1b_lightweight_openvino_det_every5": (dict(mode="lightweight", backend="openvino"), 5)}
bodies = {k: Body(device="cpu", **kw) for k, (kw, _) in VARIANTS.items()}
for b in bodies.values():  # 예열
    for img in IMGS[:5]: b(img)

def cpu_load():
    try:
        import psutil; return psutil.cpu_percent(interval=1.0)
    except Exception: return None

times = {k: [] for k in VARIANTS}; loads = []
for rnd in range(3):
    for k, (_, every) in VARIANTS.items():
        loads.append((rnd, k, cpu_load())); b = bodies[k]; boxes = None
        for i, img in enumerate(IMGS):
            t0 = time.perf_counter()
            if i % every == 0 or boxes is None: boxes = b.det_model(img)
            b.pose_model(img, bboxes=boxes); times[k].append((time.perf_counter() - t0) * 1000)
        print(rnd, k, round(float(np.median(times[k][-100:])), 1), flush=True)
res = {k: {"median_ms": round(float(np.median(v)), 1), "p95_ms": round(float(np.percentile(v, 95)), 1),
           "per_round_median": [round(float(np.median(v[r * 100:(r + 1) * 100])), 1) for r in range(3)]} for k, v in times.items()}
res["cpu_load_before_each"] = loads
done("t03b_latency_interleaved", res)
