# Colab T4에서 실행: GPU 지연 측정 (노트북 CPU 결과와 비교용). 결과: gpu_latency.json
# 같은 300프레임(rec_002103 30~40초). rtmlib Body(onnxruntime-gpu) lightweight/balanced, RF-DETR(PyTorch), Depth Anything V2 Small
import json, time, sys, subprocess, traceback
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rtmlib", "onnxruntime-gpu", "rfdetr"], check=False)
import numpy as np, cv2, torch

cap = cv2.VideoCapture("clips/rec_002103.mp4"); fps = cap.get(5); cap.set(1, int(30 * fps))
IMGS = [cap.read()[1] for _ in range(300)]
def stat(xs): xs = np.array(xs[10:]); return {"median_ms": round(float(np.median(xs)), 1), "p95_ms": round(float(np.percentile(xs, 95)), 1)}
res = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}
from rtmlib import Body
for mode in ("lightweight", "balanced"):
    try:
        b = Body(mode=mode, backend="onnxruntime", device="cuda"); det, pose = [], []
        for img in IMGS:
            t0 = time.perf_counter(); bx = b.det_model(img); t1 = time.perf_counter(); b.pose_model(img, bboxes=bx); t2 = time.perf_counter()
            det.append((t1 - t0) * 1000); pose.append((t2 - t1) * 1000)
        res[f"body_{mode}_cuda"] = {"detect": stat(det), "pose": stat(pose)}
    except Exception as e: res[f"body_{mode}_cuda"] = {"error": str(e), "trace": traceback.format_exc()[-400:]}
try:
    import rfdetr
    M = getattr(rfdetr, "RFDETRSmall", None) or rfdetr.RFDETRBase; m = M(); xs = []
    for img in IMGS:
        torch.cuda.synchronize(); t0 = time.perf_counter(); m.predict(img[:, :, ::-1].copy(), threshold=0.5); torch.cuda.synchronize(); xs.append((time.perf_counter() - t0) * 1000)
    res["rfdetr_pytorch_cuda"] = {"model": M.__name__, **stat(xs)}
except Exception as e: res["rfdetr_pytorch_cuda"] = {"error": str(e), "trace": traceback.format_exc()[-400:]}
try:
    from transformers import pipeline
    from PIL import Image
    p = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0); xs = []
    for img in IMGS[:100]:
        torch.cuda.synchronize(); t0 = time.perf_counter(); p(Image.fromarray(img[:, :, ::-1])); torch.cuda.synchronize(); xs.append((time.perf_counter() - t0) * 1000)
    res["depth_anything_v2_small_cuda"] = stat(xs)
except Exception as e: res["depth_anything_v2_small_cuda"] = {"error": str(e)}
json.dump(res, open("gpu_latency.json", "w"), indent=1); print(res)
