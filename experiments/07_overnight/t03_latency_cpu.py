# 실험 07-3: A안 기준선과 B안 후보의 단계별 지연 (노트북 CPU). 같은 300프레임(rec_002103 30~40초)에 돌린다
# A1 rtmlib Body lightweight(YOLOX 탐지 + RTMPose-s) / A1b balanced / A2 RF-DETR 탐지 + RTMPose
# B1 OpenVINO 백엔드 / B2 탐지는 5프레임마다, 그 사이에는 이전 박스 재사용
# 변형마다 결과를 partial.json에 바로 저장하고, 이미 잰 변형은 건너뛴다(중단돼도 이어서 잼)
# RF-DETR은 CPU에서 매우 느려(첫 시도에서 2시간 넘게 안 끝남) 30프레임만 잰다
import json, os, time, traceback
import numpy as np
from common import DATA, frames, done, outdir

CLIP = DATA + "raw_cctv/rec_002103.mp4"
IMGS = [img for _, img in frames(CLIP, 30.0, 40.0)][:300]
WARM = 10
PART = os.path.join(outdir("t03_latency_cpu"), "partial.json")
res = json.load(open(PART)) if os.path.exists(PART) else {}
def keep(key, val): res[key] = val; json.dump(res, open(PART, "w"), indent=1); print(key, val, flush=True)

def stat(xs): xs = np.array(xs[WARM:]); return {"median_ms": round(float(np.median(xs)), 1), "p95_ms": round(float(np.percentile(xs, 95)), 1)}

def two_stage(body, det_every=1):
    det, pose, tot, boxes = [], [], [], None
    for i, img in enumerate(IMGS):
        t0 = time.perf_counter()
        if i % det_every == 0 or boxes is None: boxes = body.det_model(img)
        t1 = time.perf_counter(); body.pose_model(img, bboxes=boxes); t2 = time.perf_counter()
        det.append((t1 - t0) * 1000); pose.append((t2 - t1) * 1000); tot.append((t2 - t0) * 1000)
    return {"detect": stat(det), "pose": stat(pose), "total": stat(tot), "frames": len(IMGS)}

from rtmlib import Body
for key, kw, every in [("A1_lightweight_ort", dict(mode="lightweight", backend="onnxruntime"), 1),
                       ("A1b_balanced_ort", dict(mode="balanced", backend="onnxruntime"), 1),
                       ("B2_lightweight_ort_det_every5", dict(mode="lightweight", backend="onnxruntime"), 5),
                       ("B1_lightweight_openvino", dict(mode="lightweight", backend="openvino"), 1),
                       ("B1b_lightweight_openvino_det_every5", dict(mode="lightweight", backend="openvino"), 5)]:
    if key in res: continue
    try:
        keep(key, two_stage(Body(device="cpu", **kw), every))
    except Exception as e:
        keep(key, {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-500:]})

if "A2_rfdetr_plus_rtmpose" not in res:
  try:  # A2: RF-DETR로 사람을 찾고 그 박스를 RTMPose에 바로 넘긴다(중복 탐지 제거)
      import rfdetr
      Model = getattr(rfdetr, "RFDETRSmall", None) or getattr(rfdetr, "RFDETRBase")
      m = Model(); body = Body(mode="lightweight", backend="onnxruntime", device="cpu")
      det, pose, tot = [], [], []
      for img in IMGS[:30]:
          t0 = time.perf_counter(); d = m.predict(img[:, :, ::-1].copy(), threshold=0.5)
          names = getattr(m, "class_names", None)
          person = [i for i, c in enumerate(d.class_id) if (names and str(names[c]).lower() == "person") or (not names and c in (0, 1))]
          boxes = d.xyxy[person] if len(person) else np.zeros((0, 4))
          t1 = time.perf_counter(); body.pose_model(img, bboxes=list(boxes)); t2 = time.perf_counter()
          det.append((t1 - t0) * 1000); pose.append((t2 - t1) * 1000); tot.append((t2 - t0) * 1000)
      keep("A2_rfdetr_plus_rtmpose", {"model": Model.__name__, "frames": 30, "detect": stat(det), "pose": stat(pose), "total": stat(tot)})
  except Exception as e:
      keep("A2_rfdetr_plus_rtmpose", {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-500:]})
done("t03_latency_cpu", res)
