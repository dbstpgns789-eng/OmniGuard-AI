# 실험 07-5: 야간·숙임 조건에서 포즈 모델 크기와 밝기 보정이 작업자 검출을 살리는가
# 변수: 모델 크기(lightweight/balanced/performance) × 전처리(원본/CLAHE+감마). 영상: 야간 rec_001918 12~18.5초(접촉 15초 포함)
# 실행 전에 정한 "작업자 검출" 기준: 신뢰도 0.3 이상 관절 8개 이상이고, 몸 중심이 화면 왼쪽 절반(x<0.5, 롤 쪽)에 있는 사람
# 실험 05에서 설비 기둥 오검출은 x≈0.55~0.65에 있었으므로 이 기준으로 걸러진다. 기둥 쪽 검출 수도 따로 센다
import time
import numpy as np, cv2
from rtmlib import Body
from common import DATA, frames, done, KP_MIN

IMGS = [(t, img) for t, img in frames(DATA + "raw_cctv/rec_001918.mp4", 12.0, 18.5)]

def enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return np.clip(255 * (out / 255.0) ** 0.7, 0, 255).astype(np.uint8)

res = {}
for mode in ("lightweight", "balanced", "performance"):
    body = Body(mode=mode, backend="onnxruntime", device="cpu")
    for pre in ("raw", "clahe"):
        found, contact_found, pillar, ms, wconf = 0, [], 0, [], []
        for t, img in IMGS:
            x = enhance(img) if pre == "clahe" else img
            t0 = time.perf_counter(); kps, sc = body(x); ms.append((time.perf_counter() - t0) * 1000)
            w = img.shape[1]; ok = False
            for k, s in zip(kps, sc):
                good = s >= KP_MIN
                if good.sum() < 8: continue
                cx = k[good][:, 0].mean() / w
                if cx < 0.5: ok = True; wconf.append(float(max(s[9], s[10])))
                else: pillar += 1
            found += ok
            if 15.0 <= t <= 17.0: contact_found.append(ok)
        key = f"{mode}_{pre}"
        res[key] = {"worker_found_ratio": round(found / len(IMGS), 3), "worker_found_ratio_contact_15_17s": round(float(np.mean(contact_found)), 3),
                    "right_side_detections(pillar)": pillar, "wrist_conf_mean": round(float(np.mean(wconf)), 3) if wconf else None,
                    "ms_median": round(float(np.median(ms[3:])), 1)}
        print(key, res[key])
done("t05_pose_conditions", res)
