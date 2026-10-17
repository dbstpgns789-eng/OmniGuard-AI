# 실험 07-7: MediaPipe Hands가 현장 CCTV에서 손을 찾는가 (손끝 정밀 측정의 전제 조건)
# 측정: 3프레임마다 손 검출 비율, 처리 시간. 비교 기준은 같은 프레임에서 RTMPose 손목 신뢰도 0.3 이상 비율(실험 06 캐시)
# 예상: 사람이 화면에서 작아 손 검출이 거의 안 될 것. 러닝머신 옆 근접 카메라에서 다시 시험해야 한다
import os, time, urllib.request
import numpy as np, cv2
from common import CASES, DATA, OUT, load_pose, frames, done, KP_MIN

def make_detector():
    try:
        import mediapipe as mp
        h = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.3)
        return "solutions", lambda rgb: len(h.process(rgb).multi_hand_landmarks or [])
    except Exception:
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        path = os.path.join(OUT, "hand_landmarker.task")
        if not os.path.exists(path):
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", path)
        det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=path), num_hands=4, min_hand_detection_confidence=0.3))
        return "tasks", lambda rgb: len(det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).hand_landmarks)

api, detect = make_detector()
res = {"api": api}
for name, cfg in CASES.items():
    P = load_pose(name); hits, ms, rtm = [], [], []
    for i, (t, img) in enumerate(frames(DATA + f"raw_cctv/{name}.mp4", cfg["t0"], cfg["t1"])):
        if i % 3: continue
        t0 = time.perf_counter(); n = detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ms.append((time.perf_counter() - t0) * 1000)
        hits.append(n > 0)
        if i < len(P["sc"]): rtm.append(any(max(s[9], s[10]) >= KP_MIN for s in P["sc"][i]))
    res[name] = {"mediapipe_hand_found_ratio": round(float(np.mean(hits)), 3), "rtmpose_wrist_found_ratio": round(float(np.mean(rtm)), 3),
                 "ms_median": round(float(np.median(ms)), 1), "frames": len(hits)}
    print(name, res[name])
done("t07_mediapipe_hands", res)
