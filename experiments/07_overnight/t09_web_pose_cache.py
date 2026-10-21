# 준비: Aside가 찾은 사고 구간마다 포즈 결과, 카메라 고정 여부, 1초 간격 장면 시트를 미리 만든다
# 아침에 사람이 위험한계와 접촉 시점만 표시하면 바로 실험할 수 있게 하기 위함. 구간마다 따로 저장해서 끊겨도 이어서 한다
import json, os
import numpy as np, cv2
from rtmlib import Body
from common import DATA, CACHE, frames, done

SEGS = [  # (파일, 시작초, 끝초, 메모) Aside 보고 기준 ±2초
    ("web_BiliBili_BV1VB4y1H7Hs.mp4", 0, 40, "필름·봉투기 권취 롤러, 고정·무편집"),
    ("web_youtube_Wp7EZ8j4bb8.mp4", 0, 12, "벨트 컨베이어 끝단, 고정·무편집, 저해상도"),
    ("web_BiliBili_BV1hA411P7Ci.mp4", 0, 80, "빨간 상의 정경기 드럼, 모니터 촬영"),
    ("web_BiliBili_BV1aZ42187w6.mp4", 36, 60, "인도 제지 권취 롤러"),
    ("web_BiliBili_BV1aZ42187w6.mp4", 96, 126, "제지 리와인더"),
    ("web_BiliBili_BV1aZ42187w6.mp4", 138, 152, "휴지 원단 리와인더"),
    ("web_BiliBili_BV12f42127F3.mp4", 60, 120, "택배 컨베이어 끝 롤러 2건"),
    ("web_BiliBili_BV1jH4y1W7CU.mp4", 0, 28, "판지 슬리터-리와인더"),
    ("web_BiliBili_BV1nG411E7Qh.mp4", 0, 18, "녹색 원단 권취 롤"),
    ("web_BiliBili_BV1HS4y1X7Fi.mp4", 67, 105, "원단 권취기 + 택배 컨베이어"),
    ("web_BiliBili_BV1Knes6vEHr.mp4", 0, 10, "대형 종이 롤 리와인더"),
    ("web_youtube_WCGIljakqmQ.mp4", 6, 24, "직물 기계 끝 롤러"),
    ("web_youtube_RurcFEZnHkc.webm", 0, 22, "인도 정경기 드럼"),
    ("web_rutube_12c295a605d706a6ad68ef3a5927caa4.mp4", 520, 566, "필름 롤러 + 강관 롤러 컨베이어"),
    ("web_rutube_95ad51b9355345d96944ed9aeda7c4a5.mp4", 25, 40, "부직포 원단 기계"),
]
os.makedirs(CACHE + "sheets", exist_ok=True)
body = Body(mode="lightweight", backend="onnxruntime", device="cpu"); orb = cv2.ORB_create(1500); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
summary = {}
for fn, a, b, memo in SEGS:
    key = f"{fn.rsplit('.', 1)[0]}_{a}-{b}"; npz = CACHE + key + "_pose.npz"
    if os.path.exists(npz) and os.path.exists(CACHE + key + ".json"):
        summary[key] = json.load(open(CACHE + key + ".json", encoding="utf-8")); continue
    T, K, S, tiles, drift, ref = [], [], [], [], [], None
    for t, img in frames(DATA + "raw_cctv/" + fn, a, b):
        kps, sc = body(img); T.append(t); K.append(np.asarray(kps)); S.append(np.asarray(sc))
        if abs(t - round(t)) < 0.02:  # 1초마다 시트 타일과 카메라 이동량
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); kp, d = orb.detectAndCompute(g, None)
            if ref is None: ref = (kp, d)
            elif d is not None and ref[1] is not None:
                m = sorted(bf.match(ref[1], d), key=lambda x: x.distance)[:300]
                if len(m) >= 20:
                    A, _ = cv2.estimateAffinePartial2D(np.float32([ref[0][x.queryIdx].pt for x in m]), np.float32([kp[x.trainIdx].pt for x in m]))
                    if A is not None: drift.append(float(np.hypot(A[0, 2], A[1, 2])))
            tile = cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1])))
            for k, s in zip(kps, sc):
                for j in range(17):
                    if s[j] >= 0.3: cv2.circle(tile, (int(k[j][0] * 320 / img.shape[1]), int(k[j][1] * 320 / img.shape[1])), 2, (0, 255, 0) if j in (9, 10) else (255, 255, 0), -1)
            cv2.putText(tile, f"{t:.0f}s", (4, 18), 0, 0.6, (0, 0, 255), 2); tiles.append(tile)
    if not T: summary[key] = {"error": "no frames"}; continue
    np.savez(npz, t=np.array(T), kps=np.array(K, dtype=object), sc=np.array(S, dtype=object))
    H = max(x.shape[0] for x in tiles); tiles = [cv2.copyMakeBorder(x, 0, H - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in tiles]
    while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(CACHE + f"sheets/{key}.jpg", np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
    info = {"memo": memo, "frames": len(T), "person_frame_ratio": round(float(np.mean([len(s) > 0 for s in S])), 3),
            "camera_drift_px_max": round(max(drift), 1) if drift else None, "fixed_camera": bool(drift) and max(drift) < 10}
    json.dump(info, open(CACHE + key + ".json", "w", encoding="utf-8"), ensure_ascii=False); summary[key] = info; print(key, info)
done("t09_web_pose_cache", summary)
