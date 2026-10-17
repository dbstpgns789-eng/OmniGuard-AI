# 롤 맞물림부 끼임 사고 직전 캡처(정지 화면 12장)에서 구역 규칙이 반응하는가
# 캡처는 재배포 금지라 레포 밖에 있다. 결과 표와 집계만 레포에 남긴다
import glob, json, time
import numpy as np, cv2
from rtmlib import Body

REFS = "C:/Users/User/Desktop/brain/컴퓨터비전/raw/accident-refs/"
# 회전 롤 몸체 + 맞물림부. 좌표는 화면 폭·높이 대비 비율, 사람이 격자를 보고 그린 값
ZONES = {
    1:  [(0.45, 0.25), (0.80, 0.25), (0.80, 0.60), (0.45, 0.60)],
    2:  [(0.45, 0.20), (0.95, 0.25), (0.95, 0.70), (0.55, 0.60)],
    3:  [(0.50, 0.25), (1.00, 0.30), (1.00, 0.75), (0.55, 0.60)],
    8:  [(0.00, 0.45), (0.60, 0.35), (0.60, 0.70), (0.25, 0.95), (0.00, 0.95)],
    9:  [(0.00, 0.40), (0.50, 0.20), (0.55, 0.55), (0.20, 0.70), (0.00, 0.75)],
    11: [(0.33, 0.30), (0.58, 0.30), (0.58, 0.80), (0.40, 0.80)],
    12: [(0.40, 0.42), (0.62, 0.42), (0.62, 0.80), (0.45, 0.80)],
    13: [(0.42, 0.40), (0.60, 0.40), (0.60, 0.78), (0.45, 0.78)],
    17: [(0.42, 0.45), (0.58, 0.22), (0.78, 0.22), (0.78, 0.50), (0.58, 0.72), (0.42, 0.72)],
    18: [(0.42, 0.45), (0.60, 0.15), (0.80, 0.15), (0.80, 0.50), (0.58, 0.72), (0.42, 0.72)],
    19: [(0.12, 0.05), (0.70, 0.05), (0.72, 0.75), (0.12, 0.75)],
    20: [(0.20, 0.10), (0.80, 0.15), (0.80, 0.72), (0.20, 0.72)],
}
KP_MIN, MIN_H = 0.3, 720
body = Body(mode="lightweight", backend="onnxruntime", device="cpu")

def in_zone(pt, zone): return cv2.pointPolygonTest(zone, (float(pt[0]), float(pt[1])), False) >= 0

rows, tiles = [], []
for i, zn in ZONES.items():
    p = glob.glob(REFS + f"user_{i:02d}_*.png")[0]
    img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    s = max(1.0, MIN_H / img.shape[0]); img = cv2.resize(img, None, fx=s, fy=s)  # 캡처가 작아서 키운다
    h, w = img.shape[:2]; zone = (np.array(zn) * [w, h]).astype(np.int32)
    t0 = time.perf_counter(); kps, sc = body(img); ms = (time.perf_counter() - t0) * 1000
    people = []
    for k, c in zip(kps, sc):
        wrist = any(c[j] >= KP_MIN and in_zone(k[j], zone) for j in (9, 10))
        anykp = sum(c[j] >= KP_MIN and in_zone(k[j], zone) for j in range(17))
        people.append({"wrist_in": bool(wrist), "kps_in": int(anykp), "wrist_conf": round(float(max(c[9], c[10])), 2)})
    rows.append({"ref": i, "h_px": h, "people": len(people), "wrist_hit": any(q["wrist_in"] for q in people),
                 "body_hit": any(q["kps_in"] > 0 for q in people), "detail": people, "pose_ms": round(ms, 1)})
    print(rows[-1], flush=True)
    vis = img.copy(); cv2.polylines(vis, [zone], True, (0, 0, 255), 3)
    for k, c in zip(kps, sc):
        for j in range(17):
            if c[j] >= KP_MIN: cv2.circle(vis, tuple(int(v) for v in k[j]), 7 if j in (9, 10) else 4, (0, 255, 0) if j in (9, 10) else (255, 200, 0), -1)
    vis = cv2.resize(vis, (400, int(400 * h / w)))
    cv2.putText(vis, f"{i} wrist {'Y' if rows[-1]['wrist_hit'] else 'N'} body {'Y' if rows[-1]['body_hit'] else 'N'}", (5, 25), 0, 0.7, (0, 0, 255), 2)
    tiles.append(vis)

json.dump(rows, open("result.json", "w"), ensure_ascii=False, indent=1)
H = max(t.shape[0] for t in tiles); tiles = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
sheet = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
cv2.imencode(".jpg", sheet)[1].tofile(REFS + "nip_check_sheet.jpg")  # 캡처가 들어 있으니 레포 밖에 저장
print("wrist", sum(r["wrist_hit"] for r in rows), "body", sum(r["body_hit"] for r in rows), "/", len(rows))
