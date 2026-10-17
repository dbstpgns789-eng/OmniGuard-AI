# 바닥 선 그리기 도구(카메라 캘리브레이션용): 바닥 위의 직선 4개를 각각 두 점으로 찍는다
#   선 1, 2: 설비를 따라 길게 뻗은 방향(레일, 설비 밑단이 바닥과 만나는 선 등), 서로 평행
#   선 3, 4: 그와 직각인 방향(바닥의 노란 줄무늬 등), 서로 평행
# 왼쪽 클릭 = 점(2점이 선 하나, 8번 클릭이면 끝), 오른쪽 클릭 = 마지막 점 지우기, ESC = 끝내기. 클릭마다 자동 저장
# 사용: python draw_floor.py [시각(초), 기본 20]
import json, os, sys
import numpy as np, cv2
from common import HERE, DATA, load_pose

t_want = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
P = load_pose("lightweight"); t = P["t"]; i = int(np.argmin(np.abs(t - t_want))); w, h = P["wh"]
cap = cv2.VideoCapture(DATA + "raw_cctv/rec_002103.mp4"); cap.set(1, int(round(t[i] * cap.get(5)))); ok, base = cap.read()
S = 2.0 if w < 900 else 1.0
base = cv2.resize(base, None, fx=S, fy=S); pts = []; Hinv = np.linalg.inv(P["H"][i])
COL = [(255, 80, 0), (255, 80, 0), (0, 200, 255), (0, 200, 255)]

def redraw():
    img = base.copy()
    for k in range(0, len(pts), 2):
        c = COL[k // 2]; cv2.circle(img, pts[k], 5, c, -1)
        if k + 1 < len(pts):
            a, b = np.array(pts[k], float), np.array(pts[k + 1], float); d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
            cv2.line(img, tuple((a - d * 2000).astype(int)), tuple((b + d * 2000).astype(int)), c, 1)  # 연장선(소실점 확인용)
            cv2.line(img, pts[k], pts[k + 1], c, 3); cv2.circle(img, pts[k + 1], 5, c, -1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, f"line {min(len(pts) // 2 + 1, 4)} of 4   ({['along machine', 'along machine (parallel)', 'perpendicular', 'perpendicular (parallel)', 'done - ESC'][min(len(pts) // 2, 4)]})",
                (6, 18), 0, 0.55, (255, 255, 255), 1)
    cv2.imshow("draw floor", img); return img

def save():
    q = np.array(pts, np.float64).reshape(-1, 2) / S
    r = cv2.perspectiveTransform(q.reshape(-1, 1, 2), Hinv).reshape(-1, 2) / [w, h] if len(q) else q
    json.dump({"drawn_on_t": round(float(t[i]), 2), "lines": np.round(r, 4).tolist()}, open(os.path.join(HERE, "floor_user.json"), "w"), indent=1)
    cv2.imwrite(DATA + "exp09/floor_user.jpg", redraw())

def on_mouse(ev, x, y, *_):
    if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 8: pts.append((x, y)); save()
    elif ev == cv2.EVENT_RBUTTONDOWN and pts: pts.pop(); save()
    redraw()

cv2.namedWindow("draw floor", cv2.WINDOW_NORMAL); cv2.setMouseCallback("draw floor", on_mouse); redraw()
cv2.resizeWindow("draw floor", base.shape[1], base.shape[0]); print("window open. 8 clicks = 4 floor lines. ESC = quit", flush=True)
while True:
    k = cv2.waitKey(50) & 0xFF
    if k in (27, ord("q")): break
    redraw()
cv2.destroyAllWindows()
