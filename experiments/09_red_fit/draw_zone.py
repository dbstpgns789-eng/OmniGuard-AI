# 위험한계 그리기 도구: 영상 한 프레임 위에 실제 회전체 윤곽(초록 다각형)과 회전축(파랑 선)을 클릭으로 그린다
# 사용: python draw_zone.py [시각(초), 기본 33.0]
#   왼쪽 클릭: 점 추가 / 오른쪽 클릭 또는 z: 마지막 점 지우기
#   a: 회전축 그리기로 전환(2점)  p: 윤곽 그리기로 전환  n: 윤곽을 하나 더(롤이 여러 개일 때)
#   s: 저장(zone_user.json)  q: 끝내기
# 클릭 좌표는 화면 보정 행렬로 기준 프레임 좌표로 바꿔 저장하므로, 어느 프레임에서 그려도 모든 프레임에 따라 붙는다
import json, os, sys
import numpy as np, cv2
from common import HERE, DATA, load_pose

t_want = float(sys.argv[1]) if len(sys.argv) > 1 else 33.0
P = load_pose("lightweight"); t = P["t"]; i = int(np.argmin(np.abs(t - t_want))); w, h = P["wh"]
cap = cv2.VideoCapture(DATA + "raw_cctv/rec_002103.mp4"); cap.set(1, int(round(t[i] * cap.get(5)))); ok, base = cap.read()
S = 2.0 if w < 900 else 1.0  # 작은 화면은 키워서 정밀하게
base = cv2.resize(base, None, fx=S, fy=S)
polys, cur, axis, mode = [], [], [], "poly"
Hinv = np.linalg.inv(P["H"][i])

def redraw():
    img = base.copy()
    for pl in polys + [cur]:
        if len(pl) >= 2: cv2.polylines(img, [np.array(pl, np.int32)], pl is not cur, (0, 220, 0), 2)
        for p in pl: cv2.circle(img, p, 4, (0, 220, 0), -1)
    for p in axis: cv2.circle(img, p, 5, (255, 80, 0), -1)
    if len(axis) == 2: cv2.line(img, axis[0], axis[1], (255, 80, 0), 3)
    msg = f"[{'윤곽' if mode == 'poly' else '회전축'}] click=point  rclick/z=undo  a=axis p=outline n=new outline  s=save q=quit"
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, msg.replace("윤곽", "OUTLINE").replace("회전축", "AXIS"), (6, 18), 0, 0.5, (255, 255, 255), 1)
    cv2.imshow("draw zone", img); return img

def save():
    allp = polys + ([cur] if len(cur) >= 3 else [])
    if not allp and len(axis) < 2: return
    out = {"drawn_on_t": round(float(t[i]), 2), "outlines": [to_ref(p) for p in allp], "axis": to_ref(axis) if len(axis) == 2 else None}
    json.dump(out, open(os.path.join(HERE, "zone_user.json"), "w"), indent=1)
    cv2.imwrite(DATA + "exp09/zone_user.jpg", redraw())

def on_mouse(ev, x, y, *_):
    global cur
    if ev == cv2.EVENT_LBUTTONDBLCLK:  # 더블클릭: 지금 원판(윤곽)을 마치고 다음 원판 시작
        if cur and cur[-1] == (x, y): cur.pop()
        if len(cur) >= 3: polys.append(cur); cur = []
    elif ev == cv2.EVENT_LBUTTONDOWN:
        if mode == "poly": cur.append((x, y))
        elif len(axis) < 2: axis.append((x, y))
    elif ev == cv2.EVENT_MBUTTONDOWN:  # 휠 클릭: 키보드 없이 회전축 점(2개, 3번째부터는 새로 시작)
        if len(axis) >= 2: axis.clear()
        axis.append((x, y))
    elif ev == cv2.EVENT_RBUTTONDOWN:
        if mode == "poly" and cur: cur.pop()
        elif mode == "axis" and axis: axis.pop()
    if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN): save()
    redraw()

def to_ref(pts):  # 표시 좌표 → 이 프레임 픽셀 → 기준 프레임 픽셀 → 화면 비율
    q = np.array(pts, np.float64) / S
    r = cv2.perspectiveTransform(q.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    return (r / [w, h]).round(4).tolist()

cv2.namedWindow("draw zone", cv2.WINDOW_NORMAL); cv2.setMouseCallback("draw zone", on_mouse); redraw()
cv2.resizeWindow("draw zone", base.shape[1], base.shape[0]); print("window open. auto-save on every click. ESC or q = quit", flush=True)
while True:
    k = cv2.waitKey(50) & 0xFF
    if k in (ord("q"), ord("Q"), 27): save(); break
    if k == ord("z"):
        if mode == "poly" and cur: cur.pop()
        elif axis: axis.pop()
    if k == ord("a"): mode = "axis"
    if k == ord("p"): mode = "poly"
    if k == ord("n") and len(cur) >= 3: polys.append(cur); cur = []
    if k == ord("s"): save()
    redraw()
cv2.destroyAllWindows()
