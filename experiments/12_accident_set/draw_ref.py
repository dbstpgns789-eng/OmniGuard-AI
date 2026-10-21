# 실험 12: 사고 영상마다 회전체 윤곽과 바닥 선을 그린다. 사용: python draw_ref.py <영상이름> zone / floor
# 사용: python draw_ref.py zone   → 원판 윤곽: 왼쪽 클릭으로 점, 더블클릭으로 원판 하나 완성, 오른쪽 클릭으로 지우기
#       python draw_ref.py floor  → 바닥 선 4개: 클릭 8번 (선1·2 설비 방향 평행, 선3·4 직각 방향 평행)
import json, os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = "C:/Users/User/Desktop/OmniGuard-data/exp12/"
CLIP = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "zone"
base0 = cv2.imread(OUT + f"ref_{CLIP}.jpg"); h, w = base0.shape[:2]; S = min(2.0, 1600 / w)
base = cv2.resize(base0, None, fx=S, fy=S); polys, cur, pts = [], [], []

def ratio(p): return (np.array(p, float) / S / [w, h]).round(4).tolist()
def redraw():
    img = base.copy()
    if mode == "zone":
        for pl in polys + [cur]:
            if len(pl) >= 2: cv2.polylines(img, [np.array(pl, np.int32)], pl is not cur, (0, 220, 0), 2)
            for p in pl: cv2.circle(img, p, 4, (0, 220, 0), -1)
        msg = f"ZONE: shapes done {len(polys)}  (click=point, dblclick=finish shape, rclick=undo, ESC=quit)"
    else:
        col = [(255, 80, 0)] * 2 + [(0, 200, 255)] * 2
        for k in range(0, len(pts), 2):
            cv2.circle(img, pts[k], 5, col[k // 2], -1)
            if k + 1 < len(pts):
                a, b = np.array(pts[k], float), np.array(pts[k + 1], float); d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
                cv2.line(img, tuple((a - d * 3000).astype(int)), tuple((b + d * 3000).astype(int)), col[k // 2], 1); cv2.line(img, pts[k], pts[k + 1], col[k // 2], 3)
        msg = f"FLOOR: line {min(len(pts) // 2 + 1, 4)} of 4  ({['along machine', 'along machine (parallel)', 'perpendicular', 'perpendicular (parallel)', 'done - ESC'][min(len(pts) // 2, 4)]})"
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1); cv2.putText(img, msg, (6, 18), 0, 0.55, (255, 255, 255), 1)
    cv2.imshow("draw", img); return img
def save():
    import shutil, time
    if (mode == "zone" and not polys and len(cur) < 3) or (mode != "zone" and not pts): return  # 그린 것이 없으면 기존 파일을 덮어쓰지 않음
    if mode == "zone":
        allp = polys + ([cur] if len(cur) >= 3 else [])
        _p = os.path.join(HERE, f"zone_{CLIP}.json")
        if os.path.exists(_p): shutil.copy(_p, _p + ".bak")
        json.dump({"outlines": [ratio(p) for p in allp]}, open(os.path.join(HERE, f"zone_{CLIP}.json"), "w"), indent=1)
    else:
        _p = os.path.join(HERE, f"floor_{CLIP}.json")
        if os.path.exists(_p): shutil.copy(_p, _p + ".bak")
        json.dump({"lines": ratio(pts) if pts else []}, open(os.path.join(HERE, f"floor_{CLIP}.json"), "w"), indent=1)
    cv2.imwrite(OUT + f"{mode}_{CLIP}.jpg", redraw())
def on_mouse(ev, x, y, *_):
    global cur
    if mode == "zone":
        if ev == cv2.EVENT_LBUTTONDBLCLK:
            if cur and cur[-1] == (x, y): cur.pop()
            if len(cur) >= 3: polys.append(cur); cur = []
        elif ev == cv2.EVENT_LBUTTONDOWN: cur.append((x, y))
        elif ev == cv2.EVENT_RBUTTONDOWN and cur: cur.pop()
    else:
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 8: pts.append((x, y))
        elif ev == cv2.EVENT_RBUTTONDOWN and pts: pts.pop()
    if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_RBUTTONDOWN): save()
    redraw()
cv2.namedWindow("draw", cv2.WINDOW_NORMAL); cv2.setMouseCallback("draw", on_mouse); redraw(); cv2.resizeWindow("draw", base.shape[1], base.shape[0])
print("window open. auto-save on every click. ESC = quit", flush=True)
while True:
    if (cv2.waitKey(50) & 0xFF) in (27, ord("q")): save(); break
    redraw()
cv2.destroyAllWindows()
