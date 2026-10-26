# OmniGuard 위험한계·바닥 앞선 그리기 도구
# 사용: python tools/draw_zone.py <영상 폴더>   예: python tools/draw_zone.py OmniGuard-data/세훈_작업/판단표_사고19/A04_carpet
#       영상 폴더 안에 frames/*.jpg(0.2초 간격 프레임)가 있어야 한다. 결과는 같은 폴더의 위험한계.json, 앞선.json, 그린화면.jpg
# 필요: pip install opencv-python numpy pillow
#
# 1) 위험한계 (Z 키): 롤 양 끝 원판의 윤곽을 점으로 찍는다. 왼쪽 클릭 = 점, 더블클릭 = 원판 하나 완성, 오른쪽 클릭 = 마지막 점 지우기
#    롤이 여러 개면 위험한 롤마다 원판 2개(양 끝)를 그린다
# 2) 바닥 앞선 (F 키): 위험한 롤 바로 아래, 사람이 서는 쪽 바닥 위에 설비 방향으로 선 하나 (클릭 2번). 오른쪽 클릭 = 지우기
# ←/→ 키 또는 아래 막대: 롤이 잘 보이는 화면으로 바꾸기.  ESC = 끝. 클릭할 때마다 자동 저장
import json, os, sys, glob
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
FONT = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 20)

DIR = sys.argv[1]
FR = sorted(glob.glob(os.path.join(DIR, "frames", "*.jpg")))
read = lambda p: cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
h, w = read(FR[0]).shape[:2]; S = 1400 / w  # 작은 영상은 키우고 큰 영상은 줄여 화면 폭 1400
ZONE, FRONT = os.path.join(DIR, "위험한계.json"), os.path.join(DIR, "앞선.json")
load = lambda p, k: json.load(open(p))[k] if os.path.exists(p) else []
to_px = lambda r: (int(r[0] * w * S), int(r[1] * h * S))
polys = [[to_px(q) for q in o] for o in load(ZONE, "outlines")]; cur = []
rolls = (json.load(open(ZONE)).get("roll") if os.path.exists(ZONE) else None) or [0] * len(polys); roll_now = [max(rolls) if rolls else 0]
ref = {"zone": (json.load(open(ZONE)).get("frame") if os.path.exists(ZONE) else None), "front": (json.load(open(FRONT)).get("frame") if os.path.exists(FRONT) else None)}  # 그린 화면 (화면 보정의 기준)
pts = [to_px(q) for l in load(FRONT, "lines") for q in l]; mode = "zone"; idx = [0]

def ratio(p): return (np.array(p, float) / S / [w, h]).round(4).tolist()
def redraw():
    img = cv2.resize(read(FR[idx[0]]), None, fx=S, fy=S)
    ov = img.copy()
    for r in sorted(set(rolls)):
        ps = [np.array(p) for p, k in zip(polys, rolls) if k == r and len(p) >= 10]
        if ps: cv2.fillPoly(ov, [cv2.convexHull(np.vstack(ps).astype(np.int32))], (0, 200, 0))
    img = cv2.addWeighted(ov, 0.3, img, 0.7, 0)
    for p in polys:
        if len(p) < 10: cv2.polylines(img, [np.array(p, np.int32)], True, (0, 0, 255), 3)
    for pl in polys + [cur]:
        if len(pl) >= 2: cv2.polylines(img, [np.array(pl, np.int32)], pl is not cur, (0, 220, 0), 2)
        for p in pl: cv2.circle(img, p, 4, (0, 220, 0), -1)
    for k in range(0, len(pts), 2):
        cv2.circle(img, pts[k], 6, (255, 0, 255), -1)
        if k + 1 < len(pts): cv2.line(img, pts[k], pts[k + 1], (255, 0, 255), 4)
    if mode == "zone":
        small = sum(len(p) < 10 for p in polys)
        l1 = f"1단계 위험한계 (초록) - 롤 {len(set(rolls)) if rolls else 0}개, 원판 {len(polys)}개, 지금 롤 {roll_now[0] + 1}번" + (f" | 찍는 중 점 {len(cur)}개 (더블클릭해야 저장!)" if cur else "") + (f" | 빨간 원판 {small}개: 점 10개 미만이라 무시됨" if small else "")
        l2 = "원판 테두리를 점 10개 이상 클릭 → 더블클릭으로 완성. 색칠된 곳 = 판단기가 보는 위험한계. 다른 롤은 N 키 후 그리기. 다 하면 F 키"
    else:
        l1 = f"2단계 바닥 앞선 (분홍) - 그은 선 {len(pts) // 2}개 / 롤 {len(set(rolls)) if rolls else 0}개 (롤마다 1개씩, 같아야 함)" + (" (끝점 하나 더 클릭)" if len(pts) % 2 else "")
        l2 = "롤 바로 아래, 사람이 다가오는 쪽 바닥을 롤 방향으로 2번 클릭 = 선 하나. 롤 1번부터 순서대로. 다 했으면 ESC"
    l3 = f"오른쪽 클릭 = 되돌리기 | Z = 1단계, F = 2단계, N = 다음 롤 | ←/→ 또는 아래 막대 = 화면 바꾸기 ({os.path.basename(FR[idx[0]])}) | 자동 저장"
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); top = Image.new("RGB", (pil.width, 84), (0, 0, 0)); d = ImageDraw.Draw(top)
    d.text((8, 2), l1, font=FONT, fill=(80, 230, 80) if mode == "zone" else (255, 90, 255)); d.text((8, 30), l2, font=FONT, fill=(255, 255, 255)); d.text((8, 57), l3, font=FONT, fill=(180, 180, 180))
    out = Image.new("RGB", (pil.width, pil.height + 84)); out.paste(top, (0, 0)); out.paste(pil, (0, 84)); img = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    cv2.imshow("draw", img); return img
def save():
    json.dump({"outlines": [ratio(p) for p in polys], "roll": rolls, "frame": ref["zone"]}, open(ZONE, "w"), indent=1)
    q = pts[:len(pts) // 2 * 2]; r = ratio(q) if q else []
    json.dump({"lines": [r[k:k + 2] for k in range(0, len(r), 2)], "frame": ref["front"]}, open(FRONT, "w"), indent=1)
    cv2.imencode(".jpg", redraw())[1].tofile(os.path.join(DIR, "그린화면.jpg"))
def on_mouse(ev, x, y, *_):
    global cur
    y -= 84
    if y < 0: return
    if mode == "zone":
        if ev == cv2.EVENT_LBUTTONDBLCLK:
            if cur and cur[-1] == (x, y): cur.pop()
            if len(cur) >= 3: polys.append(cur); rolls.append(roll_now[0]); cur = []
        elif ev == cv2.EVENT_LBUTTONDOWN: cur.append((x, y))
        elif ev == cv2.EVENT_RBUTTONDOWN:
            if cur: cur.pop()
            elif polys: polys.pop(); rolls.pop()
    else:
        if ev == cv2.EVENT_LBUTTONDOWN: pts.append((x, y))
        elif ev == cv2.EVENT_RBUTTONDOWN and pts: pts.pop()
    if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_RBUTTONDOWN):
        ref[mode] = os.path.basename(FR[idx[0]]); save()
    redraw()
names = [os.path.basename(f) for f in FR]
if ref["zone"] in names: idx[0] = names.index(ref["zone"])  # 다시 열면 그렸던 화면에서 시작
def on_bar(v): idx[0] = v; redraw()
cv2.namedWindow("draw", cv2.WINDOW_NORMAL); cv2.setMouseCallback("draw", on_mouse)
cv2.createTrackbar("frame", "draw", idx[0], len(FR) - 1, on_bar); redraw()
print("창이 열렸습니다. 화면 위쪽 안내를 따라 하세요. 1단계 위험한계부터 시작합니다. ESC = 끝", flush=True)
while True:
    k = cv2.waitKeyEx(50)
    if k in (27, ord("q"), ord("f"), ord("F"), ord("n"), ord("N")) and len(cur) >= 3: polys.append(cur); rolls.append(roll_now[0]); cur = []; save()
    if k in (ord("n"), ord("N")) and mode == "zone" and rolls and rolls[-1] == roll_now[0]: roll_now[0] += 1
    if k in (27, ord("q")): save(); break
    if k in (ord("z"), ord("Z")): mode = "zone"
    if k in (ord("f"), ord("F")): mode = "front"
    if k in (2424832, 2555904): idx[0] = max(0, min(len(FR) - 1, idx[0] + (1 if k == 2555904 else -1))); cv2.setTrackbarPos("frame", "draw", idx[0])
    if k != -1: redraw()
cv2.destroyAllWindows()
