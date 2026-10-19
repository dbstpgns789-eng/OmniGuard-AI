# 실험 14: 위험 설비의 "바닥 앞선"을 긋는다. 위험한 롤 바로 아래, 사람이 서는 쪽 바닥 위의 선 (클릭 2번 = 선 하나)
# 사용: python draw_front.py <영상>   영상 = red, wide, press, slitter, paper_rewinder, night_paper, conveyor_parcel, bag_machine
#   press는 선 2개(1번 피더 롤러, 2번 언코일러). 왼쪽 클릭 = 점, 오른쪽 클릭 = 마지막 점 지우기, ESC = 끝. 클릭마다 자동 저장
# 배경은 그때 그린 위험한계 그림(zone_*.jpg). 좌표는 각 실험의 기준 화면 비율로 저장한다(red만 t=20초 화면 → 기준 화면으로 변환)
import json, os, sys, shutil
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); D = "C:/Users/User/Desktop/OmniGuard-data/"
KEY = sys.argv[1]; N_LINES = 2 if KEY == "press" else 1
BASE = {"red": "exp09/zone_user.jpg", "wide": "exp10/zone_user.jpg", "press": "exp11/zone_user.jpg"}.get(KEY, f"exp12/zone_{KEY}.jpg")
base = cv2.imread(D + BASE); H0, W0 = base.shape[:2]; S = min(1.0, 1500 / W0); base = cv2.resize(base, None, fx=S, fy=S); pts = []
OUT = os.path.join(HERE, f"front_{KEY}.json")

def to_ref(q):  # 배경 그림 픽셀 → 기준 화면 비율
    r = np.array(q, float) / S / [W0, H0]
    if KEY == "red":  # 09 그림은 t=20초 화면. 기준 화면으로 되돌린다
        P = dict(np.load(D + "exp09/red_crop_pose_performance.npz", allow_pickle=True)); i = int(np.argmin(np.abs(P["t"] - 20.0))); w, h = P["wh"]
        r = cv2.perspectiveTransform((r * [w, h]).reshape(-1, 1, 2), np.linalg.inv(P["H"][i])).reshape(-1, 2) / [w, h]
    return r.round(4).tolist()

def redraw():
    img = base.copy()
    for k in range(0, len(pts), 2):
        cv2.circle(img, pts[k], 6, (255, 0, 255), -1)
        if k + 1 < len(pts): cv2.line(img, pts[k], pts[k + 1], (255, 0, 255), 4); cv2.circle(img, pts[k + 1], 6, (255, 0, 255), -1)
    n = len(pts) // 2; name = ["feeder roller", "uncoiler"][min(n, 1)] if KEY == "press" else "danger machine"
    msg = "done - ESC" if n >= N_LINES else f"line {n + 1} of {N_LINES}: floor line right under the {name}, worker side"
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1); cv2.putText(img, msg, (6, 20), 0, 0.6, (255, 255, 255), 1)
    cv2.imshow("draw front", img); return img

def save():
    if len(pts) < 2: return  # 선이 없으면 기존 파일을 덮어쓰지 않음
    if os.path.exists(OUT): shutil.copy(OUT, OUT + ".bak")
    q = pts[:len(pts) // 2 * 2]; r = to_ref(q)
    json.dump({"lines": [r[k:k + 2] for k in range(0, len(r), 2)]}, open(OUT, "w"), indent=1)
    cv2.imwrite(D + f"exp14/front_{KEY}.jpg", redraw())

def on_mouse(ev, x, y, *_):
    if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2 * N_LINES: pts.append((x, y)); save()
    elif ev == cv2.EVENT_RBUTTONDOWN and pts: pts.pop(); save()
    redraw()

cv2.namedWindow("draw front", cv2.WINDOW_NORMAL); cv2.setMouseCallback("draw front", on_mouse); redraw(); cv2.resizeWindow("draw front", base.shape[1], base.shape[0])
print("window open. click 2 points per line. ESC = quit", flush=True)
while (cv2.waitKey(50) & 0xFF) not in (27, ord("q")): redraw()
cv2.destroyAllWindows()
