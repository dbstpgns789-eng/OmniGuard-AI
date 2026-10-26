# 데모 영상: 원본 + 위험한계(화면 보정으로 따라감) + 관절 + 맨 위 판단기 상태(C1 = 원칙 + 정지 유지) + 아래 판단기·정답 시간 막대
# 사용: python render.py still A04_carpet 441.2   /   python render.py video A04_carpet
import os, sys, json, numpy as np, cv2, imageio_ffmpeg, subprocess
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); MODE, NAME = sys.argv[1], sys.argv[2]; STILL = float(sys.argv[3]) if len(sys.argv) > 3 else None
sys.argv = sys.argv[:1]; src = open(os.path.join(HERE, "..", "24_temporal", "train_eval.py"), encoding="utf-8").read(); exec(src[:src.index("clips = {n: clip_data(n)")])
SRCV = {"red": DATA + "raw_cctv/rec_002103.mp4", "A04_carpet": DATA + "collected/A/A04.mp4", "B38_mixer": DATA + "collected/B/B38.mp4", "D09": DATA + "collected/D/D09.mp4"}
OUT = DATA + "demo/"; os.makedirs(OUT, exist_ok=True)
FB, FS = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 30), ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 18)
COL = {0: (120, 120, 120), 1: (255, 200, 0), 2: (230, 30, 30), -1: (45, 45, 45)}; ST = {0: "정상", 1: "주의", 2: "정지"}
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

if NAME == "red":  # 빨간 상의(개발용): 실험 09 포즈·위험한계·바닥선·세훈 판단표(4단계 → 감속은 주의로), 닿음 칸 없음
    import csv as _csv
    P = dict(np.load(DATA + "exp09/red_crop_pose_performance.npz", allow_pickle=True)); R9 = os.path.join(X, "09_red_fit")
    zone = json.load(open(os.path.join(R9, "zone_user.json"))); floor = json.load(open(os.path.join(R9, "floor_user.json"))); objects = None; tc = None
    raw = {round(float(r["시각(초)"]), 1): r["내 판단(기존)"].strip() for r in _csv.DictReader(open(DATA + "exp09/frames_humanlike/판단표_판단기비교.csv", encoding="utf-8-sig"))}
    raw.update({8.6: "정지", 22.6: "주의", 35.2: "정지"}); lab = {k: {"정상": 0, "주의": 1, "감속": 1, "정지": 2}[v] for k, v in raw.items() if v in ("정상", "주의", "감속", "정지")}
else:
    P = dict(np.load(POSE + f"{NAME}_pose.npz", allow_pickle=True)); lab, tc, hs = labels(NAME); zone, objects, floor, fronts = geometry(NAME)
F = J.features(P, zone, floor, objects); t = F["t"]; fps = F["fps"]; M, near = frame_feats(P, F)
j13 = J.run("J13", P, F); c1 = hold(j13, near, fps)
out3 = np.where(c1 == 3, 2, np.where(c1 >= 1, 1, 0))  # 판단기 감속은 주의로
ex = RULES["제외"].get(NAME); y = np.array([lab.get(round(round(v / 0.2) * 0.2, 1), -1) for v in t])
def reason(i):
    if c1[i] == 3: return "위험한계에 닿음" if j13[i] == 3 else "정지 유지 중 (사람이 아직 근처)"
    return {2: "손이 위험한계 바로 앞", 1: "위험한계 쪽으로 다가오는 중", 0: ""}[int(j13[i])]

BAR = None
def frame_img(img, i):
    h0, w0 = img.shape[:2]; hl = F["hull_at"](i); hl = hl if isinstance(hl, list) else [hl]
    ov = img.copy()
    for h_ in hl: cv2.fillPoly(ov, [h_.astype(np.int32)], (0, 200, 0))
    img = cv2.addWeighted(ov, 0.28, img, 0.72, 0)
    for h_ in hl: cv2.polylines(img, [h_.astype(np.int32)], True, (0, 230, 0), max(2, w0 // 500))
    lw = max(2, w0 // 400)
    for k, s in zip(P["kps"][i], P["sc"][i]):
        k, s = np.asarray(k), np.asarray(s)
        if (s >= KP_MIN).sum() < 3: continue
        for a, b in EDGES:
            if s[a] >= KP_MIN and s[b] >= KP_MIN: cv2.line(img, tuple(k[a].astype(int)), tuple(k[b].astype(int)), (255, 220, 0), lw)
        for j in range(17):
            if s[j] >= KP_MIN: cv2.circle(img, tuple(k[j].astype(int)), lw + (3 if j in (9, 10) else 1), (0, 255, 255) if j in (9, 10) else (255, 255, 255), -1)
    S = 1280 / w0; img = cv2.resize(img, (1280, int(h0 * S))); W = 1280
    top = Image.new("RGB", (W, 60), COL[int(out3[i])]); d = ImageDraw.Draw(top)
    d.text((16, 10), ST[int(out3[i])], font=FB, fill=(255, 255, 255)); d.text((120, 16), reason(i), font=FS, fill=(255, 255, 255))
    d.text((W - 220, 18), f"{NAME}  {t[i]:.1f}초", font=FS, fill=(255, 255, 255))
    global BAR
    if BAR is None:  # 아래 시간 막대는 한 번만 그려 두고, 프레임마다 지금 위치 선만 더한다
        BAR = Image.new("RGB", (W, 86), (20, 20, 20)); d = ImageDraw.Draw(BAR); x0, x1 = 90, W - 20; N = len(t)
        for row, (lb, arr) in enumerate((("판단기", out3), ("정답", y))):
            yy = 10 + row * 32; d.text((10, yy), lb, font=FS, fill=(255, 255, 255))
            for j in range(N):
                v = int(arr[j]) if arr[j] >= 0 else -1
                if lb == "정답" and ex and ex[0] <= t[j] < ex[1]: v = -1
                xa = x0 + int(j * (x1 - x0) / N); d.line([(xa, yy + 2), (xa, yy + 24)], fill=COL[v], width=max(1, (x1 - x0) // N + 1))
        if tc is not None:
            xc = x0 + int(np.searchsorted(t, tc) * (x1 - x0) / N); d.polygon([(xc - 7, 78), (xc + 7, 78), (xc, 68)], fill=(255, 255, 255)); d.text((xc + 10, 60), "닿음", font=FS, fill=(255, 255, 255))
    bot = BAR.copy(); d = ImageDraw.Draw(bot); x0, x1 = 90, W - 20; N = len(t)
    xp = x0 + int(i * (x1 - x0) / N); d.line([(xp, 6), (xp, 66)], fill=(255, 255, 255), width=3)
    if ex and ex[0] <= t[i] < ex[1]: ImageDraw.Draw(top).text((W // 2 - 60, 16), "(편집 구간)", font=FS, fill=(255, 255, 255))
    body = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); out = Image.new("RGB", (W, 60 + body.height + 86))
    out.paste(top, (0, 0)); out.paste(body, (0, 60)); out.paste(bot, (0, 60 + body.height)); return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)

cap = cv2.VideoCapture(SRCV[NAME]); f0 = int(round(t[0] * cap.get(5))); cap.set(1, f0)
if MODE == "still":
    i = int(np.argmin(np.abs(t - STILL))); cap.set(1, f0 + i); ok, img = cap.read()
    cv2.imencode(".jpg", frame_img(img, i), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(OUT + f"still_{NAME}_{STILL}.jpg"); print("saved still")
else:
    tmp = OUT + f"_{NAME}.mp4"; vw = None
    for i in range(len(t)):
        ok, img = cap.read()
        if not ok: break
        fr = frame_img(img, i)
        if vw is None: vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fr.shape[1], fr.shape[0]))
        vw.write(fr)
    vw.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUT + f"데모_{NAME}.mp4"]); os.remove(tmp); print("saved video", len(t))
