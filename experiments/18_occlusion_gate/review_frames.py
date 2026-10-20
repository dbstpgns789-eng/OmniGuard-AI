# 실험 18 설계 확인용: (1) 사고 영상에서 가림 단서가 켜진 장면, (2) 프레스 평소 작업에서 J13이 헛정지를 낸 장면을 사진으로 뽑는다
# 사진마다 위험한계(초록), 관절(하늘색), 손목(분홍)·팔꿈치(노랑)를 그리고, 세훈이 적을 확인표(확인표.csv)를 만든다
import os, sys, json, glob, csv, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
src = open(os.path.join(X, "17_reach_cues", "evaluate.py"), encoding="utf-8").read(); exec(src[:src.index("\nR = {}\n")])
from huggingface_hub import hf_hub_download
OUT = DATA + "exp18/review/"; os.makedirs(OUT, exist_ok=True); FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 26)
NS = P_RULE[T.NAMES.index("near_slow")]; rows = []

def render(img, P, F, i, title, fn):
    hl = F["hull_at"](i); hl = hl if isinstance(hl, list) else [hl]
    for hh in hl: cv2.polylines(img, [hh.astype(np.int32)], True, (0, 220, 0), 3)
    for k, s in zip(P["kps"][i], P["sc"][i]):
        for j in range(17):
            c = (255, 0, 255) if j in (9, 10) else (0, 230, 255) if j in (7, 8) else (255, 220, 0)
            if s[j] >= KP_MIN: cv2.circle(img, (int(k[j][0]), int(k[j][1])), 7 if j in (7, 8, 9, 10) else 4, c, -1)
            elif j in (9, 10) and s[j] > 0.05: cv2.circle(img, (int(k[j][0]), int(k[j][1])), 9, (255, 0, 255), 2)  # 안 보이는 손목(추정 위치)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil); d.rectangle([0, 0, pil.width, 40], fill=(0, 0, 0)); d.text((8, 5), title, font=FONT, fill=(255, 255, 255))
    pil.save(OUT + fn, quality=88)

def frame(cap, t): cap.set(0, t * 1000); ok, img = cap.read(); return img if ok else None

C12 = json.load(open(os.path.join(X, "12_accident_set", "clips.json"), encoding="utf-8"))
for k in VIDEOS:
    labA, _ = read(SRC["A"].format(k)); labB, _ = read(SRC["B"].format(k))
    P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True))
    T.Z = json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    T.L = json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json")))["lines"]; F = T.features(P)
    occ, fore, _ = cues(P, F, NS); idx = np.flatnonzero(occ)
    if not len(idx): continue
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 3) + 1); pick = [r[len(r) // 2] for r in runs][:4]
    cap = cv2.VideoCapture(DATA + "raw_cctv/" + C12[k]["file"])
    for i in pick:
        t = float(P["t"][i]); a = labA.get(g(t)); b = labB.get(g(t)); img = frame(cap, t)
        if img is None: continue
        fn = f"사고_{k}_{t:05.1f}s.jpg"; lab = f"세훈 {STAGES[a] if a is not None else '-'} / 채우 {STAGES[min(b + 1, 3)] if b is not None else '-'}"
        render(img, P, F, i, f"[사고] {k} {t:.1f}초  가림 켜짐   {lab}", fn); rows.append([fn, "사고", k, round(t, 1), lab, "", ""])

OBJ = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values()); T.L = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))["lines"]
for p in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    name = os.path.basename(p)[:-9]; P = dict(np.load(p, allow_pickle=True)); F = T.features(P, OBJ); o = T.judge(F, P_RULE)[0]; t_ = F["t"]
    eps = episodes(t_, o == 3, 1.0)
    if not eps: continue
    occ, fore, _ = cues(P, F, NS); cap = cv2.VideoCapture(hf_hub_download("Voxel51/Safe_and_Unsafe_Behaviours", f"data/{name}.mp4", repo_type="dataset"))
    for s, e in eps:
        tm = (s + e) / 2; i = int(np.argmin(abs(t_ - tm))); img = frame(cap, float(t_[i]))
        if img is None: continue
        fn = f"평소_{name}_{tm:05.1f}s.jpg"; cue = ("가림 " if occ[max(0, i - 12):i + 1].any() else "") + ("팔단축" if fore[max(0, i - 12):i + 1].any() else "")
        render(img, P, F, i, f"[평소 작업] {name} {tm:.1f}초  J13 헛정지 ({s:.1f}~{e:.1f}초)  단서: {cue or '없음'}", fn); rows.append([fn, "평소", name, round(tm, 1), "정상(세훈)", "", ""])

with open(OUT + "확인표.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["사진", "종류", "영상", "시각(초)", "판단표", "진짜 손이 롤 뒤로 들어갔나(Y/N/모름)", "메모(왜 위험/안전한가)"]); w.writerows(rows)
print(len(rows), "장", OUT)
