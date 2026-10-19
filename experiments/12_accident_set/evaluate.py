# 실험 12: 사용자가 판단한 사고 영상 5건에, 빨간 상의 영상(09)에 맞춘 판단기를 그대로 돌린다(경계값 바꾸지 않음)
# 정답: frames_for_review/<영상>/판단표.csv. 중간 빈칸은 앞 판단이 이어짐, 끝부분 빈칸(모자이크·사고 이후)은 제외 (사용자 확인 규칙)
# 측정: 사람이 처음 정지로 본 시각 vs 판단기가 처음 정지한 시각(차이), 사람 정지 구간 중 판단기 정지 비율, 사람 정지 전 판단기의 헛정지, 단계 일치도
import os, sys, csv, json, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "10_isafety_transfer")); sys.path.insert(0, os.path.join(HERE, "..", "09_red_fit"))
import transfer as T
from common import STAGES, KP_MIN, macro_f1, onsets

DATA = "C:/Users/User/Desktop/OmniGuard-data/"; EXP = DATA + "exp12/"
C = json.load(open(os.path.join(HERE, "clips.json"), encoding="utf-8"))

def labels(k, t):
    rows = list(csv.DictReader(open(DATA + f"exp08/frames_for_review/{k}/판단표.csv", encoding="utf-8-sig")))
    seq = [(round(float(r["시각(초)"]), 1), r["단계(정상/주의/감속/정지)"].strip()) for r in rows if r["시각(초)"].strip()]
    last = max((i for i, (_, v) in enumerate(seq) if v in STAGES), default=-1); filled, cur = {}, ""
    for i, (s, v) in enumerate(seq):
        if v in STAGES: cur = v
        filled[s] = cur if i <= last else ""  # 마지막 판단 이후 빈칸은 제외
    return np.array([STAGES.index(filled.get(round(round(x / 0.2) * 0.2, 1), "")) if filled.get(round(round(x / 0.2) * 0.2, 1), "") in STAGES else -1 for x in t])

res, demo = {}, []
for k, c in C.items():
    P = dict(np.load(EXP + f"{k}_pose.npz", allow_pickle=True))
    T.Z = json.load(open(os.path.join(HERE, f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    T.L = json.load(open(os.path.join(HERE, f"floor_{k}.json")))["lines"]
    F = T.features(P); out, ctx = T.judge(F, T.p_red); y = labels(k, F["t"]); t = F["t"]
    hs = next((t[i] for i in range(len(y)) if y[i] == 3), None); js = next((t[i] for i in range(len(out)) if out[i] == 3), None)
    m = y >= 0
    res[k] = {"memo": c["memo"], "scored_frames": int(m.sum()), "human_first_stop": None if hs is None else round(float(hs), 2),
              "judge_first_stop": None if js is None else round(float(js), 2),
              "judge_minus_human_s": None if hs is None or js is None else round(float(js - hs), 2),
              "stop_recall_on_human_stop": round(float((out[y == 3] == 3).mean()), 3) if (y == 3).any() else None,
              "judge_stop_before_human_stop_frames": int(((out == 3) & (y >= 0) & (y < 3)).sum()),
              "macro_f1": round(macro_f1(y, out)[0], 3), "human_stage_onsets": {s: onsets(t, np.where(y < 0, 0, y), i) for i, s in ((1, "주의"), (2, "감속"), (3, "정지"))},
              "judge_stage_onsets": {s: onsets(t, out, i) for i, s in ((1, "주의"), (2, "감속"), (3, "정지"))},
              "per_stage": {STAGES[a]: {STAGES[b]: int((out[y == a] == b).sum()) for b in range(4)} for a in range(4) if (y == a).any()}}
    print(k, json.dumps(res[k], ensure_ascii=False), flush=True); demo.append((k, c, P, F, out, ctx, y))
json.dump(res, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

COL = {0: (90, 90, 90), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 255), -1: (40, 40, 40)}
REASON = {0: "", 1: "가까워지는 중", 2: "손을 넣으려 함", 3: "손이 들어감"}
for k, c, P, F, out, ctx, y in demo:
    T.Z = json.load(open(os.path.join(HERE, f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
    cap = cv2.VideoCapture(DATA + "raw_cctv/" + c["file"]); fps = cap.get(5); cap.set(1, int(round(F["t"][0] * fps))); w, h = P["wh"]; n = len(F["t"])
    FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", max(18, int(w / 40))); SM = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", max(14, int(w / 60)))
    vw = None; os.makedirs(EXP + f"frames_{k}", exist_ok=True)
    for i in range(n):
        ok, img = cap.read()
        if not ok: break
        hl = F["hull_at"](i).astype(np.int32); ov = img.copy(); cv2.fillPoly(ov, [hl], (0, 200, 0)); img = cv2.addWeighted(ov, 0.2, img, 0.8, 0); cv2.polylines(img, [hl], True, (0, 200, 0), 2)
        for kk, s in zip(P["kps"][i], P["sc"][i]):
            for j in range(17):
                if s[j] >= KP_MIN: cv2.circle(img, (int(kk[j][0]), int(kk[j][1])), max(4, w // 250) if j in (9, 10) else max(2, w // 500), (0, 255, 0) if j in (9, 10) else (255, 255, 0), -1)
        lv = int(out[i]); bh = int(h * 0.07); cv2.rectangle(img, (0, 0), (w, bh), COL[lv], -1)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil)
        dr.text((10, bh * 0.15), f"판단기: {STAGES[lv]} {REASON[lv]}{(' · ' + ctx[i]) if ctx[i] else ''}    사람: {STAGES[y[i]] if y[i] >= 0 else '-'}    t={F['t'][i]:.1f}s", font=FONT, fill=(255, 255, 255))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR); bars = []
        for lb, arr in (("사람", y), ("판단기", out)):
            bar = np.full((max(22, h // 30), w, 3), 25, np.uint8)
            for j in range(i + 1): x = 80 + int(j * (w - 80) / n); cv2.line(bar, (x, 2), (x, bar.shape[0] - 3), COL[int(arr[j])], max(1, (w - 80) // n + 1))
            p2 = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)); ImageDraw.Draw(p2).text((4, 2), lb, font=SM, fill=(255, 255, 255)); bars.append(cv2.cvtColor(np.array(p2), cv2.COLOR_RGB2BGR))
        frame = np.vstack([img] + bars)
        if vw is None: vw = cv2.VideoWriter(EXP + f"demo_{k}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
        if abs(F["t"][i] * 5 - round(F["t"][i] * 5)) < 2.5 / fps: cv2.imencode(".jpg", frame)[1].tofile(EXP + f"frames_{k}/{F['t'][i]:05.1f}s.jpg")
    vw.release()
