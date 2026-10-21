# 실험 19 출발점: 모든 관절이 위험한계에 닿으면 정지(S0). 사고 4건(택배 컨베이어 폐기)은 다 잡는 상태에서 헛정지를 하나씩 없앤다
# 이번 단계: (1) 바닥선(지나감 판단) 끄기 비교, (2) 평소 작업 헛정지마다 위험한계에 들어간 관절 기록 + 장면 사진
import os, sys, json, glob, csv, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); X = os.path.join(HERE, "..")
src = open(os.path.join(X, "15_two_raters", "evaluate.py"), encoding="utf-8").read(); exec(src[:src.index("R, AGREE, pooled")])
from common import torso_len
from huggingface_hub import hf_hub_download
VIDEOS = ["slitter", "paper_rewinder", "night_paper", "bag_machine"]
JN = ["코", "왼눈", "오른눈", "왼귀", "오른귀", "왼어깨", "오른어깨", "왼팔꿈치", "오른팔꿈치", "왼손목", "오른손목", "왼엉덩이", "오른엉덩이", "왼무릎", "오른무릎", "왼발목", "오른발목"]
GROUP = {**{j: "머리" for j in range(5)}, 5: "어깨", 6: "어깨", 7: "팔꿈치", 8: "팔꿈치", 9: "손", 10: "손", 11: "엉덩이", 12: "엉덩이", 13: "무릎", 14: "무릎", 15: "발", 16: "발"}
def params(floor_ctx=True):
    p = list(T.p_red)
    for k, v in (("hand_inside", 0.0), ("body_inside", 0.0), ("near_slow", 0.25), ("near_warn", 0.5), ("hold_s", 1.0)): p[T.NAMES.index(k)] = v
    if not floor_ctx: p[T.NAMES.index("walk_speed")] = 1e9  # 지나감 판단 끔(바닥선 미사용)
    return p
def allpart(P, F):  # 프레임마다 모든 관절의 최소 거리와, 위험한계 안(≤0)에 들어간 관절 부위
    h = float(P["wh"][1]); db = np.full(len(P["t"]), np.nan); parts = [set() for _ in P["t"]]; last = None
    for i, (K, S) in enumerate(zip(P["kps"], P["sc"])):
        hl = F["hull_at"](i); hl = [hh.astype(np.float32) for hh in (hl if isinstance(hl, list) else [hl])]; best = np.inf
        for k, s in zip(K, S):
            tl = torso_len(k, s) or last or 0.25 * h
            if torso_len(k, s): last = tl
            for j in range(17):
                if s[j] < KP_MIN: continue
                d = min(-cv2.pointPolygonTest(hh, (float(k[j][0]), float(k[j][1])), True) for hh in hl) / tl; best = min(best, d)
                if d <= 0: parts[i].add(GROUP[j])
        if best < np.inf: db[i] = best
    return db, parts
def run(F, db, floor_ctx): F2 = dict(F); F2["db"] = db; return T.judge(F2, params(floor_ctx))[0]

print("== 사고 4건")
for floor_ctx in (True, False):
    hitsA = hitsB = 0; firsts = {}
    for k in VIDEOS:
        labA, tcA = read(SRC["A"].format(k)); labB, tcB = read(SRC["B"].format(k)); tc = min([v for v in (tcA, tcB) if v is not None], default=None)
        P = dict(np.load(DATA + f"exp12/{k}_pose.npz", allow_pickle=True))
        T.Z = json.load(open(os.path.join(X, "12_accident_set", f"zone_{k}.json"))); T.Z["outlines"] = [o for o in T.Z["outlines"] if len(o) >= 10]
        T.L = json.load(open(os.path.join(X, "12_accident_set", f"floor_{k}.json")))["lines"]; F = T.features(P); db, _ = allpart(P, F)
        m = np.array([g(v) in labA and g(v) in labB for v in F["t"]]); tt = F["t"][m]; o = run(F, db, floor_ctx)[m]
        for gname, y in (("A", np.array([labA[g(v)] for v in tt])), ("B+1", np.minimum(np.array([labB[g(v)] for v in tt]) + 1, 3))):
            gs = y == 3; h0 = float(tt[gs][0]); r = score(tt, o, gs, y < 3, h0, tc, y)
            if gname == "A": hitsA += r["hit"]; firsts[k] = None if r.get("first_stop") is None else round(r["first_stop"] - h0, 2)
            else: hitsB += r["hit"]
    print("바닥선(지나감 판단)", "켬" if floor_ctx else "끔", f"| 사고 세훈 {hitsA}/4, 채우 {hitsB}/4 | 첫 정지 차이(초, 음수=세훈보다 빠름) {firsts}", flush=True)

print("== 평소 작업")
OBJ = list(json.load(open(os.path.join(X, "11_press_normal", "zone_objects.json"), encoding="utf-8"))["objects"].values()); T.L = json.load(open(os.path.join(X, "11_press_normal", "floor_user.json")))["lines"]
OUT = DATA + "exp19/step0/"; os.makedirs(OUT, exist_ok=True); FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 26); rows = []; total = 0.0; cnt = {True: 0, False: 0}
for p in sorted(glob.glob(DATA + "exp11/*_pose.npz")):
    name = os.path.basename(p)[:-9]; P = dict(np.load(p, allow_pickle=True)); F = T.features(P, OBJ); t = F["t"]; total += float(t[-1] - t[0]) + 1 / F["fps"]
    db, parts = allpart(P, F)
    for floor_ctx in (True, False):
        o = run(F, db, floor_ctx); eps = episodes(t, o == 3, 1.0); cnt[floor_ctx] += len(eps)
        if not floor_ctx: continue
        cap = cv2.VideoCapture(hf_hub_download("Voxel51/Safe_and_Unsafe_Behaviours", f"data/{name}.mp4", repo_type="dataset"))
        for s, e in eps:
            idx = np.flatnonzero((t >= s) & (t <= e)); grp = {}
            for i in idx:
                for gname in parts[i]: grp[gname] = grp.get(gname, 0) + 1
            i = idx[len(idx) // 2]; cap.set(0, float(t[i]) * 1000); ok, img = cap.read()
            for hh in F["hull_at"](i): cv2.polylines(img, [hh.astype(np.int32)], True, (0, 220, 0), 3)
            for k, s_ in zip(P["kps"][i], P["sc"][i]):
                for j in range(17):
                    if s_[j] >= KP_MIN: cv2.circle(img, (int(k[j][0]), int(k[j][1])), 6, (0, 0, 255) if GROUP[j] in parts[i] else (255, 220, 0), -1)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil); d.rectangle([0, 0, pil.width, 40], fill=(0, 0, 0))
            txt = ", ".join(f"{k_} {v}" for k_, v in sorted(grp.items(), key=lambda x: -x[1])); d.text((8, 5), f"{name} {s:.1f}~{e:.1f}초 헛정지  들어간 부위(프레임 수): {txt}", font=FONT, fill=(255, 255, 255))
            fn = f"{name}_{s:05.1f}s.jpg"; pil.save(OUT + fn, quality=88); rows.append([fn, name, round(s, 1), round(e, 1), txt, "", ""])
print(f"평소 작업 {total:.0f}초 헛정지: 바닥선 켬 {cnt[True]}회 ({cnt[True]/total*60:.2f}/분), 끔 {cnt[False]}회 ({cnt[False]/total*60:.2f}/분)")
with open(OUT + "확인표.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["사진", "영상", "시작(초)", "끝(초)", "위험한계에 들어간 부위", "실제로 위험한가(Y/N)", "왜 안전한가(메모)"]); w.writerows(rows)
for r in rows: print(r[:5])
