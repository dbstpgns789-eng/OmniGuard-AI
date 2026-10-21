# 실험 11: 평소 작업·통행 장면(사용자 판단 124초 모두 "정상")에서 판단기가 헛정지를 얼마나 내는가
# 판단기는 빨간 상의 영상에 맞춘 09 t13 경계값을 그대로 쓴다. 위험한계는 사용자가 그린 설비 2대(피더 롤러 입구, 언코일러)를 따로 본다
# 측정: 정지가 켜진 횟수와 1분당 횟수, 단계별 프레임 비율. 비교로 "손목이 윤곽 안이면 정지"(V1)도 함께 잰다
import os, sys, json, glob, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "10_isafety_transfer")); sys.path.insert(0, os.path.join(HERE, "..", "09_red_fit"))
import transfer as T
from common import STAGES, KP_MIN, onsets

DATA = "C:/Users/User/Desktop/OmniGuard-data/"; EXP = DATA + "exp11/"
OBJ = list(json.load(open(os.path.join(HERE, "zone_objects.json"), encoding="utf-8"))["objects"].values())
T.L = json.load(open(os.path.join(HERE, "floor_user.json")))["lines"]
res, total_s, all_out = {}, 0.0, []
HULLS = [cv2.convexHull((np.array(o) * [1920, 1080]).astype(np.float32)).reshape(-1, 2) for o in OBJ]  # 고정 CCTV라 화면 좌표 그대로
for path in sorted(glob.glob(EXP + "*_pose.npz")):
    name = os.path.basename(path)[:-9]; P = dict(np.load(path, allow_pickle=True)); F = T.features(P, OBJ)
    out, ctx = T.judge(F, T.p_red); dur = float(F["t"][-1] - F["t"][0]) + 1 / F["fps"]; total_s += dur
    v1 = np.array([any(s[j] >= KP_MIN and max(cv2.pointPolygonTest(hh, (float(k[j][0]), float(k[j][1])), False) for hh in HULLS) >= 0
                       for k, s in zip(P["kps"][i], P["sc"][i]) for j in (9, 10)) for i in range(len(F["t"]))])
    v1c = np.array([i >= 2 and v1[i - 2:i + 1].all() for i in range(len(v1))])
    res[name] = {"sec": round(dur, 1), "stop_onsets": onsets(F["t"], out, 3), "stop_ratio": round(float((out == 3).mean()), 3),
                 "slow_ratio": round(float((out == 2).mean()), 3), "warn_ratio": round(float((out == 1).mean()), 3),
                 "v1_stop_onsets": len(onsets(F["t"], v1c.astype(int) * 3, 3)), "min_hand_dist": None if np.all(np.isnan(F["dh"])) else round(float(np.nanmin(F["dh"])), 2)}
    all_out.append((name, P, F, out, ctx)); print(name, res[name], flush=True)
n_stop = sum(len(r["stop_onsets"]) for r in res.values()); n_v1 = sum(r["v1_stop_onsets"] for r in res.values())
res["_summary"] = {"clips": len(all_out), "total_sec": round(total_s, 1), "judge_stop_onsets": n_stop, "judge_stops_per_min": round(n_stop / total_s * 60, 2),
                   "v1_stop_onsets": n_v1, "v1_stops_per_min": round(n_v1 / total_s * 60, 2),
                   "frames_stage_ratio": {STAGES[c]: round(float(np.mean(np.concatenate([o for _, _, _, o, _ in all_out]) == c)), 3) for c in range(4)}}
json.dump(res, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res["_summary"], ensure_ascii=False), flush=True)

# 데모: 클립마다 판단 결과를 얹은 영상(피더·언코일러 주변 확대)
from huggingface_hub import hf_hub_download
COL = {0: (90, 90, 90), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 255)}; FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22)
REASON = {0: "", 1: "가까워지는 중", 2: "손을 넣으려 함", 3: "손이 들어감"}; vw = None
for name, P, F, out, ctx in all_out:
    cap = cv2.VideoCapture(hf_hub_download("Voxel51/Safe_and_Unsafe_Behaviours", f"data/{name}.mp4", repo_type="dataset"))
    for i in range(len(F["t"])):
        ok, img = cap.read()
        if not ok: break
        for o in OBJ: cv2.polylines(img, [cv2.convexHull((np.array(o) * P["wh"]).astype(np.int32))], True, (0, 200, 0), 3)
        for k, s in zip(P["kps"][i], P["sc"][i]):
            for j in range(17):
                if s[j] >= KP_MIN: cv2.circle(img, (int(k[j][0]), int(k[j][1])), 7 if j in (9, 10) else 4, (0, 255, 0) if j in (9, 10) else (255, 255, 0), -1)
        crop = img[0:760, 380:1300].copy(); lv = int(out[i]); cv2.rectangle(crop, (0, 0), (crop.shape[1], 44), COL[lv], -1)
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text((8, 8), f"{STAGES[lv]} {REASON[lv]}{(' · ' + ctx[i]) if ctx[i] else ''}   {name} t={F['t'][i]:.1f}s  (사람: 정상)", font=FONT, fill=(255, 255, 255))
        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        if vw is None: vw = cv2.VideoWriter(EXP + "demo_press_normal.mp4", cv2.VideoWriter_fourcc(*"mp4v"), F["fps"], (frame.shape[1], frame.shape[0]))
        vw.write(frame)
vw.release()
