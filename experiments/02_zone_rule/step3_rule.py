# 3단계: 학습 없이 규칙만으로 '설비 개입'과 '통행'을 구별할 수 있는가
# 클립은 safeeye_verify_v3_pose.py와 같은 60개(seed 1, 클래스당 15, 클립당 12프레임) -> 랜덤 포레스트 73%와 직접 비교
import json, random, time, csv
import numpy as np, cv2
from huggingface_hub import hf_hub_download
from rtmlib import Body
from zones import REPO, ZONES, view_of

random.seed(1)
samples = json.load(open(hf_hub_download(REPO, "samples.json", repo_type="dataset")))
samples = samples["samples"] if isinstance(samples, dict) else samples
GROUPS = {"intervention": ["Unauthorized Intervention", "Authorized Intervention"], "walking": ["Safe Walkway", "Safe Walkway Violation"]}
N_PER_CLASS, N_FRAMES, KP_MIN, DWELL = 15, 12, 0.3, 0.5
clips = []
for g, labels in GROUPS.items():
    for lab in labels:
        pool = [s for s in samples if s["ground_truth"]["label"] == lab]
        clips += [(g, lab, s["filepath"]) for s in random.sample(pool, N_PER_CLASS)]

body = Body(mode="lightweight", backend="onnxruntime", device="cpu")

def in_zone(pt, zone): return cv2.pointPolygonTest(zone, (float(pt[0]), float(pt[1])), False) >= 0

rows, t_pose = [], []
for g, lab, fp in clips:
    cap = cv2.VideoCapture(hf_hub_download(REPO, fp, repo_type="dataset")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    zone, view = None, None
    n_zone, n_zone_dwell, prev_centers = 0, 0, []
    for f in np.linspace(0, n - 1, N_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, img = cap.read()
        if not ok: continue
        if zone is None: view = view_of(img); zone = ZONES[view]
        t0 = time.perf_counter(); kps, sc = body(img); t_pose.append((time.perf_counter() - t0) * 1000)
        hit, hit_dwell, centers = False, False, []
        for k, s in zip(kps, sc):
            torso = np.linalg.norm((k[5] + k[6]) / 2 - (k[11] + k[12]) / 2) + 1e-6
            c = (k[5] + k[6] + k[11] + k[12]) / 4; centers.append((c, torso))
            # 조건 3: 신뢰도 0.3 이상인 손목이 위험 구역 안
            if any(s[j] >= KP_MIN and in_zone(k[j], zone) for j in (9, 10)):
                hit = True
                # 조건 2(대용): 직전 샘플 프레임과 비교해 몸 중심이 몸통 길이의 0.5배 미만으로 움직임 = 설비 앞에 머무름
                if any(np.linalg.norm(c - pc) < DWELL * max(torso, pt) for pc, pt in prev_centers): hit_dwell = True
        n_zone += hit; n_zone_dwell += hit_dwell; prev_centers = centers
    cap.release()
    rows.append({"clip": fp.split("/")[-1], "label": lab, "group": g, "view": view, "zone_frames": n_zone, "zone_dwell_frames": n_zone_dwell})
    print(f"{fp.split('/')[-1]:>12} {lab:<26} view {view} zone {n_zone:2d} dwell {n_zone_dwell:2d}", flush=True)

with open("step3_clips.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

y = np.array([r["group"] == "intervention" for r in rows])
res = {}
for rule, key in [("R1 wrist_in_zone", "zone_frames"), ("R2 wrist_in_zone+dwell", "zone_dwell_frames")]:
    cnt = np.array([r[key] for r in rows]); res[rule] = []
    for k in range(1, 9):
        p = cnt >= k
        tp, fp_, fn, tn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())
        res[rule].append({"k": k, "acc": round((tp + tn) / len(y), 3), "recall": round(tp / max(tp + fn, 1), 3),
                          "false_alarm_rate": round(fp_ / max(fp_ + tn, 1), 3), "TP": tp, "FP": fp_, "FN": fn, "TN": tn})
out = {"clips": len(rows), "pose": "rtmlib Body lightweight, CPU", "pose_ms_median": round(float(np.median(t_pose)), 1), "rules": res}
json.dump(out, open("step3_result.json", "w"), ensure_ascii=False, indent=1)
print("===RESULT===")
for rule, lst in res.items():
    for r in lst: print(rule, r)
print("pose_ms_median", out["pose_ms_median"])
