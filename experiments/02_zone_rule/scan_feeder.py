# Safe_and_Unsafe_Behaviours에서 롤러 설비(코일 피더·교정기, 언코일러) 근처에 사람이 오는 장면이 있는지 찾는다
# 가까운 시점(A) 영상 60개를 무작위로 골라 1초마다 포즈를 구하고, 손목이 두 설비 영역 안이거나 발이 피더 앞 바닥에 있으면 기록
# 영역은 시점 A 기준(1920x1080) 화면 좌표, 사람이 화면을 보고 크게 잡음. 결과: 레포 밖 OmniGuard-data/exp02/feeder_hits/
import os, json, random, cv2, numpy as np
from huggingface_hub import hf_hub_download
from rtmlib import Body
from zones import REPO, view_of

OUT = "C:/Users/User/Desktop/OmniGuard-data/exp02/feeder_hits/"; os.makedirs(OUT, exist_ok=True)
FEEDER = np.array([(760, 170), (1000, 170), (1000, 440), (760, 440)], np.int32)        # 주황 피더 + 롤러 교정기
DECOILER = np.array([(600, 60), (810, 60), (810, 240), (600, 240)], np.int32)          # 노란 언코일러
FLOOR = np.array([(720, 440), (1040, 440), (1060, 600), (700, 600)], np.int32)         # 피더 앞 바닥(작업 위치)
inside = lambda p, poly, m=0: cv2.pointPolygonTest(poly.astype(np.float32), (float(p[0]), float(p[1])), True) >= -m
s = json.load(open(hf_hub_download(REPO, "samples.json", repo_type="dataset"))); s = s["samples"] if isinstance(s, dict) else s
cand = [x for x in s if x["ground_truth"]["label"] in ("Opened Panel Cover", "Closed Panel Cover", "Safe Walkway", "Authorized Intervention", "Unauthorized Intervention")]
random.seed(11); random.shuffle(cand)
body = Body(mode="lightweight", backend="openvino", device="cpu"); res, done_a = [], 0
for x in cand:
    if done_a >= 60: break
    cap = cv2.VideoCapture(hf_hub_download(REPO, x["filepath"], repo_type="dataset")); fps = cap.get(5); n = int(cap.get(7))
    cap.set(1, n // 2); ok, mid = cap.read()
    if not ok or view_of(mid) != "A": continue
    done_a += 1; hits = []
    for f in range(0, n, max(1, int(fps))):
        cap.set(1, f); ok, img = cap.read()
        if not ok: break
        kps, sc = body(img)
        for k, c in zip(kps, sc):
            hand = [where for j in (9, 10) if c[j] >= 0.3 for where, poly in (("피더", FEEDER), ("언코일러", DECOILER)) if inside(k[j], poly, 30)]
            foot = [j for j in (15, 16) if c[j] >= 0.3 and inside(k[j], FLOOR)]
            if hand or foot:
                hits.append({"t": round(f / fps, 1), "hand": sorted(set(hand)), "foot_in_front": bool(foot)})
                vis = img.copy()
                for poly, col in ((FEEDER, (0, 140, 255)), (DECOILER, (0, 255, 255)), (FLOOR, (255, 255, 255))): cv2.polylines(vis, [poly], True, col, 3)
                for j in range(17):
                    if c[j] >= 0.3: cv2.circle(vis, (int(k[j][0]), int(k[j][1])), 6 if j in (9, 10) else 3, (0, 255, 0), -1)
                cv2.imwrite(OUT + f"{x['filepath'].split('/')[-1][:-4]}_{f / fps:04.1f}s.jpg", cv2.resize(vis[0:700, 450:1250], None, fx=0.6, fy=0.6))
                break
    res.append({"clip": x["filepath"], "label": x["ground_truth"]["label"], "hits": hits}); print(x["filepath"], x["ground_truth"]["label"], len(hits), flush=True)
json.dump(res, open(OUT + "scan.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("clips", len(res), "with hits", sum(bool(r["hits"]) for r in res), "hand-in", sum(any(h["hand"] for h in r["hits"]) for r in res))
