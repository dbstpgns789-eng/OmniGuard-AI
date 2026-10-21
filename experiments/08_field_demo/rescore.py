# 실험 08 재채점: V1(겹침)·V3(손 움직임)을 Claude의 닿음/안 닿음 정답이 아니라 세훈의 판단표로 다시 채점한다
# 규칙·포즈·구역(Claude가 그린 것)은 그대로. evaluate.py의 판단 함수만 가져와 쓴다(데모 영상과 result.json은 다시 만들지 않음)
# 정답: red_crop = 09 반영본 판단표, 나머지 = exp08/frames_for_review/<영상>/판단표.csv (빈칸은 앞 판단, 마지막 판단 뒤 빈칸은 제외)
import os, csv, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(HERE, "evaluate.py"), encoding="utf-8").read()
exec(src[:src.index("summary = {}")])
ST = ["정상", "주의", "감속", "정지"]

def labels(name):
    if name == "red_crop":
        lab = {round(float(r["시각(초)"]), 1): r["내 판단(기존)"].strip() for r in csv.DictReader(open(DATA + "exp09/frames_humanlike/판단표_판단기비교.csv", encoding="utf-8-sig"))}
        lab.update({8.6: "정지", 22.6: "주의", 35.2: "정지"}); lab[7.0] = ""; return lab
    rows = list(csv.DictReader(open(DATA + f"exp08/frames_for_review/{name}/판단표.csv", encoding="utf-8-sig")))
    seq = [(round(float(r["시각(초)"]), 1), r["단계(정상/주의/감속/정지)"].strip()) for r in rows if r["시각(초)"].strip()]
    last = max(i for i, (_, v) in enumerate(seq) if v in ST); out, cur = {}, ""
    for i, (s, v) in enumerate(seq):
        if v in ST: cur = v
        out[s] = cur if i <= last else ""
    return out

R = {}
for name in ("red_crop", "slitter", "night_paper", "bag_machine"):
    _, t, _, s1, s3, _ = run(name); lab = labels(name)
    y = np.array([ST.index(lab.get(round(round(x / 0.2) * 0.2, 1), "")) if lab.get(round(round(x / 0.2) * 0.2, 1), "") in ST else -1 for x in t])
    hs = next((x for x, v in zip(t, y) if v == 3), None); R[name] = {"scored_frames": int((y >= 0).sum()), "human_first_stop": None if hs is None else round(float(hs), 2)}
    for key, s in (("V1_overlap", np.array(s1)), ("V3_motion", np.array(s3))):
        js = next((x for x, a, v in zip(t, s, y) if a and v >= 0), None)
        R[name][key] = {"human_stop_caught": round(float(s[y == 3].mean()), 3) if (y == 3).any() else None,
                        "false_stop_on_human_nonstop": round(float(s[(y >= 0) & (y < 3)].mean()), 3) if ((y >= 0) & (y < 3)).any() else None,
                        "judge_first_stop": None if js is None else round(float(js), 2)}
    print(name, json.dumps(R[name], ensure_ascii=False), flush=True)
json.dump(R, open(os.path.join(HERE, "result_rescore.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
