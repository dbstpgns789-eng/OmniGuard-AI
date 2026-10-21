# 실험 07 재채점: V1(겹침)·V3(손 움직임)·V4(깊이)를 Claude가 정한 닿은 시각이 아니라 세훈의 판단표로 다시 채점한다
# 규칙과 포즈는 그대로. t02·t04를 그대로 실행하되 채점 함수만 바꿔 끼운다(결과 파일은 덮어쓰지 않음)
# 정답: 빨간 상의 = 09 반영본 판단표, 야간 = 12 night_paper 판단표(빈칸은 앞 판단을 잇고, 끝 빈칸은 제외)
import sys, csv, json, runpy, numpy as np
import common
DATA = common.DATA; ST = ["정상", "주의", "감속", "정지"]

def lab_red():
    lab = {round(float(r["시각(초)"]), 1): r["내 판단(기존)"].strip() for r in csv.DictReader(open(DATA + "exp09/frames_humanlike/판단표_판단기비교.csv", encoding="utf-8-sig"))}
    lab.update({8.6: "정지", 22.6: "주의", 35.2: "정지"}); lab[7.0] = ""; return lab
def lab_night():
    rows = list(csv.DictReader(open(DATA + "exp08/frames_for_review/night_paper/판단표.csv", encoding="utf-8-sig")))
    seq = [(round(float(r["시각(초)"]), 1), r["단계(정상/주의/감속/정지)"].strip()) for r in rows if r["시각(초)"].strip()]
    last = max(i for i, (_, v) in enumerate(seq) if v in ST); out, cur = {}, ""
    for i, (s, v) in enumerate(seq):
        if v in ST: cur = v
        out[s] = cur if i <= last else ""
    return out
LAB = {"rec_002103": lab_red(), "rec_001918": lab_night()}
TC = {c["tc"]: n for n, c in common.CASES.items()}

def metrics(t, raw, tc):
    stop = np.array(common.confirm(list(raw))); lab = LAB[TC[tc]]
    y = np.array([ST.index(lab.get(round(round(x / 0.2) * 0.2, 1), "")) if lab.get(round(round(x / 0.2) * 0.2, 1), "") in ST else -1 for x in t])
    hs = next((x for x, v in zip(t, y) if v == 3), None); js = next((x for x, s, v in zip(t, stop, y) if s and v >= 0), None)
    return {"human_stop_caught": round(float(stop[y == 3].mean()), 3), "false_stop_on_human_nonstop": round(float(stop[(y >= 0) & (y < 3)].mean()), 3),
            "human_first_stop": round(float(hs), 2), "judge_first_stop": None if js is None else round(float(js), 2)}
R = {}
common.metrics = metrics; common.done = lambda task, res: R.__setitem__(task, res)
for task in sys.argv[1:]: runpy.run_path(task + ".py", run_name="__main__")
json.dump(R, open("out/rescore_" + "_".join(sys.argv[1:]) + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
