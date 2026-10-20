# 사후 분석: 채점을 두 단계(안전 = 정상+주의 / 정지)로 바꿔 보고, J13의 정지 경계값을 바꿔 가며 맞교환 곡선을 그린다
# 경계값은 고르지 않는다(곡선 모양만 본다). 맞춤용 10개 화면과 시험용 10개를 따로. 화면 보정 켬
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
HERE = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(HERE, "evaluate.py"), encoding="utf-8").read(); exec(src[:src.index("res = {}")])
plt.rcParams["font.family"] = fm.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
XS = [-1.0, -0.7, -0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5]  # 정지 경계값(몸통 길이, 음수 = 위험한계 안쪽). J13 = 0.0
def p_at(x):
    p = list(J.PARAMS["J13"]); p[J.NAMES.index("hand_inside")] = x; p[J.NAMES.index("body_inside")] = x; return p

data = {}
for name in sorted(os.listdir(LAB)):
    if not os.path.isdir(LAB + name): continue
    P = dict(np.load(POSE + f"{name}_pose.npz", allow_pickle=True)); lab, tc, hs = labels(name); zone, objects, floor, fronts = geometry(name)
    F = J.features(P, zone, floor, objects); t = F["t"]; key = np.array([round(round(v / 0.2) * 0.2, 1) for v in t]); y = np.array([lab.get(k, -1) for k in key])
    data[name] = (F, t, y, tc)

res = {"test": [], "tune": []}
for part in res:
    clips = [n for n in data if (n in TEST) == (part == "test")]
    for x in XS:
        tp = fp = fn = tn = hits = fs = 0; mins = 0.0
        for n in clips:
            F, t, y, tc = data[n]; o = J.T.judge(F, p_at(x))[0] == 3; m = y >= 0; pos = y == 2
            tp += int((o & pos & m).sum()); fp += int((o & ~pos & m).sum()); fn += int((~o & pos & m).sum()); tn += int((~o & ~pos & m).sum())
            win = (t >= tc - 0.5) & (t <= tc); hits += bool(win.any() and o[win].all()); fs += len(episodes(t, o & (y == 0))); mins += (y == 0).sum() / F["fps"] / 60
        res[part].append({"x": x, "recall": round(tp / max(1, tp + fn), 3), "precision": round(tp / max(1, tp + fp), 3), "fpr": round(fp / max(1, fp + tn), 3),
                          "f1": round(2 * tp / max(1, 2 * tp + fp + fn), 3), "stop_rate": f"{hits}/{len(clips)}", "hits": hits, "false_per_min": round(fs / mins, 2)})
for part, rows in res.items():
    print("==", part); [print(r) for r in rows]
json.dump(res, open(os.path.join(HERE, "binary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for part, c in (("tune", "tab:blue"), ("test", "tab:red")):
    rows = res[part]; lab_ = "맞춤용 10" if part == "tune" else "시험용 10"
    ax[0].plot([r["fpr"] for r in rows], [r["recall"] for r in rows], "o-", c=c, label=lab_)
    ax[1].plot([r["recall"] for r in rows], [r["precision"] for r in rows], "o-", c=c, label=lab_)
    ax[2].plot([r["false_per_min"] for r in rows], [r["hits"] / 10 for r in rows], "o-", c=c, label=lab_)
    j = [r for r in rows if r["x"] == 0.0][0]
    for a_, (xx, yy) in zip(ax, ((j["fpr"], j["recall"]), (j["recall"], j["precision"]), (j["false_per_min"], j["hits"] / 10))): a_.annotate("J13", (xx, yy), textcoords="offset points", xytext=(6, -12), color=c)
ax[0].plot([0, 1], [0, 1], ":", c="gray"); ax[0].set(xlabel="헛정지 비율 (정지가 아닌 프레임 중 정지로 낸 비율)", ylabel="재현율 (정지 프레임 중 잡은 비율)", title="ROC (프레임 단위)")
ax[1].set(xlabel="재현율", ylabel="정밀도 (정지로 낸 프레임 중 진짜 정지)", title="PR (프레임 단위)")
ax[2].set(xlabel="1분당 헛정지 (정상 구간)", ylabel="정지율 (닿기 0.5초 전부터 유지)", title="사고 단위 맞교환")
for a_ in ax: a_.grid(alpha=0.3); a_.legend()
fig.suptitle("J13 정지 경계값을 위험한계 안쪽 1.0 ~ 바깥 1.5로 바꿀 때 (점 = 경계값 하나)"); fig.tight_layout(); fig.savefig(os.path.join(HERE, "binary_curves.png"), dpi=130)
print("saved")
