# 09-7: 결과 모음표. 사용자 판단과 각 방식의 일치도, 정지 시작 시각을 한 표로 정리한다
import json, glob, os
from common import OUT, done

rows = []
for kind in ("lightweight", "performance"):
    p = OUT + f"/t1_capsule_rule_{kind}/result.json"
    if os.path.exists(p):
        r = json.load(open(p, encoding="utf-8"))
        for k in ("baseline_untuned", "fit_on_first_half", "fit_all"):
            rows.append({"pose": kind, "method": f"원통 규칙 ({k})", "macro_f1_all": r[k]["macro_f1_all"][0],
                         "macro_f1_test_27s+": r[k]["macro_f1_test_27s+"][0], "stop_onsets": r[k]["stop_onsets"], "human": r[k]["human_stop_onsets"]})
    p = OUT + f"/t2_classifier_{kind}/result.json"
    if os.path.exists(p):
        r = json.load(open(p, encoding="utf-8"))
        for k, v in r.items():
            if isinstance(v, dict) and "cv_macro_f1" in v:
                rows.append({"pose": kind, "method": f"분류기 {k}", "macro_f1_cv": v["cv_macro_f1"][0], "macro_f1_fit_all": v["fit_all_macro_f1"][0],
                             "stop_onsets_fit_all": v["fit_all_stop_onsets"], "stop_onsets_cv": v["cv_stop_onsets"], "human": r["human_stop_onsets"]})
v5 = os.path.join(os.path.dirname(OUT), "..", "08_field_demo", "result_v5.json")
done("t7_summary", {"rows": rows, "v5": json.load(open(v5, encoding="utf-8")) if os.path.exists(v5) else None})
