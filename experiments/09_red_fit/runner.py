# 실험 09 실행기(07과 같은 방식): 끝난 작업(out/<id>/DONE)은 건너뛰고, 작업마다 최대 3번 시도, PC 절전 방지
import ctypes, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP08 = os.path.join(HERE, "..", "08_field_demo")
TASKS = [  # (작업 id = 끝 표시 폴더 이름, 실행 폴더, 명령, 제한 시간 초)
    ("t1_capsule_rule_lightweight", HERE, ["t1_capsule_rule.py", "lightweight"], 3600),
    ("t2_classifier_lightweight", HERE, ["t2_classifier.py", "lightweight"], 1800),
    ("t5_render_lightweight", HERE, ["t5_render.py", "lightweight"], 3600),
    ("t3_pose_performance", HERE, ["t3_pose_big.py", "performance"], 14400),
    ("t1_capsule_rule_performance", HERE, ["t1_capsule_rule.py", "performance"], 3600),
    ("t2_classifier_performance", HERE, ["t2_classifier.py", "performance"], 1800),
    ("t5_render_performance", HERE, ["t5_render.py", "performance"], 3600),
    ("t6_v5_depth", EXP08, ["v5_depth3d.py", "red_crop"], 14400),
    ("t7_summary", HERE, ["t7_summary.py"], 600),
]
STATUS = os.path.join(HERE, "status.json")
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception: pass
st = json.load(open(STATUS, encoding="utf-8")) if os.path.exists(STATUS) else {}
def save(): json.dump(st, open(STATUS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
for tid, cwd, cmd, limit in TASKS:
    s = st.setdefault(tid, {"tries": 0, "state": "pending"})
    if os.path.exists(os.path.join(HERE, "out", tid, "DONE")): s["state"] = "done"; save(); continue
    if s["tries"] >= 3: s["state"] = "failed"; save(); continue
    s["tries"] += 1; s["state"] = "running"; s["started"] = time.strftime("%H:%M:%S"); save()
    with open(os.path.join(HERE, "logs", f"{tid}.log"), "a", encoding="utf-8") as log:
        log.write(f"\n===== try {s['tries']} {time.ctime()} =====\n"); log.flush()
        try:
            code = subprocess.run([sys.executable, "-u"] + cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=limit,
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8", "OMNI09_TASK": tid}).returncode
        except subprocess.TimeoutExpired: code = "timeout"
    if tid == "t6_v5_depth" and code == 0:  # v5는 08 폴더에 결과를 쓰므로 여기서 끝 표시를 남긴다
        os.makedirs(os.path.join(HERE, "out", tid), exist_ok=True); open(os.path.join(HERE, "out", tid, "DONE"), "w").write(time.ctime())
    ok = os.path.exists(os.path.join(HERE, "out", tid, "DONE"))
    s["state"] = "done" if ok else ("failed" if s["tries"] >= 3 else "retry"); s["last_exit"] = code; s["ended"] = time.strftime("%H:%M:%S"); save()
if all(st[x[0]]["state"] in ("done", "failed") for x in TASKS): open(os.path.join(HERE, "ALL_DONE"), "w").write(time.ctime())
