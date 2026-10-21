# 밤샘 실행기: 작업을 순서대로 별도 프로세스로 돌린다. 끝난 작업(out/<작업>/DONE)은 건너뛰므로 몇 번을 다시 실행해도 이어서 한다
# 작업마다 최대 3번 시도, 실패는 status.json에 남기고 다음 작업으로 넘어간다. 실행 중에는 PC가 잠들지 않게 한다
import ctypes, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = [  # (작업, 제한 시간 초)
    ("t01_setup", 3600), ("t02_v3_motion", 1800), ("t08_keyframes", 1800), ("t06_static_filter", 1800),
    ("t03_latency_cpu", 5400), ("t03b_latency_interleaved", 3600), ("t04_depth_gate", 7200), ("t05_pose_conditions", 7200),
    ("t07_mediapipe_hands", 3600), ("t09_web_pose_cache", 14400), ("t10_vlm_cpu", 10800),
]
MAX_TRY = 3
STATUS = os.path.join(HERE, "status.json")
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
except Exception: pass

st = json.load(open(STATUS, encoding="utf-8")) if os.path.exists(STATUS) else {}
def save(): json.dump(st, open(STATUS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for name, limit in TASKS:
    s = st.setdefault(name, {"tries": 0, "state": "pending"})
    if os.path.exists(os.path.join(HERE, "out", name, "DONE")): s["state"] = "done"; save(); continue
    if s["tries"] >= MAX_TRY: s["state"] = "failed"; save(); continue
    s["tries"] += 1; s["state"] = "running"; s["started"] = time.strftime("%H:%M:%S"); save()
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", f"{name}.log"), "a", encoding="utf-8") as log:
        log.write(f"\n===== try {s['tries']} {time.ctime()} =====\n"); log.flush()
        try:
            r = subprocess.run([sys.executable, "-u", f"{name}.py"], cwd=HERE, stdout=log, stderr=subprocess.STDOUT, timeout=limit,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            code = r.returncode
        except subprocess.TimeoutExpired: code = "timeout"
    ok = os.path.exists(os.path.join(HERE, "out", name, "DONE"))
    s["state"] = "done" if ok else ("failed" if s["tries"] >= MAX_TRY else "retry"); s["last_exit"] = code; s["ended"] = time.strftime("%H:%M:%S"); save()
if all(st[n]["state"] in ("done", "failed") for n, _ in TASKS):
    open(os.path.join(HERE, "ALL_DONE"), "w").write(time.ctime())
