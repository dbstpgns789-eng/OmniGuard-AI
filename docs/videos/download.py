# manifest.csv의 영상을 yt-dlp로 OmniGuard-data/collected/<묶음>/<id>.<ext>에 내려받는다. 이미 있으면 건너뜀. 결과는 download_log.csv
import csv, os, subprocess, sys, glob
import imageio_ffmpeg
OUT = "C:/Users/User/Desktop/OmniGuard-data/collected/"; FF = imageio_ffmpeg.get_ffmpeg_exe()
rows = list(csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.csv"), encoding="utf-8")))
log = []
for r in rows:
    d = OUT + r["group"] + "/"; os.makedirs(d, exist_ok=True)
    if glob.glob(d + r["id"] + ".*"): log.append((r["id"], "skip")); continue
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--no-playlist", "--ffmpeg-location", FF,
           "-f", "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b", "--merge-output-format", "mp4",
           "--socket-timeout", "30", "--retries", "3", "-o", d + r["id"] + ".%(ext)s", r["url"]]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=900)
    ok = bool(glob.glob(d + r["id"] + ".*")); log.append((r["id"], "ok" if ok else "fail: " + (p.stderr.strip().splitlines() or ["?"])[-1][:200]))
    print(r["id"], log[-1][1], flush=True)
with open(OUT + "download_log.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "result"]); w.writerows(log)
print("done", sum(1 for _, s in log if s in ("ok", "skip")), "/", len(log))
