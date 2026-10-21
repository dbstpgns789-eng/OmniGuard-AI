# Colab T4에서 실행: 느린 경로(VLM)가 2초짜리 영상 조각을 보고 "이 사람이 지금 무엇을 하는가"를 고른다
# 사용: 같은 폴더에 rec_002103.mp4를 두고  !python colab_vlm_activity.py   → vlm_activity.csv
# 모델: Qwen2.5-VL-3B-Instruct(fp16). 조각마다 4장(0.5초 간격)을 영상으로 넣는다. 영상 7~46.5초(뒤쪽은 같은 장면 반복)
import sys, subprocess, csv, time, json
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "qwen-vl-utils", "accelerate"], check=False)
import cv2, torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-VL-3B-Instruct"
CHOICES = ["walking past the machine", "standing or watching near the machine", "cleaning or wiping the machine",
           "reaching a hand into the machine", "body being pulled into the machine", "no person"]
PROMPT = ("These frames are 2 seconds of factory CCTV. A large rotating roll (drum) is running. "
          "Focus on the worker in the red shirt. What is the worker doing? Choose exactly one:\n"
          + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(CHOICES))
          + "\nAnswer with the number first, then one short reason.")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MID, torch_dtype=torch.float16, device_map="cuda")
proc = AutoProcessor.from_pretrained(MID)
cap = cv2.VideoCapture("rec_002103.mp4"); fps = cap.get(5)
def frame_at(s):
    cap.set(1, int(round(s * fps))); ok, img = cap.read()
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((448, 464)) if ok else None
rows = []
s = 7.0
while s + 2.0 <= 46.5:
    frames = [frame_at(s + k * 0.5) for k in range(4)]
    msgs = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], videos=[frames], return_tensors="pt").to("cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    ids = model.generate(**inp, max_new_tokens=40, do_sample=False); torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000
    ans = proc.batch_decode(ids[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    num = next((int(c) for c in ans if c.isdigit()), 0)
    rows.append({"start_s": round(s, 1), "end_s": round(s + 2.0, 1), "choice": CHOICES[num - 1] if 1 <= num <= len(CHOICES) else "?", "answer": ans, "ms": round(ms)})
    print(rows[-1], flush=True); s += 1.0  # 1초씩 겹쳐서 이동
with open("vlm_activity.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("done", len(rows), "clips, median ms", sorted(r["ms"] for r in rows)[len(rows) // 2])
