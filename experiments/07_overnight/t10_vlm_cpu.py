# 실험 07-10: 같은 VLM(Qwen2.5-VL-3B)을 노트북 CPU에서 돌리면 한 장에 몇 초 걸리는가 (C안을 GPU 없이 쓸 수 있는가)
# 메모리 15.7GB라 bfloat16, 작은 해상도로 핵심 프레임 12장만 돌린다. 정확도 비교는 Colab 결과가 주이고, 여기서는 지연이 주
import json, time
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from common import DATA, done

MID = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = ("This is a factory CCTV frame. A rotating roller/drum is running. "
          "Is any person's hand or body touching the rotating roller or being pulled into it? Answer YES or NO first, then give one short reason.")
labels = json.load(open(DATA + "keyframes/labels.json"))
pick = [x for x in labels if x["label"] == "YES"][:6] + [x for x in labels if x["label"] == "NO"][:6]
torch.set_num_threads(16)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MID, torch_dtype=torch.bfloat16, device_map="cpu")
proc = AutoProcessor.from_pretrained(MID, min_pixels=128 * 28 * 28, max_pixels=320 * 28 * 28)
rows = []
for x in pick:
    msgs = [{"role": "user", "content": [{"type": "image", "image": DATA + "keyframes/" + x["file"]}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True); im, _ = process_vision_info(msgs)
    inp = proc(text=[text], images=im, return_tensors="pt")
    t0 = time.perf_counter(); ids = model.generate(**inp, max_new_tokens=24, do_sample=False); ms = (time.perf_counter() - t0) * 1000
    ans = proc.batch_decode(ids[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    rows.append({**x, "answer": ans, "ms": round(ms)}); print(x["file"], x["label"], ans[:50], round(ms))
done("t10_vlm_cpu", {"model": MID, "ms_median": sorted(r["ms"] for r in rows)[len(rows) // 2],
                     "correct": sum(r["answer"].upper().startswith(r["label"]) for r in rows), "n": len(rows), "rows": rows})
