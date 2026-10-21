# Colab T4에서 실행: C안 VLM이 핵심 프레임을 보고 "닿음/안 닿음"을 맞히는가, 한 장에 몇 초 걸리는가
# 사용: !python colab_vlm.py  (colab_pack.zip을 풀어 둔 폴더에서). 결과: vlm_result.json
import json, time, sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.49", "accelerate", "qwen-vl-utils"], check=False)
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODELS = sys.argv[1:] or ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]
PROMPT = ("This is a factory CCTV frame. A rotating roller/drum is running. "
          "Is any person's hand or body touching the rotating roller or being pulled into it? "
          "Answer YES or NO first, then give one short reason.")
labels = json.load(open("keyframes/labels.json"))
out = {}
for mid in MODELS:
    try:
        kw = dict(torch_dtype=torch.float16, device_map="cuda")
        if "7B" in mid:  # T4 16GB에 맞추기 위해 4비트
            from transformers import BitsAndBytesConfig
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mid, **kw)
        proc = AutoProcessor.from_pretrained(mid, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28)
        rows = []
        for i, x in enumerate(labels):
            msgs = [{"role": "user", "content": [{"type": "image", "image": "keyframes/" + x["file"]}, {"type": "text", "text": PROMPT}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            im, _ = process_vision_info(msgs)
            inp = proc(text=[text], images=im, return_tensors="pt").to("cuda")
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ids = model.generate(**inp, max_new_tokens=40, do_sample=False)
            torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000
            ans = proc.batch_decode(ids[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            rows.append({**x, "answer": ans, "pred": "YES" if ans.upper().startswith("YES") else "NO", "ms": round(ms, 1)})
            print(mid, i, x["file"], x["label"], ans[:60], round(ms))
        ev = [r for r in rows if r["label"] in ("YES", "NO")]
        out[mid] = {"accuracy_yes_no": round(sum(r["pred"] == r["label"] for r in ev) / len(ev), 3),
                    "yes_recall": round(sum(r["pred"] == "YES" for r in ev if r["label"] == "YES") / max(1, sum(r["label"] == "YES" for r in ev)), 3),
                    "false_yes_rate": round(sum(r["pred"] == "YES" for r in ev if r["label"] == "NO") / max(1, sum(r["label"] == "NO" for r in ev)), 3),
                    "ms_median": sorted(r["ms"] for r in rows[2:])[len(rows[2:]) // 2], "rows": rows}
        del model; torch.cuda.empty_cache()
    except Exception as e:
        out[mid] = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, open("vlm_result.json", "w"), ensure_ascii=False, indent=1)
print({k: {kk: v for kk, v in d.items() if kk != "rows"} for k, d in out.items()})
