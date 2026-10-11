"""Optional local Transformers server; run in the separate inference venv.

No inference dependencies are imported by the simulator or baseline client.
"""

from __future__ import annotations

import argparse
from threading import Lock
from time import perf_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    args = parser.parse_args()
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Local code-model inference requires a working CUDA device")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    ).eval()
    app = FastAPI()
    lock = Lock()
    metadata = {
        "model": args.model,
        "revision": getattr(model.config, "_commit_hash", args.revision),
        "backend": "transformers",
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(),
        "max_input_tokens": args.max_input_tokens,
    }

    @app.get("/health")
    def health():
        return {"status": "ready", **metadata}

    @app.post("/generate")
    def generate(payload: dict):
        with lock:
            messages = payload["messages"]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            input_tokens = inputs.input_ids.shape[1]
            if input_tokens > args.max_input_tokens:
                raise HTTPException(400, "Input token budget exhausted")
            max_new_tokens = min(int(payload.get("max_tokens", 2048)), 4096)
            temperature = float(payload.get("temperature", 0.2))
            seed = int(payload.get("seed", 0))
            torch.manual_seed(seed)
            settings = dict(
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
            if temperature > 0:
                settings.update(temperature=temperature, top_p=0.95)
            start = perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**inputs, **settings)
            generated = outputs[0, input_tokens:]
            return {
                "text": tokenizer.decode(generated, skip_special_tokens=True),
                "input_tokens": input_tokens,
                "output_tokens": len(generated),
                "elapsed_seconds": perf_counter() - start,
                "finish_reason": (
                    "length" if len(generated) == max_new_tokens else "stop"
                ),
                "seed": seed,
                **metadata,
            }

    print(f"Model ready: {metadata}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
