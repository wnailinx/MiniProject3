"""Only LoRA A/B matrices are optimized. Base weights/tokenizer stay unchanged."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "submission"))
from repair.rewriter import messages, BASE_REVISION


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--name", default="initial")
    cli.add_argument("--epochs", type=int, default=3)
    cli.add_argument("--lr", type=float, default=0.0002)
    cli.add_argument("--rank", type=int, default=16)
    cli.add_argument("--accumulate", type=int, default=8)
    cli.add_argument("--max-length", type=int, default=2304)
    cli.add_argument("--holdout", action="store_true")
    cli.add_argument("--modules", default="q_proj,v_proj,o_proj")
    cli.add_argument("--data", type=Path, default=ROOT / "training/examples.jsonl")
    args = cli.parse_args()
    output = ROOT.parent / "outputs/v10/training" / args.name
    output.mkdir(parents=True, exist_ok=False)
    (output / "training-source.py").write_bytes(Path(__file__).read_bytes())
    seed = 73106
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    base = ROOT.parent / "tmp/qwen3-0.6b"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    config = LoraConfig(r=args.rank, lora_alpha=2*args.rank, lora_dropout=.05,
                        target_modules=args.modules.split(","), bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_A" in n or "lora_B" in n for n, p in trainable)
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()]
    train, held = [], []
    dropped = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(messages(row["payload"]), tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        target = tokenizer.encode(row["target"].strip()+tokenizer.eos_token, add_special_tokens=False)
        if len(prefix)+len(target) > args.max_length:
            dropped.append({"id": row["id"], "length": len(prefix)+len(target)})
            continue
        item = {"input_ids": torch.tensor([prefix+target], dtype=torch.long),
                "labels": torch.tensor([[-100]*len(prefix)+target], dtype=torch.long)}
        is_held = int(hashlib.sha256(row["db_id"].encode()).hexdigest()[:8], 16) % 5 == 0
        (held if args.holdout and is_held else train).append(item)
    parameters = sum(p.numel() for _, p in trainable)
    record = vars(args) | {"data": str(args.data), "seed": seed, "base_revision": BASE_REVISION,
                          "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                          "train_examples": len(train), "held_examples": len(held),
                          "trainable_parameters": parameters, "total_parameters": sum(p.numel() for p in model.parameters()),
                          "dropped_overlength": dropped, "trainable_names": [n for n, p in trainable],
                          "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest()}
    (output / "config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k:record[k] for k in ["train_examples", "held_examples", "trainable_parameters", "total_parameters"]}), flush=True)
    optimizer = torch.optim.AdamW([p for n, p in trainable], lr=args.lr, weight_decay=.01)
    total_steps = math.ceil(len(train)/args.accumulate)*args.epochs
    warmup = max(1, int(total_steps*.06))
    def factor(step):
        return min(1., (step+1)/warmup)*(.15+.85*.5*(1+math.cos(math.pi*step/total_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    with (output / "log.jsonl").open("w", encoding="utf-8") as stream:
        for epoch in range(args.epochs):
            model.train()
            random.shuffle(train)
            window = []
            for index, item in enumerate(train):
                batch = {k:v.cuda() for k,v in item.items()}
                loss = model(**batch).loss
                (loss/args.accumulate).backward()
                window.append(float(loss.detach()))
                if (index+1)%args.accumulate == 0 or index+1 == len(train):
                    torch.nn.utils.clip_grad_norm_([p for n,p in trainable], 1.)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    if step % 10 == 0:
                        event = {"epoch":epoch+1,"step":step,"loss":sum(window)/len(window),
                                 "seconds":time.perf_counter()-started,
                                 "peak_cuda_mib":torch.cuda.max_memory_allocated()/2**20}
                        stream.write(json.dumps(event)+"\n"); stream.flush()
                        print(json.dumps(event), flush=True)
                        window.clear()
            model.eval()
            validation_loss = []
            with torch.inference_mode():
                for item in held:
                    validation_loss.append(float(model(**{k:v.cuda() for k,v in item.items()}).loss))
            checkpoint = output / f"epoch-{epoch+1}"
            model.peft_config["default"].base_model_name_or_path = "Qwen/Qwen3-0.6B"
            model.peft_config["default"].revision = BASE_REVISION
            model.save_pretrained(checkpoint, safe_serialization=True, save_embedding_layers=False)
            # Metadata remains outside the two-file adapter export.
            (checkpoint / "README.md").unlink(missing_ok=True)
            event = {"epoch":epoch+1,"checkpoint":str(checkpoint),
                     "held_loss":sum(validation_loss)/len(validation_loss) if validation_loss else None,
                     "seconds":time.perf_counter()-started}
            stream.write(json.dumps(event)+"\n"); stream.flush()
            print(json.dumps(event), flush=True)


if __name__ == "__main__":
    main()
