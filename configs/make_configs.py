#!/usr/bin/env python
"""Generate full_<task>.json and cat_<task>.json for all GLUE tasks.

Usage: python make_configs.py [--pooling mean|cls] [--force]
Existing files are kept unless --force is given (so hand-tuned stsb/qqp configs survive).
"""
import argparse
import json
import os

RESULTS = "/var/scratch/hxu/results"
MODEL = "albert/albert-base-v2"

# Train-set sizes: cola 8.5k, mnli 393k, mrpc 3.7k, qnli 105k, qqp 364k, rte 2.5k, sst2 67k, stsb 5.7k, wnli 0.6k
# full_epochs / full_lr: stage 1 fine-tuning.  cat_epochs: early heads.  meta_iters: None -> cat_epochs.
TASKS = {
    #        full_ep full_lr full_bs cat_ep cat_bs meta_iters  large
    "cola": (5,  2e-5, 16, 20, 32, None, False),
    "mrpc": (5,  2e-5, 16, 20, 32, None, False),
    "rte":  (5,  2e-5, 16, 20, 32, None, False),
    "stsb": (5,  2e-5, 16, 20, 32, None, False),
    "wnli": (5,  2e-5, 16, 20, 32, None, False),
    "sst2": (3,  2e-5, 32, 5,  64, 2000, True),
    "qnli": (3,  2e-5, 32, 3,  64, 3000, True),
    "qqp":  (3,  2e-5, 32, 3,  64, 3000, True),
    "mnli": (3,  2e-5, 32, 3,  64, 3000, True),
}


def full_cfg(task, ep, lr, bs, large):
    return {
        "task_name": task,
        "model_name_or_path": MODEL,
        "output_dir": f"{RESULTS}/full_model_{task}",
        "overwrite_output_dir": True,
        "do_train": True,
        "do_eval": True,
        "num_train_epochs": ep,
        "per_device_train_batch_size": bs,
        "per_device_eval_batch_size": 128,
        "learning_rate": lr,
        "warmup_ratio": 0.06,
        "logging_steps": 200 if large else 50,
        "max_seq_length": 128,
        "seed": 42,
        "save_strategy": "no",
        "fp16": False,
    }


def cat_cfg(task, ep, bs, meta_iters, large, pooling):
    cfg = {
        "task_name": task,
        "model_name_or_path": f"{RESULTS}/full_model_{task}",
        "tokenizer_name": MODEL,
        "output_dir": f"{RESULTS}/cat_model_{task}_{pooling}",
        "overwrite_output_dir": True,
        "do_train": True,
        "do_eval": True,
        "do_predict_eval": True,
        "num_train_epochs": ep,
        "per_device_train_batch_size": bs,
        "per_device_eval_batch_size": 256 if large else 128,
        "learning_rate": 5e-4,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "linear",
        "logging_steps": 200 if large else 50,
        # Per-epoch eval on 10k-40k val examples is slow; final eval still runs.
        "eval_strategy": "no" if large else "epoch",
        "max_seq_length": 128,
        "seed": 42,
        "save_strategy": "no",
        "fp16": False,
        "use_early_poolers": True,
        "early_pooling": pooling,
        "init_early_from_top": True,
        "use_meta_predictors": True,
        "use_history_logits": False,
        "scaling_iterations": 500,
    }
    if meta_iters is not None:
        cfg["meta_iterations"] = meta_iters
    return cfg


def smoke(cfg, stage):
    """Shrink a config to a few-minute sanity run (results go to RESULTS/smoke/)."""
    cfg = dict(cfg)
    for key in ("output_dir", "model_name_or_path"):
        if cfg[key].startswith(RESULTS):
            cfg[key] = cfg[key].replace(RESULTS, f"{RESULTS}/smoke", 1)
    cfg.update({
        "max_train_samples": 2000,
        "max_val_samples": 500,
        "max_test_samples": 100,
        "logging_steps": 10,
        "eval_strategy": "no",
        "warmup_ratio": 0.0,
    })
    if stage == "full":
        cfg["num_train_epochs"] = 2          # 2000 ex -> ~125-250 steps, enough to beat chance
    else:
        cfg["num_train_epochs"] = 3
        cfg["scaling_iterations"] = 50
        cfg["meta_iterations"] = 100         # meta split itself is not truncated, steps are
    return cfg


def write(path, cfg, force):
    if os.path.exists(path) and os.path.getsize(path) > 0 and not force:
        print(f"keep   {path}")
        return
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")
    print(f"write  {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--force", action="store_true", help="overwrite existing configs")
    ap.add_argument("--smoke", action="store_true",
                    help="write tiny test configs to configs/smoke/ (always overwritten)")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if args.smoke:
        here = os.path.join(here, "smoke")
        os.makedirs(here, exist_ok=True)
    for task, (fep, flr, fbs, cep, cbs, meta, large) in TASKS.items():
        full = full_cfg(task, fep, flr, fbs, large)
        cat = cat_cfg(task, cep, cbs, meta, large, args.pooling)
        if args.smoke:
            full, cat = smoke(full, "full"), smoke(cat, "cat")
        write(os.path.join(here, f"full_{task}.json"), full, args.force or args.smoke)
        write(os.path.join(here, f"cat_{task}.json"), cat, args.force or args.smoke)


if __name__ == "__main__":
    main()
