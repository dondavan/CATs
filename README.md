# CATs: Confident Adaptive Transformers

Early-exit classifiers on top of a fine-tuned ALBERT, with consistency guarantees from conformal calibration.

## Workflow

```text
              ┌──────────────────────────┐
 (optional)   │ get_cat.py               │  Build AlbertWithEarlyExits and print it
              │ check the architecture   │  (no training)
              └──────────────────────────┘

┌────────────────────────────┐   ┌──────────────────────────────┐   ┌─────────────────────┐
│ 1. train_full_classifier.py│──▶│ 2. train_cat.py              │──▶│ 3. run_conformal.py │
│ Fine-tune the full ALBERT  │   │  a) early classifiers        │   │ Pick per-layer exit │
│ (last-layer classifier)    │   │  b) temperature scaling      │   │ thresholds that     │
│                            │   │  c) meta classifiers         │   │ guarantee           │
│ → fine-tuned checkpoint    │   │  + write *_preds.jsonl       │   │ consistency (1-ε)   │
└────────────────────────────┘   └──────────────────────────────┘   └─────────────────────┘
```

## Usage

### 0. Inspect the architecture (optional)

```bash
python3 get_cat.py --model_name_or_path=albert/albert-xlarge-v1
```


```bash
python3 get_cat.py --model_name_or_path=albert/albert-base-v2
```

### 1. Fine-tune the full classifier

```bash
python3 train_full_classifier.py \
    --model_name_or_path=albert/albert-xlarge-v1 \
    --task_name=stsb
```

```bash
python3 train_full_classifier.py \
    --model_name_or_path=albert/albert-base-v2 \
    --task_name=stsb
```

### 2. Train the CAT

```bash
python3 train_cat.py \
    --model_name_or_path=albert/albert-xlarge-v1 \
    --task_name=stsb \
    --output-dir=model_output \
    --do_train=True \
    --use_early_poolers=True \
    --use_meta_predictors=True \
    --use_history_logits=False \
    --early_pooler_hidden_siz=2048 \
    --num_train_epochs=1
```

## Modifications from the original code

- Disabled `use_auth_token` in the `from_pretrained` calls:

  ```python
  # use_auth_token=True if model_args.use_auth_token else None
  ```
