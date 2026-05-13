# CATs
Confident Adaptive Transformers


# Run
python3 get_cat.py --model_name_or_path=albert/albert-xlarge-v1


python3 train_full_classifier.py --model_name_or_path=albert/albert-xlarge-v1 --task_name=stsb 
python3 train_cat.py --model_name_or_path=albert/albert-xlarge-v1 --task_name=stsb  --output-dir=model_output --do_train=True --use_early_poolers=True --use_meta_predictors=True --num_train_epochs=1 --early_pooler_hidden_siz=2048 --use_history_logits=False


# Modified
Disabled use_auth_token
#use_auth_token=True if model_args.use_auth_token else None