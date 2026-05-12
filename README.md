# CATs
Confident Adaptive Transformers


# Run
python3 get_cat.py --model_name_or_path=albert/albert-xlarge-v1


python3 train_full_classifier.py --model_name_or_path=albert/albert-xlarge-v1 --task_name=stsb 
python3 train_cat.py --model_name_or_path=albert/albert-xlarge-v1 --task_name=stsb  --output-dir=model_output --do_train=True 


# Modified
Disabled use_auth_token
#use_auth_token=True if model_args.use_auth_token else None