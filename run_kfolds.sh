TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 python eval_kfold.py \
	--checkpoint 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext' \
	--max_epoch 500

TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 python eval_kfold.py \
	--checkpoint 'models/EyeBERTv5_epoch44' \
	--max_epoch 500

