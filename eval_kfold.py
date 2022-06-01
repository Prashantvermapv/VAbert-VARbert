from argparse import Namespace
from tqdm import trange, tqdm
import json
from pprint import pprint

from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer
import copy
from torch.utils.data import ConcatDataset
from sklearn.model_selection import KFold

from hpobert.dataset import HPODataset
from hpobert.trainer import BaselineNERTrainer
from hpobert.utils import calc_scores
import os
import typer
from pathlib import Path
from argparse import ArgumentParser
import operator

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # specify which GPU(s) to be used

# TODO: handle input arguments
# TODO: re-train with ugaray96/biobert_ncbi_disease_ner (https://huggingface.co/ugaray96/biobert_ncbi_disease_ner)

# SAVING
patience = 50
RESULTS_DIR = 'results'


# FOLDS
num_folds = 10
kfold = KFold(n_splits=num_folds, random_state=42, shuffle=True)

def index(ls, ids):
	return operator.itemgetter(*list(ids))(ls)

def main(args):
	# Load and handling arguments 
	args_inp = json.load(open(args.config, 'r'))	# default parameters
	keys = list(args.__dict__.keys())				# overriding ones
	for key in keys:
		args.__dict__[key.upper()] = args.__dict__.pop(key)
		if args.__dict__[key.upper()]:
			args_inp[key.upper()] = args.__dict__[key.upper()]
	args = Namespace(**args_inp)
	pprint(args_inp)

	MODEL_NAME = args.CHECKPOINT.split('/')[-1]
	SAVE_PATH = f'trained_models/{MODEL_NAME}'

    # Load data
	dataset = HPODataset(args.DATA_FILE)                                                                                                                                                                     

	#  Tokenize and Pad 
	'''
	if 'eyebert' in args.CHECKPOINT:
		tokenizer = RobertaTokenizer.from_pretrained(args.CHECKPOINT, do_lower_case=False)
	else:
		tokenizer = BertTokenizer.from_pretrained(args.CHECKPOINT, do_lower_case=False)
	'''
	tokenizer = AutoTokenizer.from_pretrained(args.CHECKPOINT, do_lower_case=False)

	dataset.set_params(tokenizer=tokenizer,
	 				   max_len=args.MAX_LEN,
	 				   batch_size=args.BATCH_SIZE,
                       val_size=args.VAL_SIZE,
	 				   seed=args.SEED
    )
 
	# Data statistics
	tag2idx, tag_values = dataset.get_tag_info()
	dataset.stats()
	print('--------------- DATA LOADED ----------------------')
 
	# Trainers
	trainer = BaselineNERTrainer()
	args.DEVICE = 'cuda'
	
	trainer.set_params(full_finetuning=args.FULL_FINETUNING,
                    	checkpoint=args.CHECKPOINT,
                     	max_epoch=args.MAX_EPOCH,
                      	max_grad_norm=args.MAX_GRAD_NORM,
                       	device=args.DEVICE)
	
	#dataset_concat = ConcatDataset([dataset.sentences, dataset.tags])

	# -------------- TRAINING
	for i, (train_ids, test_ids) in enumerate(tqdm(kfold.split(dataset.sentences),
												   desc='k-fold',
												   total=num_folds)):

		# Train (train and val)
		train_sents = index(dataset.sentences, train_ids);	train_tags = index(dataset.tags, train_ids)
		# Test
		test_sents = index(dataset.sentences, test_ids); 	test_tags = index(dataset.tags, test_ids)

		# Dataloaders for each fold
		train_dataloader, valid_dataloader = dataset.get_dataloaders(train_sents, train_tags, mode='split')
		test_dataloader = dataset.get_dataloaders(test_sents, test_tags, mode='all')

		# Training assets
		model, optimizer, scheduler = trainer.setup_training(train_dataloader=train_dataloader, 
															tag2idx=tag2idx)
	
		## Store the average loss after each epoch so we can plot them.
		loss_values, validation_loss_values = [], []
		F1_best = 0
		F1_prev = 0

		# Tensorboard writer
		writer = SummaryWriter(filename_suffix=f'_{MODEL_NAME}_fold{i}')

		# --------------- TRAINING: FIND BEST MODEL FOR EACH FOLD ------------------
		for epoch in trange(args.MAX_EPOCH, desc="Epoch"):
			# ========================================
			#               Training
			# ========================================
			# Perform one full pass over the training set.

			# Training loop
			avg_train_loss, predictions_train, true_labels_train = trainer.epoch_train(model, train_dataloader, optimizer, scheduler)
			#print("Average train loss: {}".format(avg_train_loss))	
			
			# Store the loss value for plotting the learning curve.
			loss_values.append(avg_train_loss)

			# calculate train scores
			P_train, R_train, F1_train = calc_scores(predictions_train, true_labels_train, tag_values, verbose=False)

			# ========================================
			#               Validation
			# ========================================
			# After the completion of each training epoch, measure our performance on
			# our validation set.
			avg_eval_loss, predictions_val, true_labels_val = trainer.epoch_validate(model, valid_dataloader)
			validation_loss_values.append(avg_eval_loss)
			P_val, R_val, F1_val = calc_scores(predictions_val, true_labels_val, tag_values, verbose=False)
			#tqdm.write(str(F1_val))

			# Save model based on validation best F1'score
			SAVE_PATH_FOLD = f'{SAVE_PATH}/fold={i}'
			if not os.path.exists(SAVE_PATH_FOLD):
				os.makedirs(SAVE_PATH_FOLD)

			'''
			if F1_best < F1_val and F1_val > 0.7:
				tqdm.write('Improve F1-score from {:.4f} to {:.4f} at epoch {} | P: {:.4f} | R: {:.4f}'.format(F1_best, F1_val, epoch, P_val, R_val))
				F1_best = F1_val
				nonimproved_epoch = 0
				model.save_pretrained('models_temp/hponer_epoch{}_f1_{:.4f}'.format(epoch, F1_val)) # Make sure the folder `models/` exists
			'''
			if F1_val > F1_best and F1_val > 0.50: 
				torch.save(model.state_dict(), f'{SAVE_PATH_FOLD}/{MODEL_NAME}.model')
				F1_best = F1_val
			
			if F1_best == F1_prev and F1_val > 0.50:
				patience_counter += 1
			else:
				patience_counter = 0
			if (patience_counter == patience and avg_train_loss < 1) or patience_counter == 2 * patience:
				break

			F1_prev = F1_best

			# Write to tensorboard
			writer.add_scalar('Loss/train', avg_train_loss, epoch)
			writer.add_scalar('Loss/val', avg_eval_loss, epoch)
			writer.add_scalar('F1/train', F1_train, epoch)
			writer.add_scalar('F1/val', F1_val, epoch)
			
		
	# --------------------- TESTING --------------------
	# --------------------------------------------------
	# Run trained models through k testset (of k folds) 
	# --------------------------------------------------
	f1_scores_df = pd.DataFrame()
	for i in range(num_folds):
		# Load testor and model
		testor = BaselineNERTrainer()
		args.DEVICE = 'cuda'

		checkpoint_dir = f'{SAVE_PATH}/fold={i}/{MODEL_NAME}.model'
		testor.set_params(full_finetuning=False,
							checkpoint=checkpoint_dir,
							max_epoch=15,
							max_grad_norm=args.MAX_GRAD_NORM,
							device=args.DEVICE)

		model, optimizer, scheduler = trainer.setup_training(train_dataloader=train_dataloader, #TODO: optimize this, load train_dataloader for nothing
														tag2idx=tag2idx)

		model.load_state_dict(torch.load(checkpoint_dir, map_location=torch.device('cuda')))

		# Validating
		avg_eval_loss, predictions_val, true_labels_val = testor.epoch_validate(model, test_dataloader)
		validation_loss_values.append(avg_eval_loss)
		_, _, F1_test = calc_scores(predictions_val, true_labels_val, tag_values, verbose=False)

		f1_scores_df = f1_scores_df.append({'k_fold':i, 'f1 score': F1_test}, ignore_index = True)
		
	f1_scores_df = f1_scores_df.append({'k_fold':'Mean', 'f1 score': f1_scores_df['f1 score'].mean()}, 
									ignore_index = True)
	f1_scores_df = f1_scores_df.append({'k_fold':'Standard Deviation', 'f1 score': f1_scores_df['f1 score'].std()},
									ignore_index = True)

	f1_scores_df = f1_scores_df.set_index('k_fold')
	if not os.path.exists(RESULTS_DIR):
		os.mkdir(RESULTS_DIR)
	f1_scores_df.to_csv(f'{RESULTS_DIR}/{MODEL_NAME}_f1_score.csv')


if __name__ == '__main__':
	# Handling arguments
	parser = ArgumentParser()
	parser.add_argument('--config', 	type=str, default='config/train_config.json') # config's always required, all other argument will overdrive when requested
	parser.add_argument('--data_file', 	type=str)
	parser.add_argument('--checkpoint', type=str)
	parser.add_argument('--max_epoch', 	type=int)
	args = parser.parse_args()

	# Run and measure time
	from datetime import datetime 
	start_time = datetime.now() 

	main(args)

	time_elapsed = datetime.now() - start_time 
	print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))