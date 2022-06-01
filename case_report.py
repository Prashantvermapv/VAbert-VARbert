import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # specify which GPU(s) to be used

from bs4 import BeautifulSoup
import requests
import json
from hpobert.dataset import HPODataset
from hpobert.utils import view_all_entities_terminal
from hpobert.dataset import HPODataset
from hpobert.trainer import BaselineNERTrainer
from hpobert.utils import calc_scores
from transformers import BertForTokenClassification
from transformers import BertTokenizer

from pprint import pprint
from argparse import Namespace
import torch

import copy
import srsly
import csv 

des_folder = 'data/'



# ---------------- Preparing ----------------
# Utils
def retrieve_abstract(pmid):
	url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=XML&rettype=abstract'

	r = requests.get(url)
	soup = BeautifulSoup(r.content, 'html.parser')

	return soup.abstracttext.string
	

def make_data():
	# Load test set
	with open('data/eye_ner_benchmark_ann.json') as f: 			# annotation (spans)
		cases_ann = json.load(f)

	cases_text = {}
	with open('data/pubmed_case_report_10000_labels.csv') as file: 	# text
		reader = csv.reader(file, delimiter=',') 
		for row in reader: 
			cases_text[row[0]] = row[-1]     
	cases_text.pop('pmid')

	# test
	assert len(cases_text) == len(cases_ann)


	# Testing
	pmids = list(cases_text)
	all_anns = []
	all_anns_with_spans = []
	print('N: ', len(cases_text))
	for i in range(len(cases_text)):
		span_formatted = []

		pmid = pmids[i]
		try:
			#text = retrieve_abstract(pmid)
			text = cases_text[pmid]
		except:
			print(f'Cant retrieve abstract {i} | pmid: {pmid}')
			continue
		spans = copy.deepcopy(cases_ann[pmid])

		# remove irrelevant spans
		spans_target = [item for item in spans if item['sty'] == 'Disease or Syndrome']
		#print(view_all_entities_terminal(text, spans_target))

		for span in spans_target:
			span_formatted.append({'start': span['start'], 'end': span['end'], 'label': 'pnt'})

		all_anns.append({'text': text, 'spans': span_formatted})
	
		if len(span_formatted) > 0:
			all_anns_with_spans.append({'text': text, 'spans': span_formatted})
	
		if i%1000 == 0:
			print('Reached: ', i)

	#srsly.write_jsonl(os.path.join(des_folder, 'cases_report_ner_benchmark.jsonl'), all_anns)
	srsly.write_jsonl(os.path.join(des_folder, 'eye_ner_benchmark_ann.jsonl'), all_anns_with_spans)


def evaluate():
	# --------- Evaluating -----------------
	# Load parameter 
	print('------- Loading data... --------')
	params = json.load(open('config/train_config.json', 'r'))
	pprint(params)
	p = Namespace(**params)
	
	# Create test dataloader
	#testset = HPODataset('data/cases_report_ner_benchmark.jsonl')
	testset = HPODataset('data/eye_ner_benchmark_ann.jsonl') 
	tokenizer = BertTokenizer.from_pretrained(p.CHECKPOINT, do_lower_case=False)
	testset.set_params(tokenizer=tokenizer,
	 				   max_len=p.MAX_LEN,
	 				   batch_size=p.BATCH_SIZE,
                       val_size=p.VAL_SIZE,
	 				   seed=p.SEED
    )

	# Dataloader
	testloader = testset.get_dataloaders(mode='all')
	tag2idx, tag_values = testset.get_tag_info()
	testset.stats()
	print('Done')

	# Load validator
	print('-------- Loading validator... --------')
	validator = BaselineNERTrainer()
	p.DEVICE = 'cuda'

	validator.set_params(full_finetuning=p.FULL_FINETUNING,
						checkpoint=p.CHECKPOINT,
						max_epoch=p.MAX_EPOCH,
						max_grad_norm=p.MAX_GRAD_NORM,
						device=p.DEVICE)

	# Load model
	checkpoint = 'models/hponer_epoch167_f1_0.7012_best/'
	model = BertForTokenClassification.from_pretrained(checkpoint).to(p.DEVICE)
	print('Done')
	#model.classifier = torch.nn.Linear(768, len(tag2idx))
	#model.num_labels = len(tag2idx)
	#model.output_attentions = False
	#model.output_hidden_states = False

	# Evaluate
	print('-------- Validating... --------')
	_, predictions, true_labels = validator.epoch_validate(model, testloader)
	P_val, R_val, F1_val = calc_scores(predictions, true_labels, tag_values, verbose=True)
	print('DONE')

if __name__ == '__main__':
	from datetime import datetime 
	start_time = datetime.now() 

	# INSERT YOUR CODE 
	#make_data()
	evaluate()

	time_elapsed = datetime.now() - start_time 
	print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
