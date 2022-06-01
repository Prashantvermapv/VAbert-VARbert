from transformers import BertForTokenClassification, BertTokenizer, RobertaForTokenClassification, RobertaTokenizer
import torch
import numpy as np
from hpobert.dataset import HPODataset

data_file = "data/meh_eyedisease.jsonl"
device = 'cpu'
model_checkpoint_path = 'models/hponer_epoch167_f1_0.7012.pth/'
checkpoint = "dmis-lab/biobert-v1.1"


def predict(test_sentence):
	dataset = HPODataset(data_file)   
	_, tag_values = dataset.get_tag_info()

	tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=False)

	tokenized_sentence = tokenizer.encode(test_sentence)
	input_ids = torch.tensor([tokenized_sentence]).cpu()

	model = BertForTokenClassification.from_pretrained(model_checkpoint_path)

	with torch.no_grad():
		output = model(input_ids)
	label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

	# join bpe split tokens
	tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
	new_tokens, new_labels = [], []
	for token, label_idx in zip(tokens, label_indices[0]):
		if token not in ['[CLS]', '[SEP]', '[PAD]']:
			if token.startswith("##"):
				new_tokens[-1] = new_tokens[-1] + token[2:]
			else:
				new_labels.append(tag_values[label_idx])
				new_tokens.append(token)
 
	for token, label in zip(new_tokens, new_labels):
		print("{}\t{}".format(label, token))


if __name__ == '__main__':
	test_sentence = "The patient has loss of vision on both eyes and a history of severe nyctalopia and macular atrophy."
	predict(test_sentence)