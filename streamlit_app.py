import streamlit as st
from transformers import BertForTokenClassification, BertTokenizer
import torch
import numpy as np
from hpobert.dataset import HPODataset
from hpobert.utils import bio_to_entity_tokens, assign_entities
from hpobert.utils import get_html
import spacy
from spacy import displacy

data_file = 'data/meh_eyedisease.jsonl'
device = 'cpu'
model_checkpoint_path = 'models/hponer_epoch167_f1_0.7012_best/'
checkpoint = "dmis-lab/biobert-v1.1"
custom_tokenizer = 'tokenizers/super-tokenizer'

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
	"""Load model and utility variable"""
	dataset = HPODataset(data_file)   
	tag2idx, tag_values = dataset.get_tag_info()
	tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=False)
	model = BertForTokenClassification.from_pretrained(model_checkpoint_path)
	return model, tokenizer, tag_values

# Creating UI
st.title('HPOBERT App')
test_sentence = st.text_area('')
st.caption('Ex:') 
st.caption('* The patient has loss of vision on both eyes and a history of severe nyctalopia and macular atrophy.') 
st.caption('* Periarteritis nodosa and thrombotic thrombocytopenic purpura in siblings is reported. In both patients a localised serous retinal detachment and lesions of the retinal pigment epithelium had developed owing to choroidal vascular obstruction. These cases support the suggested possible relationship between the two conditions.')

# Preparing
model, tokenizer, tag_values = load_model()
tokenized_sentence = tokenizer.encode(test_sentence)
input_ids = torch.tensor([tokenized_sentence]).cpu()

# Predict
if st.button('Run'):
	model.eval()
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

	# Convert to spacy document for display
	nlp = spacy.load(custom_tokenizer)
	doc = nlp(test_sentence)

	# Convert list of BIO tags to token spans and assign them to the spacy doc
	out_spans = bio_to_entity_tokens(new_labels)
	doc = assign_entities(doc, out_spans)

	# Display
	colors = {"pnt": "#00ebc7"}
	options = {"ents": ["pnt"], "colors": colors}
	html = displacy.render(doc, style="ent", options=options)
	style = "<style>mark.entity { display: inline-block }</style>"
	st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)