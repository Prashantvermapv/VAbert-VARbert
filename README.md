# VA BERT and VA-R BERT

This repository contains the code and pipelines to train named entity recognizers for inherited retinal diseases' Human Phenotype Ontology (HPO) terms in clinical letters.

<p align="center">
  <img src="./assets/pipeline.png" width="800" />
</p>

## Installation
Create a virtual environment:
```bash
conda create -n hpobert python=3.7
conda activate hpobert
```

Install dependencies:
```bash
pip install -e .
```

Note that installing torch for GPU should follow this [recommendation](https://pytorch.org/get-started/locally/) depending on which cuda version has been installed in your machine.

Download and set up data and model by running 
```bash
bash download_model.sh
```

## Training
Train with default config (defined in `config/train_config.json`)
```bash
python train.py
```

Override config's parameters by passing optional arguments
```
CUDA_VISIBLE_DEVICES=1 python train.py \
	--checkpoint 'models/name' \
	--max_epoch 150 
```

Other model/checkpoint can be from huggingface: `dmis-lab/biobert-v1.1`, `bert-base-cased`, etc.

Note: Check `config/train_config.json` before training

