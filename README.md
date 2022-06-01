# HPOBERT
[**Installation**](#installation) | 
[**Preprocessing**](#preprocessing-and-annotations) | [**Training**](#Training) | [**Inference**](#inference) | 
[**References**](#reference) | [**Contact**](#contact)

This repository contains the code and pipelines to train named entity recognizers for inherited retinal diseases' Human Phenotype Ontology (HPO) terms in clinical letters.

<p align="center">
  <img src="./assets/pipeline-overall.png" width="800" />
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

## Preprocessing and annotation
Convert brat annotationt to character offset format which will be used by training script
```bash
process_brat.py --src 'data/hpo/letter/' --dest 'data/jsonl/'
```
For every pair of (`X.txt`, `X.ann`) files in source folder, one corresponding `X.jsonl` file will be produced. The `all.jsonl` stores all the annotations from source folder as concatenation of all `X.jsonl`'s. The `all.jsonl` then can be fed into the training script.

To see how data is transformed in more details, check `notebooks/process_brat.ipynb`

To gain access to brat annotation files, please contact the authors (see `Contact` section). If you already had `all.jsonl` file, please skip this step.

**Run brat to visualize annotations**

Assuming that your data is stored in `data/hpo_ann` and brat config `config/brat`, you cna start brat docker by:

```bash
docker run --name=brat -d -p 80:80 \
	-v $(pwd)/data/hpo_ann/letters:/bratdata \
	-v $(pwd)/config/brat:/bratcfg \
	-e BRAT_USERNAME=brat -e BRAT_PASSWORD=brat \
	-e BRAT_EMAIL=brat@example.com cassj/brat
```

The above command maps host's data folder (`./data/hpo_ann`) to docker's (`/bratdata`), likewise the brat configuration. For more details, please refer to https://github.com/nlplab/brat. Note that to edit annotation, you need to copy 2 configuration files `config/brat/annotation.conf` and `config/brat/visual.conf` to host's data folder.

When the docker start running, open a browser and goes to `localhost`, the brat visualizing tool should appear.

## Training
Train with default config (defined in `config/train_config.json`)
```bash
python train.py
```

Override config's parameters by passing optional arguments
```
CUDA_VISIBLE_DEVICES=1 python train.py \
	--checkpoint 'models/eyebertv3' \
	--max_epoch 150 
```

Other model/checkpoint can be from huggingface: `dmis-lab/biobert-v1.1`, `bert-base-cased`, etc.

Note: Check `config/train_config.json` before training

## Inference
```bash
python predict.py 'your sentence'
```

Web app
```bash
streamlit run streamlit_app.py
```

## Read more
- progress update ([slides](https://docs.google.com/presentation/d/1MNFCnlRZtXzDiqqk-l1foeoqscfvxyDmdtCvxiEh3LU/edit?usp=sharing))
- training report (comming soon)
- project description (slides) (comming soon)

## References
- Pontikos lab's website: https://pontikoslab.com 
- HPO website: https://hpo.jax.org
- Brat annotation tool: [source](https://github.com/nlplab/brat), [docker](https://github.com/cassj/brat-docker)
- Huggingface's transformer: https://huggingface.co/docs/transformers/index

## Contact
* Quang Nguyen <quang.nguyen.21@ucl.ac.uk>
* Nikolas Pontikos <n.pontikos@ucl.ac.uk>