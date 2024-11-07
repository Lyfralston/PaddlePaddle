

## Environments

1. Please install the following packages from both conda and pip.

```
conda install
  - python 3.8
  - numpy 1.24.3
  - ipdb 0.13.13
  - tqdm 4.65.0
  - beautifulsoup4 4.11.1
  - lxml 4.9.1
  - jsonlines 3.1.0
  - jsonnet 0.20.0
  - stanza=1.5.0
```
```
pip install
  - sentencepiece 0.1.96
  - scipy 1.5.4
  - spacy 3.1.4
  - nltk 3.8.1
  - tensorboardX 2.6
  - keras-preprocessing 1.1.2
  - keras 2.4.3
  - dgl-cu111 0.6.1
  - amrlib 0.7.1
  - cached_property 1.5.2
  - typing-extensions 4.4.0
  - penman==1.2.2
```
   
  Alternatively, you can use the following command.
```
conda env create -f env.yml
```

2. Run the following command.
```
python -m spacy download en_core_web_lg
```

## Running

### Training
```
./scripts/train.sh [config]
```
### Evaluation

```
python TextEE/evaluate_end2end.py --task EAE --data [eval_data] --model [saved_model_folder]
```

### Making Predictions for New Texts with End-to-End Model

```
# Predicting End-to-End
python TextEE/predict_end2end.py --input_file demo_input.txt --model [saved_model_folder] --output_file demo_output.json
```

## Pretrained models

We provide a pretrained model through the following command:
```bash
>>> . scripts/train.sh config/ace05-en/EEQA_EAE_ace05-en_roberta-large.jsonnet
```
It can be downloaded from  [BaiduYun](https://pan.baidu.com/s/1zojRMXH0fL90NYtY0lggJw?pwd=qdfm),  keyword is: qdfm. You should unzip and put it in the main directory.


