
# MRN: A Locally and Globally Mention-Based Reasoning Network for Document-Level Relation Extraction

ACL-IJCNLP 2021 findings paper: 
[MRN: A Locally and Globally Mention-Based Reasoning Network for Document-Level Relation Extraction](https://aclanthology.org/2021.findings-acl.117/).

This repository corresponds to the PaddlePaddle implementation of LBMRN.

## 1. Environments

- python (3.6.8)
- cuda (10.1)

## 2. Preparation

- Download [DocRED](https://github.com/thunlp/DocRED) dataset
- Put all the `train_annotated.json`, `dev.json`, `test.json`,`word2id.json`,`vec.npy`,`rel2id.json`,`ner2id` into the directory `data/`

```bash
>> python preprocess.py
```

## 3. Training

```bash
>> python main.py
```

## 4. Evaluation

```bash
>> python main.py --evaluate
```

## 5. Pretrained models

We provide a pretrained model, which can be downloaded from  [BaiduYun](https://pan.baidu.com/s/10Ic6B2ZJ6RPjh2FOSAD1-Q?pwd=0xqp),  keyword is: 0xqp. You should unzip and put it in the main directory.
