# HiTrans: A Transformer-Based Context- and Speaker-Sensitive Model for Emotion Detection in Conversations

COLING 2020 paper: [HiTrans: A Transformer-Based Context- and Speaker-Sensitive Model for Emotion Detection in Conversations](https://www.aclweb.org/anthology/2020.coling-main.370/)

This repository corresponds to the PaddlePaddle implementation of HiTrans.

## 1. Environments

- python (3.6.8)
- cuda (10.1)

## 2. Preparation

- Download [MELD](https://github.com/declare-lab/MELD) dataset
- Put `train_sent_emo.csv`, `dev_sent_emo.csv` and `test_sent_emo.csv` into the directory `data/`

## 3. Training

```bash
>> python main.py
```

## 4. Evaluation

```bash
>> python main.py --evaluate
```

## 5. Pretrained models

We provide a pretrained model, which can be downloaded from  [BaiduYun](https://pan.baidu.com/s/10w7fi5JOjSaMyB0GuIfUfg?pwd=ib8x),  keyword is: ib8x. You should unzip and put it in the main directory.
