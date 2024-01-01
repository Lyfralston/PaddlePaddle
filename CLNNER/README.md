 # Unified Named Entity Recognition as Word-Word Relation Classification

AAAI 2022 paper: [Unified Named Entity Recognition as Word-Word Relation Classification](https://arxiv.org/pdf/2112.10070.pdf)
 
This repository corresponds to the PaddlePaddle implementation of CLNNER. 

## 1. Environments

```
- python (3.8.12)
- cuda (11.4)
```

## 2. Dataset

- [Conll 2003](https://www.clips.uantwerpen.be/conll2003/ner/)
- [OntoNotes 4.0](https://catalog.ldc.upenn.edu/LDC2011T03)
- [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)
- [ACE 2004](https://catalog.ldc.upenn.edu/LDC2005T09)
- [ACE 2005](https://catalog.ldc.upenn.edu/LDC2006T06)
- [GENIA](http://www.geniaproject.org/genia-corpus)
- [CADEC](https://pubmed.ncbi.nlm.nih.gov/25817970/)
- [ShARe13](https://clefehealth.imag.fr/?page_id=441)
- [ShARe14](https://sites.google.com/site/clefehealth2014/)

## 3. Preparation

- Download dataset
- Process them to fit the same format as the example in `data/`
- Put the processed data into the directory `data/`

## 4. Training

```bash
>> python main.py --config ./config/example.json
```

## 5. Evaluation

```bash
>> python main.py --ckpt [Pretrained Model Path]
```

## 6. Pretrained models

We provide a pretrained model, which can be downloaded from  [BaiduYun](https://pan.baidu.com/s/1OetSyoeCHA9WQ0aomZxR7A?pwd=hp2s),  keyword is: hp2s. You should unzip and put it in the main directory.

