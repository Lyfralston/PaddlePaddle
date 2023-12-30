import paddle
import json
import numpy as np
import pandas as pd
import logging
from collections import Counter, OrderedDict, defaultdict
RELATION_NUM = 97
dis2idx = np.zeros(1000, dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


def get_logger(pathname):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'

    def __init__(self, rel2id, ner2id, frequency=0):
        self.token2id = {self.PAD: 0, self.UNK: 1}
        self.id2token = {(0): self.PAD, (1): self.UNK}
        self.token2count = {self.PAD: 1000, self.UNK: 1000}
        self.frequency = frequency
        self.rel2id = rel2id
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.ner2id = ner2id

    def add_token(self, token):
        token = token.lower()
        if token in self.token2id:
            self.token2count[token] += 1
        else:
            self.token2id[token] = len(self.token2id)
            self.id2token[self.token2id[token]] = token
            self.token2count[token] = 1
        assert token == self.id2token[self.token2id[token]]

    def remove_low_frequency_token(self):
        new_token2id = {self.PAD: 0, self.UNK: 1}
        new_id2token = {(0): self.PAD, (1): self.UNK}
        for token in self.token2id:
            if self.token2count[token
                ] > self.frequency and token not in new_token2id:
                new_token2id[token] = len(new_token2id)
                new_id2token[new_token2id[token]] = token
        self.token2id = new_token2id
        self.id2token = new_id2token

    def __len__(self):
        return len(self.token2id)

    def encode(self, text):
        return [self.token2id.get(x.lower(), 1) for x in text]

    def decode(self, ids):
        return [self.id2token.get(x) for x in ids]


def collate_fn(data):
    (doc_inputs, ner_inputs, dis_inputs, ref_inputs, rel_labels, intra_mask,
        inter_mask, intrain_mask, doc2ent_mask, doc2men_mask, men2ent_mask,
        ent2ent_mask, men2men_mask, title) = map(list, zip(*data))
    batch_size = len(doc_inputs)
    max_tok = np.max([x.shape[0] for x in doc_inputs])
# >>>>>>    doc_inputs = torch.nn.utils.rnn.pad_sequence(doc_inputs, True)
# >>>>>>    ner_inputs = torch.nn.utils.rnn.pad_sequence(ner_inputs, True)
    ent_num = [x.shape[0] for x in doc2ent_mask]
    men_num = [x.shape[0] for x in doc2men_mask]
    max_ent = np.max(ent_num)
    max_men = np.max(men_num)
    
    max_length = max(len(doc) for doc in doc_inputs)
    doc_inputs = [doc[:max_length] for doc in doc_inputs]
    for i in range(len(doc_inputs)):
        pad_length = max_length - len(doc_inputs[i])
        if pad_length > 0:
            doc_inputs[i] = paddle.concat([doc_inputs[i], paddle.zeros([pad_length], dtype=doc_inputs[i].dtype)], axis=0)
    doc_inputs = paddle.stack(doc_inputs)
    
    max_length = max(len(ner) for ner in ner_inputs)
    ner_inputs = [ner[:max_length] for ner in ner_inputs]
    for i in range(len(ner_inputs)):
        pad_length = max_length - len(ner_inputs[i])
        if pad_length > 0:
            ner_inputs[i] = paddle.concat([ner_inputs[i], paddle.zeros([pad_length], dtype=ner_inputs[i].dtype)], axis=0)
    ner_inputs = paddle.stack(ner_inputs)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data
    dis_mat = paddle.zeros(shape=(batch_size, max_men, max_men), dtype='int64')
    dis_inputs = fill(dis_inputs, dis_mat)
    ref_mat = paddle.zeros(shape=(batch_size, max_men, max_men), dtype='int64')
    ref_inputs = fill(ref_inputs, ref_mat)
    d2e_mat = paddle.zeros(shape=(batch_size, max_ent, max_tok), dtype='bool')
    doc2ent_mask = fill(doc2ent_mask, d2e_mat)
    d2m_mat = paddle.zeros(shape=(batch_size, max_men, max_tok), dtype='bool')
    doc2men_mask = fill(doc2men_mask, d2m_mat)
    m2e_mat = paddle.zeros(shape=(batch_size, max_ent, max_men), dtype='bool')
    men2ent_mask = fill(men2ent_mask, m2e_mat)
    e2e_mat = paddle.zeros(shape=(batch_size, max_ent, max_ent), dtype='bool')
    ent2ent_mask = fill(ent2ent_mask, e2e_mat)
    m2m_mat = paddle.zeros(shape=(batch_size, max_men, max_men), dtype='bool')
    men2men_mask = fill(men2men_mask, m2m_mat)
    rel_mat = paddle.zeros(shape=(batch_size, max_ent, max_ent,
        RELATION_NUM), dtype='float32')
    for j, x in enumerate(rel_labels):
        rel_mat[j, :x.shape[0], :x.shape[1], :] = x
    rel_labels = rel_mat
    intrain_mat = paddle.zeros(shape=(batch_size, max_ent, max_ent,
        RELATION_NUM), dtype='bool')
    for j, x in enumerate(intrain_mask):
        intrain_mat[j, :x.shape[0], :x.shape[1], :] = x
    intrain_mask = intrain_mat
    intra_mat = paddle.zeros(shape=(batch_size, max_ent, max_ent,
        RELATION_NUM), dtype='bool')
    for j, x in enumerate(intra_mask):
        intra_mat[j, :x.shape[0], :x.shape[1], :] = x
    intra_mask = intra_mat
    inter_mat = paddle.zeros(shape=(batch_size, max_ent, max_ent,
        RELATION_NUM), dtype='bool')
    for j, x in enumerate(inter_mask):
        inter_mat[j, :x.shape[0], :x.shape[1], :] = x
    inter_mask = inter_mat
    return (doc_inputs, ner_inputs, dis_inputs, ref_inputs, rel_labels,
        intra_mask, inter_mask, intrain_mask, doc2ent_mask, doc2men_mask,
        men2ent_mask, ent2ent_mask, men2men_mask, title)


class RelationDataset(paddle.io.Dataset):

    def __init__(self, doc_inputs, ner_inputs, dis_inputs, ref_inputs,
        rel_labels, intra_mask, inter_mask, intrain_mask, doc2ent_mask,
        doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask, title):
        self.doc_inputs = doc_inputs
        self.ner_inputs = ner_inputs
        self.dis_inputs = dis_inputs
        self.ref_inputs = ref_inputs
        self.rel_labels = rel_labels
        self.intra_mask = intra_mask
        self.inter_mask = inter_mask
        self.intrain_mask = intrain_mask
        self.doc2ent_mask = doc2ent_mask
        self.doc2men_mask = doc2men_mask
        self.men2ent_mask = men2ent_mask
        self.ent2ent_mask = ent2ent_mask
        self.men2men_mask = men2men_mask
        self.title = title

    def __getitem__(self, item):
        return paddle.to_tensor(data=self.doc_inputs[item], dtype='int64'
            ), paddle.to_tensor(data=self.ner_inputs[item], dtype='int64'
            ), paddle.to_tensor(data=self.dis_inputs[item], dtype='int64'
            ), paddle.to_tensor(data=self.ref_inputs[item], dtype='int64'
            ), paddle.to_tensor(data=self.rel_labels[item], dtype='float32'
            ), paddle.to_tensor(data=self.intra_mask[item], dtype='bool'
            ), paddle.to_tensor(data=self.inter_mask[item], dtype='bool'
            ), paddle.to_tensor(data=self.intrain_mask[item], dtype='bool'
            ), paddle.to_tensor(data=self.doc2ent_mask[item], dtype='bool'
            ), paddle.to_tensor(data=self.doc2men_mask[item], dtype='bool'
            ), paddle.to_tensor(data=self.men2ent_mask[item], dtype='bool'
            ), paddle.to_tensor(data=self.ent2ent_mask[item], dtype='bool'
            ), paddle.to_tensor(data=self.men2men_mask[item], dtype='bool'
            ), self.title[item]

    def __len__(self):
        return len(self.doc_inputs)


def process(data, vocab, is_train):
    doc_inputs = []
    ner_inputs = []
    dis_inputs = []
    ref_inputs = []
    rel_labels = []
    intra_mask = []
    inter_mask = []
    intrain_mask = []
    doc2ent_mask = []
    doc2men_mask = []
    men2ent_mask = []
    men2men_mask = []
    ent2ent_mask = []
    title = []
    for index, doc in enumerate(data):
        _doc_inputs = vocab.encode(doc['sents'])
        men_num = len(doc['mentions'])
        ent_num = len(doc['vertexSet'])
        doc_len = len(_doc_inputs)
        _ner_inputs = np.zeros((doc_len,), dtype=np.int)
        _dis_inputs = np.zeros((men_num, men_num), dtype=np.int)
        _ref_inputs = np.zeros((men_num, men_num), dtype=np.int)
        _rel_labels = np.zeros((ent_num, ent_num, RELATION_NUM), dtype=np.int)
        _intra_mask = np.zeros((ent_num, ent_num, RELATION_NUM), dtype=np.bool)
        _inter_mask = np.ones((ent_num, ent_num, RELATION_NUM), dtype=np.bool)
        _intrain_mask = np.zeros((ent_num, ent_num, RELATION_NUM), dtype=np
            .bool)
        _doc2ent_mask = np.zeros((ent_num, doc_len), dtype=np.bool)
        _doc2men_mask = np.zeros((men_num, doc_len), dtype=np.bool)
        _men2ent_mask = np.zeros((ent_num, men_num), dtype=np.bool)
        _ent2ent_mask = np.ones((ent_num, ent_num), dtype=np.bool)
        _men2men_mask = np.ones((men_num, men_num), dtype=np.bool)
        men_id = 0
        for ent_id, entity in enumerate(doc['vertexSet']):
            for mention in entity:
                s_pos, e_pos = mention['pos']
                ner_id = vocab.ner2id[mention['type']]
                _ner_inputs[s_pos:e_pos] = ner_id
                _doc2ent_mask[ent_id][s_pos:e_pos] = 1
                _doc2men_mask[men_id][s_pos:e_pos] = 1
                men_id += 1
        for i, ent1 in enumerate(doc['vertexSet']):
            for j, ent2 in enumerate(doc['vertexSet']):
                for men1 in ent1:
                    for men2 in ent2:
                        if men1['sent_id'] == men2['sent_id']:
                            _intra_mask[i, j] = True
                            _inter_mask[i, j] = False
        for i, men in enumerate(doc['mentions']):
            pos = men['pos'][0]
            ent_id = men['ent_id']
            _dis_inputs[i, :] += pos
            _dis_inputs[:, i] -= pos
            _men2ent_mask[ent_id, i] = 1
        _doc2men_mask = _doc2men_mask[np.array([x['id'] for x in doc[
            'mentions']])]
        for i in range(men_num):
            for j in range(men_num):
                if _dis_inputs[i, j] < 0:
                    _dis_inputs[i, j] = dis2idx[-_dis_inputs[i, j]] + 9
                else:
                    _dis_inputs[i, j] = dis2idx[_dis_inputs[i, j]]
                if doc['mentions'][i]['sent_id'] == doc['mentions'][j][
                    'sent_id']:
                    _ref_inputs[i, j] = 1
                    _ref_inputs[j, i] = 1
        _dis_inputs[_dis_inputs == 0] = 19
        _rel_labels[..., 0] = 1
        for label in doc.get('labels', []):
            h, t, r = label['h'], label['t'], vocab.rel2id[label['r']]
            _rel_labels[h, t, 0] = 0
            _rel_labels[h, t, r] = 1
            if not is_train and label['intrain']:
                _intrain_mask[h, t, r] = 1
        doc_inputs.append(_doc_inputs)
        ner_inputs.append(_ner_inputs)
        dis_inputs.append(_dis_inputs)
        ref_inputs.append(_ref_inputs)
        rel_labels.append(_rel_labels)
        intra_mask.append(_intra_mask)
        inter_mask.append(_inter_mask)
        intrain_mask.append(_intrain_mask)
        doc2ent_mask.append(_doc2ent_mask)
        doc2men_mask.append(_doc2men_mask)
        men2ent_mask.append(_men2ent_mask)
        ent2ent_mask.append(_ent2ent_mask)
        men2men_mask.append(_men2men_mask)
        title.append(doc['title'])
    return (doc_inputs, ner_inputs, dis_inputs, ref_inputs, rel_labels,
        intra_mask, inter_mask, intrain_mask, doc2ent_mask, doc2men_mask,
        men2ent_mask, ent2ent_mask, men2men_mask, title)


def load_data(load_emb=True):
    with open('./data/docred_pre_train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/docred_pre_dev.json', 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/docred_pre_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    with open('./data/rel2id.json', 'r', encoding='utf-8') as f:
        rel2id = json.load(f)
    with open('./data/ner2id.json', 'r', encoding='utf-8') as f:
        ner2id = json.load(f)
    vocab = Vocabulary(rel2id, ner2id)
    if load_emb:
        with open('./data/DocRED_baseline_metadata/word2id.json', 'r',
            encoding='utf-8') as f:
            token2id = json.load(f)
            vocab.token2id = token2id
            vocab.id2token = {v: k for k, v in token2id.items()}
    else:
        for doc in train_data:
            for sent in doc['sents']:
                for token in sent:
                    vocab.add_token(token)
    train_dataset = RelationDataset(*process(train_data, vocab, is_train=True))
    dev_dataset = RelationDataset(*process(dev_data, vocab, is_train=False))
    test_dataset = RelationDataset(*process(test_data, vocab, is_train=False))
    return train_dataset, dev_dataset, test_dataset, vocab


def load_embedding():
    embedding = np.load('./data/DocRED_baseline_metadata/vec.npy')
    embedding = paddle.to_tensor(data=embedding, dtype='float32')
    return embedding
