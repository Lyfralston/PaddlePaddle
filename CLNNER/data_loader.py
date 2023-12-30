import paddle
import paddlenlp
import json
import numpy as np
import prettytable as pt
from gensim.models import KeyedVectors
import os
import utils
import requests
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {(0): self.PAD, (1): self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label
        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


def pad_sequence(sequences, batch_first=True, padding_value=0):
    max_len = max(len(seq) for seq in sequences)
    if batch_first:
        padded_sequences = [paddle.concat([seq, paddle.full([max_len - len(seq)], padding_value, dtype='int64')], axis=0) for seq in sequences]
    else:
        padded_sequences = [paddle.concat([paddle.full([max_len - len(seq)], padding_value, dtype='int64'), seq], axis=0) for seq in sequences]
    return paddle.to_tensor(padded_sequences)


def collate_fn(data):
    (bert_inputs, grid_labels, grid_mask2d, pieces2word, sent_length, entity_text) = map(list, zip(*data))
    max_tok = np.max(sent_length)
    sent_length = paddle.to_tensor(data=sent_length, dtype='int64')
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs)
# >>>>>>    bert_inputs = torch.nn.utils.rnn.pad_sequence(bert_inputs, True)  # modified!
    batch_size = bert_inputs.shape[0]

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data
    
    labels_mat = paddle.zeros(shape=(batch_size, max_tok, max_tok), dtype='int64')
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = paddle.zeros(shape=(batch_size, max_tok, max_tok), dtype='bool')
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = paddle.zeros(shape=(batch_size, max_tok, max_pie), dtype='bool')
    pieces2word = fill(pieces2word, sub_mat)
    return (bert_inputs, grid_labels, grid_mask2d, pieces2word, sent_length, entity_text)


class RelationDataset(paddle.io.Dataset):

    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item):
        return paddle.to_tensor(data=self.bert_inputs[item], dtype='int64'), \
               paddle.to_tensor(data=self.grid_labels[item], dtype='int64'), \
               paddle.to_tensor(data=self.grid_mask2d[item], dtype='bool'), \
               paddle.to_tensor(data=self.pieces2word[item], dtype='bool'), \
               self.sent_length[item], self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)


def process_bert(data, tokenizer, vocab):
    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    entity_text = []
    pieces2word = []
    sent_length = []
    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0:
            continue
        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
        length = len(instance['sentence'])
        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)
        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)
        for entity in instance['ner']:
            index = entity['index']
            for i in range(len(index)):
                if i + 1 >= len(index):
                    break
                _grid_labels[index[i], index[i + 1]] = 1
            _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity['type'])
        _entity_text = set([utils.convert_index_to_text(e['index'], vocab.label_to_id(e['type'])) for e in instance['ner']])
        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)
    return (bert_inputs, grid_labels, grid_mask2d, pieces2word, sent_length, entity_text)


def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        for entity in instance['ner']:
            vocab.add_label(entity['type'])
        entity_num += len(instance['ner'])
    return entity_num


def load_data_bert(config):
    with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained(config.bert_name)
# >>>>>>    tokenizer = transformers.AutoTokenizer.from_pretrained(config.bert_name, cache_dir='./cache/')  # modified
    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)
    test_ent_num = fill_vocab(vocab, test_data)
    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    table.add_row(['test', len(test_data), test_ent_num])
    config.logger.info('\n{}'.format(table))
    config.label_num = len(vocab.label2id)
    config.vocab = vocab
    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data, test_data)
