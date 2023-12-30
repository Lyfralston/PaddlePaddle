import paddle
import paddlenlp
import numpy as np
import json
from _collections import defaultdict


def paddle_pad_sequence(seqs, batch_first=False):
    max_length = max(len(seq) for seq in seqs)
    seqs = [seq[:max_length] for seq in seqs]
    for i in range(len(seqs)):
        pad_length = max_length - len(seqs[i])
        if pad_length > 0:
            seqs[i] = paddle.concat([seqs[i], paddle.zeros([pad_length], dtype=seqs[i].dtype)], axis=0)
    seqs = paddle.stack(seqs) if batch_first else paddle.stack(seqs, axis=1)
    return seqs


class Vocabulary(object):

    def __init__(self):
        self.label2id = {'Neutral': 0, 'Joyful': 1, 'Peaceful': 2, 'Powerful':
            3, 'Scared': 4, 'Mad': 5, 'Sad': 6}
        self.id2label = {value: key for key, value in self.label2id.items()}

    def num_labels(self):
        return len(self.label2id)


def collate_fn(data):
    dia_input, emo_label, spk_label, cls_index = map(list, zip(*data))
    batch_size = len(dia_input)
    max_dia_len = np.max([x.shape[0] for x in emo_label])
    emo_mask = paddle.zeros(shape=(batch_size, max_dia_len), dtype='bool')
    spk_mask = paddle.zeros(shape=(batch_size, max_dia_len, max_dia_len),
        dtype='bool')
    _spk_label = paddle.zeros(shape=(batch_size, max_dia_len, max_dia_len),
        dtype='int64')
    for i, (e, s) in enumerate(zip(emo_label, spk_label)):
        emo_mask[i, :e.shape[0]] = True
        spk_mask[i, :s.shape[0], :s.shape[1]] = True
        _spk_label[i, :s.shape[0], :s.shape[1]] = s
    spk_label = _spk_label
    
    max_length = max(len(dia) for dia in dia_input)
    dia_input = [dia[:max_length] for dia in dia_input]
    for i in range(len(dia_input)):
        pad_length = max_length - len(dia_input[i])
        if pad_length > 0:
            dia_input[i] = paddle.concat([dia_input[i], paddle.zeros([pad_length], dtype=dia_input[i].dtype)], axis=0)
    dia_input = paddle.stack(dia_input)
    
    max_length = max(len(emo) for emo in emo_label)
    emo_label = [emo[:max_length] for emo in emo_label]
    for i in range(len(emo_label)):
        pad_length = max_length - len(emo_label[i])
        if pad_length > 0:
            emo_label[i] = paddle.concat([emo_label[i], paddle.zeros([pad_length], dtype=emo_label[i].dtype)], axis=0)
    emo_label = paddle.stack(emo_label)
    
    max_length = max(len(cls) for cls in cls_index)
    cls_index = [cls[:max_length] for cls in cls_index]
    for i in range(len(cls_index)):
        pad_length = max_length - len(cls_index[i])
        if pad_length > 0:
            cls_index[i] = paddle.concat([cls_index[i], paddle.zeros([pad_length], dtype=cls_index[i].dtype)], axis=0)
    cls_index = paddle.stack(cls_index)

    return dia_input, emo_label, spk_label, cls_index, emo_mask, spk_mask


class MyDataset(paddle.io.Dataset):

    def __init__(self, dia_input, emo_label, spk_label, cls_index):
        self.dia_input = dia_input
        self.emo_label = emo_label
        self.spk_label = spk_label
        self.cls_index = cls_index

    def __getitem__(self, item):
        return paddle.to_tensor(data=self.dia_input[item], dtype='int64'
            ), paddle.to_tensor(data=self.emo_label[item], dtype='int64'
            ), paddle.to_tensor(data=self.spk_label[item], dtype='int64'
            ), paddle.to_tensor(data=self.cls_index[item], dtype='int64')

    def __len__(self):
        return len(self.dia_input)


def load_data(input_max):
    tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab = Vocabulary()

    def processing_data(path):
        with open(path) as fp:
            data = json.load(fp)
        dia_dict = defaultdict(list)
        for episode in data['episodes']:
            for scene in episode['scenes']:
                for utt in scene['utterances']:
                    utt_data = {
                        'utterance': utt['transcript'],
                        'emotion': utt['emotion'],
                        'speaker': utt['speakers'][0]
                    }
                    dia_dict[scene['scene_id']].append(utt_data)
        dia_input = []
        cls_index = []
        emo_label = []
        spk_label = []
        for dia in dia_dict.values():
            _dia_input = [tokenizer.encode(x['utterance'], return_dict=False)['input_ids'][:-1] for x in dia]
            _len_index = np.cumsum([len(x) for x in _dia_input])
            chucks = 1
            for i in range(len(_len_index) - 1):
                start = _len_index[i]
                end = _len_index[i + 1]
                if start < input_max * chucks < end:
                    _dia_input[i] += [tokenizer.pad_token_id] * (input_max -
                        start)
                    _len_index = np.cumsum([len(x) for x in _dia_input])
                    chucks += 1
            _dia_input = [x for utt in _dia_input for x in utt] + [tokenizer.sep_token_id]
            _cls_index = [i for i, x in enumerate(_dia_input) if x == tokenizer.cls_token_id]
            _emo_label = [vocab.label2id[x['emotion']] for x in dia]
            assert len(_emo_label) == len(_cls_index)
            _spk_label = np.zeros((len(dia), len(dia)), dtype=np.long)
            for i in range(len(dia)):
                for j in range(len(dia)):
                    _spk_label[i, j] = dia[i]['speaker'] == dia[j]['speaker']
            dia_input.append(_dia_input)
            emo_label.append(_emo_label)
            spk_label.append(_spk_label)
            cls_index.append(_cls_index)
        return MyDataset(dia_input, emo_label, spk_label, cls_index)
    return (processing_data('./data/emotion-detection-emotion-detection-1.0/json/emotion-detection-trn.json'), 
            processing_data('./data/emotion-detection-emotion-detection-1.0/json/emotion-detection-dev.json'), 
            processing_data('./data/emotion-detection-emotion-detection-1.0/json/emotion-detection-tst.json')
            ), vocab
