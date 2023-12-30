import sys
sys.path.append('/home/lyf/paddle/MRN-pd/utils')
import paddle_aux
import json
import tqdm
import numpy as np
fact_in_train = set([])
for file in ['train', 'dev', 'test']:
    with open('./data/{}.json'.format(file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    for doc in tqdm.tqdm(data):
        sen_len_list = np.cumsum(np.array([0] + [len(sen) for sen in doc[
            'sents']]))
        doc['sents'] = [x for sen in doc['sents'] for x in sen]
        mention_list = []
        for i, entity in enumerate(doc['vertexSet']):
            for mention in entity:
                mention['pos'][0] = int(mention['pos'][0]) + int(sen_len_list
                    [int(mention['sent_id'])])
                mention['pos'][1] = int(mention['pos'][1]) + int(sen_len_list
                    [int(mention['sent_id'])])
                mention['id'] = len(mention_list)
                mention_list.append((i, len(mention_list), mention['pos'][0
                    ], mention['pos'][1], mention['sent_id']))
        dt = np.dtype([('id', int), ('ent_id', int), ('pos_s', int), (
            'pos_e', int), ('sent_id', int)])
        mention_list = np.array(mention_list, dtype=dt)
        mention_list = np.sort(mention_list, order='pos_s')
        mentions = [{'id': int(m), 'ent_id': int(i), 'pos': [int(s), int(e)
            ], 'sent_id': int(sen)} for i, m, s, e, sen in mention_list]
        doc['mentions'] = mentions
        train_triple = set()
        labels = doc.get('labels', [])
        new_labels = []
        vertexSet = doc['vertexSet']
        for label in labels:
            rel = label['r']
            train_triple.add((label['h'], label['t']))
            if file == 'train':
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_train.add((n1['name'], n2['name'], rel))
            else:
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        if (n1['name'], n2['name'], rel) in fact_in_train:
                            label['intrain'] = True
                        else:
                            label['intrain'] = False
            new_labels.append(label)
        doc['labels'] = new_labels
    with open('./data/docred_pre_{}.json'.format(file), 'w', encoding='utf-8'
        ) as f:
        json.dump(data, f, indent=4)
