import paddle
import argparse
import tqdm
import json
import numpy as np
import prettytable as pt
import itertools
from sklearn.metrics import precision_recall_fscore_support, f1_score
import config
import data_loader
import utils
from model import Model


class Trainer(object):

    def __init__(self, model):
        self.model = model
        self.criterion = paddle.nn.CrossEntropyLoss()
        # bert_params = set(self.model.bert.parameters())
        # other_params = list(set(self.model.parameters()) - bert_params)
        # no_decay = ['bias', 'LayerNorm.weight']
        # params = [{'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': config.bert_learning_rate, 'weight_decay': config.weight_decay}, 
        #           {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)], 'lr': config.bert_learning_rate, 'weight_decay': 0.0}, 
        #           {'params': other_params, 'lr': config.learning_rate, 'weight_decay': config.weight_decay}]
        self.scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=config.bert_learning_rate, warmup_steps=int(config.warm_factor * updates_total), start_lr=0, end_lr=config.bert_learning_rate)
        self.optimizer = paddle.optimizer.AdamW(learning_rate=self.scheduler, parameters=model.parameters(), weight_decay=config.weight_decay)
# >>>>>>  self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
# >>>>>>  self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=config.warm_factor * updates_total, num_training_steps=updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []
        total_i = len(data_loader)
        for i, data_batch in enumerate(data_loader):
            data_batch = [data for data in data_batch[:-1]]
            
            (bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length) = data_batch
            
            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            
            grid_mask2d = grid_mask2d.clone()
            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])
            
            loss.backward()
            paddle.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.clear_grad()

            loss_list.append(loss.cpu().item())
            
            outputs = paddle.argmax(x=outputs, axis=-1)
            
            # flatten list
            # grid_labels = grid_labels[grid_mask2d].view(-1)
            selected_elements = paddle.masked_select(grid_labels, grid_mask2d)
            grid_labels = paddle.reshape(selected_elements, shape=[-1])
            
            # outputs = outputs[grid_mask2d].view(-1)
            selected_elements = paddle.masked_select(outputs, grid_mask2d)
            outputs = paddle.reshape(selected_elements, shape=[-1])
            
            label_result.append(grid_labels)
            pred_result.append(outputs)
            
            self.scheduler.step()
            
        label_result = paddle.concat(x=label_result)
        pred_result = paddle.concat(x=pred_result)
        
        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(), pred_result.cpu().numpy(), average='macro')
        
        table = pt.PrettyTable(['Train {}'.format(epoch), 'Loss', 'F1', 'Precision', 'Recall'])
        table.add_row(['Label', '{:.4f}'.format(np.mean(loss_list))] + ['{:3.4f}'.format(x) for x in [f1, p, r]])
        
        logger.info('\n{}'.format(table))
        return f1

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()
        pred_result = []
        label_result = []
        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        with paddle.no_grad():
            for i, data_batch in enumerate(data_loader):
                entity_text = data_batch[-1]
                data_batch = [data for data in data_batch[:-1]]
                (bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length) = data_batch
                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length
                grid_mask2d = grid_mask2d.clone()
                outputs = paddle.argmax(x=outputs, axis=-1)
                ent_c, ent_p, ent_r, _ = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())
                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c
                # flatten
                # grid_labels = grid_labels[grid_mask2d].view(-1)    
                selected_elements = paddle.masked_select(grid_labels, grid_mask2d)
                grid_labels = paddle.reshape(selected_elements, shape=[-1])
                # outputs = outputs[grid_mask2d].view(-1)
                selected_elements = paddle.masked_select(outputs, grid_mask2d)
                outputs = paddle.reshape(selected_elements, shape=[-1])
                
                label_result.append(grid_labels)
                pred_result.append(outputs)
                
        label_result = paddle.concat(x=label_result)
        pred_result = paddle.concat(x=pred_result)
        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(), pred_result.cpu().numpy(), average='macro')
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)
        
        title = 'EVAL' if not is_test else 'TEST'
        logger.info('{} Label F1 {}'.format(title, f1_score(label_result.cpu().numpy(), pred_result.cpu().numpy(), average=None)))
        
        table = pt.PrettyTable(['{} {}'.format(title, epoch), 'F1', 'Precision', 'Recall'])
        table.add_row(['Label'] + ['{:3.4f}'.format(x) for x in [f1, p, r]])
        table.add_row(['Entity'] + ['{:3.4f}'.format(x) for x in [e_f1, e_p, e_r]])
        logger.info('\n{}'.format(table))
        
        return e_f1

    def predict(self, epoch, data_loader, data):
        self.model.eval()
        pred_result = []
        label_result = []
        result = []
        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        i = 0
        with paddle.no_grad():
            for i, data_batch in enumerate(data_loader):
                sentence_batch = data[i:i + config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data for data in data_batch[:-1]]
                (bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length) = data_batch
                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length
                grid_mask2d = grid_mask2d.clone()
                outputs = paddle.argmax(x=outputs, axis=-1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())
                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence['sentence']
                    instance = {'sentence': sentence, 'entity': []}
                    for ent in ent_list:
                        flag = False
                        for x in ent[0]:
                            if x >= len(sentence):
                                flag = True
                                break
                        if flag:
                            continue
                        instance['entity'].append({'text': [sentence[x] for x in ent[0]], 'offset': [x for x in ent[0]], 'type': config.vocab.id_to_label(ent[1])})
                    result.append(instance)
                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c
                # flatten
                # grid_labels = grid_labels[grid_mask2d].view(-1)
                selected_elements = paddle.masked_select(grid_labels, grid_mask2d)
                grid_labels = paddle.reshape(selected_elements, shape=[-1])
                # outputs = outputs[grid_mask2d].view(-1)
                selected_elements = paddle.masked_select(outputs, grid_mask2d)
                outputs = paddle.reshape(selected_elements, shape=[-1])
                
                label_result.append(grid_labels)
                pred_result.append(outputs)
                i += config.batch_size
        label_result = paddle.concat(x=label_result)
        pred_result = paddle.concat(x=pred_result)
        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(), pred_result.cpu().numpy(), average='macro')
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)
        title = 'TEST'
        logger.info('{} Label F1 {}'.format('TEST', f1_score(label_result.cpu().numpy(), pred_result.cpu().numpy(), average=None)))
        table = pt.PrettyTable(['{} {}'.format(title, epoch), 'F1', 'Precision', 'Recall'])
        table.add_row(['Label'] + ['{:3.4f}'.format(x) for x in [f1, p, r]])
        table.add_row(['Entity'] + ['{:3.4f}'.format(x) for x in [e_f1, e_p, e_r]])
        logger.info('\n{}'.format(table))
        with open(config.predict_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
        return e_f1

    def save(self, path):
        paddle.save(obj=self.model.state_dict(), path=path)

    def load(self, path):
        self.model.set_state_dict(state_dict=paddle.load(path=path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/track2.json')
    parser.add_argument('--save_path', type=str, default='./model.pdparams')
    parser.add_argument('--predict_path', type=str, default='./output.json')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--device', type=int, default=7)
    
    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)
    parser.add_argument('--dilation', type=str, help='e.g. 1,2,3')
    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)
    parser.add_argument('--use_bert_last_4_layers', type=int, help='1: true, 0: false')
    parser.add_argument('--seed', type=int)
    
    args = parser.parse_args()
    config = config.Config(args)
    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger
    paddle.device.set_device(device='gpu:' + str(args.device))
    
    logger.info('Loading Data')
    datasets, ori_data = data_loader.load_data_bert(config)
    train_loader, dev_loader, test_loader = (paddle.io.DataLoader(dataset=dataset, batch_size=config.batch_size, 
                                                                               collate_fn=data_loader.collate_fn, shuffle=i == 0, 
                                                                               num_workers=0, drop_last=i ==0) for i, dataset in enumerate(datasets))
    updates_total = len(datasets[0]) // config.batch_size * config.epochs
    logger.info('Building Model')
    model = Model(config)
    trainer = Trainer(model)
    if config.ckpt == '':
        best_f1 = 0
        best_test_f1 = 0
        for i in range(config.epochs):
            logger.info('Epoch: {}'.format(i))
            trainer.train(i, train_loader)
            f1 = trainer.eval(i, dev_loader)
            test_f1 = trainer.eval(i, test_loader, is_test=True)
            if f1 > best_f1:
                best_f1 = f1
                best_test_f1 = test_f1
                trainer.save(config.save_path)
        logger.info('Best DEV F1: {:3.4f}'.format(best_f1))
        logger.info('Best TEST F1: {:3.4f}'.format(best_test_f1))
        logger.info(str(config.vocab.label2id))
        trainer.load(config.save_path)
        trainer.predict('Final', test_loader, ori_data[-1])
    else:
        logger.info('Loading checkpoint...')
        trainer.load(config.ckpt)
        logger.info('Loaded!')
        trainer.predict('Final', test_loader, ori_data[-1])
