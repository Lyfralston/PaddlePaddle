import paddle
import argparse
import time
import utils
import dataset
from models.HiTrans import HiTrans
from models.Loss import MultiTaskLoss
from sklearn.metrics import f1_score
import random
import numpy as np


class Trainer(object):

    def __init__(self, model):
        self.model = model
        self.emo_criterion = paddle.nn.CrossEntropyLoss()
        self.spk_criterion = paddle.nn.CrossEntropyLoss()
        self.multi_loss = MultiTaskLoss(2)
        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [{'params': [p for n, p in model.bert.named_parameters() if
            not any(nd in n for nd in no_decay)], 'lr': args.bert_lr,
            'weight_decay': args.weight_decay}, {'params': [p for n, p in
            model.bert.named_parameters() if any(nd in n for nd in no_decay
            )], 'lr': args.bert_lr, 'weight_decay': 0.0}, {'params':
            other_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': self.multi_loss.parameters(), 'lr': args.lr,
            'weight_decay': args.weight_decay}]
        self.scheduler = paddle.optimizer.lr.ExponentialDecay(gamma=args.alpha, learning_rate=args.lr)
        self.optimizer = paddle.optimizer.Adam(parameters=params, learning_rate=self.scheduler, weight_decay=args.weight_decay)

    def train(self, data_loader):
        self.model.train()
        loss_array = []
        emo_gold_array = []
        emo_pred_array = []
        for dia_input, emo_label, spk_label, cls_index, emo_mask, spk_mask in data_loader:
            dia_input = dia_input
            emo_label = emo_label
            spk_label = spk_label
            cls_index = cls_index
            emo_mask = emo_mask
            spk_mask = spk_mask
            emo_output, spk_output = self.model(dia_input, cls_index, emo_mask)
            emo_output = emo_output[emo_mask]
            emo_label = emo_label[emo_mask]
            emo_loss = self.emo_criterion(emo_output, emo_label)
            spk_output = spk_output[spk_mask]
            spk_label = spk_label[spk_mask]
            spk_loss = self.spk_criterion(spk_output, spk_label)
            loss = self.multi_loss(emo_loss, spk_loss)
            loss.backward()
            paddle.nn.utils.clip_grad_norm_(parameters=self.model.
                parameters(), max_norm=args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.clear_grad()
            loss_array.append(loss.item())
            emo_pred = paddle.argmax(x=emo_output, axis=-1)
            emo_pred_array.append(emo_pred.cpu().numpy())
            emo_gold_array.append(emo_label.cpu().numpy())
        self.scheduler.step()
        emo_gold_array = np.concatenate(emo_gold_array)
        emo_pred_array = np.concatenate(emo_pred_array)
        f1 = f1_score(emo_gold_array, emo_pred_array, average='weighted')
        loss = np.mean(loss_array)
        return loss, f1

    def eval(self, data_loader):
        self.model.eval()
        loss_array = []
        emo_gold_array = []
        emo_pred_array = []
        with paddle.no_grad():
            for dia_input, emo_label, spk_label, cls_index, emo_mask, spk_mask in data_loader:
                dia_input = dia_input
                emo_label = emo_label
                spk_label = spk_label
                cls_index = cls_index
                emo_mask = emo_mask
                spk_mask = spk_mask
                emo_output, spk_output = self.model(dia_input, cls_index,
                    emo_mask)
                emo_output = emo_output[emo_mask]
                emo_label = emo_label[emo_mask]
                emo_loss = self.emo_criterion(emo_output, emo_label)
                spk_output = spk_output[spk_mask]
                spk_label = spk_label[spk_mask]
                spk_loss = self.spk_criterion(spk_output, spk_label)
                loss = self.multi_loss(emo_loss, spk_loss)
                loss_array.append(loss.item())
                emo_pred = paddle.argmax(x=emo_output, axis=-1)
                emo_pred_array.append(emo_pred.cpu().numpy())
                emo_gold_array.append(emo_label.cpu().numpy())
        emo_gold_array = np.concatenate(emo_gold_array)
        emo_pred_array = np.concatenate(emo_pred_array)
        f1 = f1_score(emo_gold_array, emo_pred_array, average='weighted')
        loss = np.mean(loss_array)
        return loss, f1

    def save(self, path):
        paddle.save(obj=self.model.state_dict(), path=path)

    def load(self, path):
        self.model.set_state_dict(state_dict=paddle.load(path=path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--d_ff', type=int, default=768)
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--input_max_length', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--bert_lr', type=float, default=1e-05)
    parser.add_argument('--weight_decay', type=float, default=1e-05)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    logger = utils.get_logger('./log/Emory_{}.txt'.format(time.strftime(
        '%m-%d_%H-%M-%S')))
    logger.info(args)
    paddle.device.set_device(device='gpu:' + str(args.device))
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed=seed)
    paddle.seed(seed=seed)
    logger.info('Loading data...')
    (train_set, dev_set, test_set), vocab = dataset.load_data(args.
        input_max_length)
    if args.evaluate:
        dev_loader = paddle.io.DataLoader(dataset=dev_set, batch_size=args.
            batch_size, collate_fn=dataset.collate_fn)
        test_loader = paddle.io.DataLoader(dataset=test_set, batch_size=
            args.batch_size, collate_fn=dataset.collate_fn)
    else:
        train_loader = paddle.io.DataLoader(dataset=train_set, batch_size=
            args.batch_size, collate_fn=dataset.collate_fn, shuffle=True)
        dev_loader = paddle.io.DataLoader(dataset=dev_set, batch_size=args.
            batch_size, collate_fn=dataset.collate_fn)
    model = HiTrans(args.hidden_dim, len(vocab.label2id), d_model=args.
        d_model, d_ff=args.d_ff, heads=args.heads, layers=args.layers,
        dropout=args.dropout)
    trainer = Trainer(model)
    if args.evaluate:
        trainer.load('./checkpoint/model.pdparams')
        dev_loss, dev_f1 = trainer.eval(dev_loader)
        logger.info('Dev Loss: {:.4f} F1: {:.4f}'.format(dev_loss, dev_f1))
        test_loss, test_f1 = trainer.eval(test_loader)
        logger.info('Test Loss: {:.4f} F1: {:.4f}'.format(test_loss, test_f1))
    else:
        best_f1 = 0.0
        for epoch in range(args.epochs):
            train_loss, train_f1 = trainer.train(train_loader)
            logger.info('Epoch: {} Train Loss: {:.4f} F1: {:.4f}'.format(
                epoch, train_loss, train_f1))
            dev_loss, dev_f1 = trainer.eval(dev_loader)
            logger.info('Epoch: {} Dev Loss: {:.4f} F1: {:.4f}'.format(
                epoch, dev_loss, dev_f1))
            logger.info('---------------------------------')
            if best_f1 < dev_f1:
                best_f1 = dev_f1
                trainer.save('./checkpoint/model.pdparams')
        logger.info('Best Dev F1: {:.4f}'.format(best_f1))
