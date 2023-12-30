import sys
sys.path.append('/home/lyf/paddle/MRN-pd/utils')
import paddle_aux
import paddle
import argparse
import json
import time
import numpy as np
from sklearn.metrics import auc
import utils
from model import MRN


class AsymmetricLossOptimized(paddle.nn.Layer):
    """ Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations"""

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-08,
        disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        (self.targets) = (self.anti_targets) = (self.xs_pos) = (self.xs_neg
            ) = (self.asymmetric_w) = (self.loss) = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        self.targets = y
        self.anti_targets = 1 - y
        self.xs_pos = paddle.nn.functional.sigmoid(x=x)
        self.xs_neg = 1.0 - self.xs_pos
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(y=paddle.to_tensor(self.clip)).clip_(max=1)
        self.loss = self.targets * paddle.log(x=self.xs_pos.clip(min=self.eps))
        self.loss.add_(y=paddle.to_tensor(self.anti_targets * paddle.log(x=
            self.xs_neg.clip(min=self.eps))))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = paddle.pow(x=1 - self.xs_pos - self.xs_neg,
                y=self.gamma_pos * self.targets + self.gamma_neg * self.
                anti_targets)
            self.loss *= self.asymmetric_w
        return -self.loss.sum()


class Trainer(object):

    def __init__(self, model):
        self.model = model
        self.criterion = AsymmetricLossOptimized(gamma_neg=3)
        self.scheduler = paddle.optimizer.lr.StepDecay(step_size=10, gamma=args.gamma, learning_rate=args.lr)
        self.optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=self.scheduler, weight_decay=args.wd)

    def train(self, data_loader):
        self.model.train()
        loss_list = []
        for i, data_batch in enumerate(data_loader):
            if args.use_gpu:
                data_batch = [data for data in data_batch[:-1]]
            (doc_inputs, ner_inputs, dis_inputs, ref_inputs, rel_labels, _,
                _, _, doc2ent_mask, doc2men_mask, men2ent_mask,
                ent2ent_mask, men2men_mask) = data_batch
            outputs = model(doc_inputs, ner_inputs, dis_inputs, ref_inputs,
                doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask,
                men2men_mask)
            rel_mask = ent2ent_mask.clone()
            rel_mask[:, range(rel_mask.shape[1]), range(rel_mask.shape[2])] = 0
            loss = self.criterion(outputs[rel_mask], rel_labels[rel_mask])
            self.optimizer.clear_grad()
            loss.backward()
            paddle.nn.utils.clip_grad_norm_(parameters=self.model.
                parameters(), max_norm=1.0)
            self.optimizer.step()
            loss_list.append(loss.cpu().item())
        self.scheduler.step()
        print('Loss {:.4f}'.format(np.mean(loss_list)))

    def eval(self, data_loader):
        self.model.eval()
        test_result = []
        label_result = []
        intrain_list = []
        intra_list = []
        inter_list = []
        total_recall = 0
        intra_recall = 0
        inter_recall = 0
        with paddle.no_grad():
            for i, data_batch in enumerate(data_loader):
                if args.use_gpu:
                    data_batch = [data for data in data_batch[:-1]]
                (doc_inputs, ner_inputs, dis_inputs, ref_inputs, rel_labels, intra_mask, inter_mask, intrain_mask, doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask) = data_batch
                outputs = model(doc_inputs, ner_inputs, dis_inputs, ref_inputs, doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask)
                outputs = paddle.nn.functional.sigmoid(x=outputs)
                rel_mask = ent2ent_mask.clone()
                rel_mask[:, range(rel_mask.shape[1]), range(rel_mask.shape[2])] = 0
                
                labels = paddle.flatten(rel_labels[..., 1:])
                outputs = paddle.flatten(outputs[..., 1:])
                intra_mask = paddle.flatten(paddle.cast(intra_mask[..., 1:], 'int64'))
                inter_mask = paddle.flatten(paddle.cast(inter_mask[..., 1:], 'int64'))
                intrain_mask = paddle.flatten(paddle.cast(intrain_mask[..., 1:], 'int64'))
                
                label_result.append(labels)
                test_result.append(outputs)
                intra_list.append(intra_mask)
                inter_list.append(inter_mask)
                intrain_list.append(intrain_mask)
                total_recall += labels.sum().item()
                intra_recall += (intra_mask + labels).equal(y=2).sum().item()
                inter_recall += (inter_mask + labels).equal(y=2).sum().item()
        label_result = paddle.concat(x=label_result)
        test_result = paddle.concat(x=test_result)
        test_result, indices = paddle.sort(descending=True, x=test_result), paddle.argsort(descending=True, x=test_result)
        correct = np.cumsum(label_result[indices].cpu().numpy(), dtype=np.float)
        pr_x = correct / total_recall
        pr_y = correct / np.arange(1, len(correct) + 1)
        f1_arr = 2 * pr_x * pr_y / (pr_x + pr_y + 1e-20)
        auc_score = auc(x=pr_x, y=pr_y)
        intrain_list = paddle.concat(x=intrain_list)
        intrain = np.cumsum(intrain_list[indices].cpu().numpy(), dtype=np.int)
        nt_pr_y = correct - intrain
        nt_pr_y[nt_pr_y != 0] /= (np.arange(1, len(correct) + 1) - intrain)[nt_pr_y != 0]
        nt_f1_arr = 2 * pr_x * nt_pr_y / (pr_x + nt_pr_y + 1e-20)
        nt_f1 = nt_f1_arr.max()
        nt_f1_pos = nt_f1_arr.argmax()
        theta = test_result[nt_f1_pos].cpu().item()
        intra_mask = paddle.concat(x=intra_list)[indices][:nt_f1_pos]
        inter_mask = paddle.concat(x=inter_list)[indices][:nt_f1_pos]
        intra_correct = label_result[indices][:nt_f1_pos][intra_mask].sum().item()
        inter_correct = label_result[indices][:nt_f1_pos][inter_mask].sum().item()
        intra_r = intra_correct / intra_recall
        intra_p = intra_correct / intra_mask.sum().item()
        if intra_p + intra_r == 0:
            intra_f1 = 0.
        else:
            intra_f1 = 2 * intra_p * intra_r / (intra_p + intra_r)
        inter_r = inter_correct / inter_recall
        inter_p = inter_correct / inter_mask.sum().item()
        if inter_p + inter_r == 0:
            inter_f1 = 0.
        else:
            inter_f1 = 2 * inter_p * inter_r / (inter_p + inter_r)
        logger.info(
            'ALL : NT F1 {:3.4f} | F1 {:3.4f} | Intra F1 {:3.4f} | Inter F1 {:3.4f} | Precision {:3.4f} | Recall {:3.4f} | AUC {:3.4f} | THETA {:3.4f}'
            .format(nt_f1, f1_arr[nt_f1_pos], intra_f1, inter_f1, pr_y[
            nt_f1_pos], pr_x[nt_f1_pos], auc_score, theta))
        return nt_f1, theta

    def test(self, data_loader, theta):
        self.model.eval()
        test_result = []
        with paddle.no_grad():
            for i, data_batch in enumerate(data_loader):
                title = data_batch[-1]
                if args.use_gpu:
                    data_batch = [data for data in data_batch[:-1]]
                (doc_inputs, ner_inputs, dis_inputs, ref_inputs, rel_labels,
                    intra_mask, inter_mask, intrain_mask, doc2ent_mask,
                    doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask
                    ) = data_batch
                outputs = model(doc_inputs, ner_inputs, dis_inputs,
                    ref_inputs, doc2ent_mask, doc2men_mask, men2ent_mask,
                    ent2ent_mask, men2men_mask)
                outputs = paddle.nn.functional.sigmoid(x=outputs)
                outputs = outputs.cpu().numpy()
                rel_mask = ent2ent_mask.clone()
                rel_mask[:, range(rel_mask.shape[1]), range(rel_mask.shape[2])] = 0
                for j in range(doc_inputs.shape[0]):
                    L = paddle.sum(x=ent2ent_mask[j, 0]).item()
                    for h_idx in range(L):
                        for t_idx in range(L):
                            if h_idx != t_idx:
                                for r in range(1, 97):
                                    test_result.append((float(outputs[j,
                                        h_idx, t_idx, r]), title[j], vocab.
                                        id2rel[r], h_idx, t_idx, r))
        test_result.sort(key=lambda x: x[0], reverse=True)
        w = 0
        for i, item in enumerate(test_result):
            if item[0] > theta:
                w = i
        output = [{'h_idx': x[3], 't_idx': x[4], 'r_idx': x[-1], 'r': x[-4],
            'title': x[1]} for x in test_result[:w + 1]]
        with open('result.json', 'w', encoding='utf-8') as f:
            json.dump(output, f)

    def save(self, path):
        paddle.save(obj=self.model.state_dict(), path=path)

    def load(self, path):
        self.model.set_state_dict(state_dict=paddle.load(path=path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log_dir', type=str, default='./log/')
    parser.add_argument('-tok_emb_size', type=int, default=100)
    parser.add_argument('-ner_emb_size', type=int, default=20)
    parser.add_argument('-dis_emb_size', type=int, default=20)
    parser.add_argument('-hid_size', type=int, default=256)
    parser.add_argument('-channels', type=int, default=64)
    parser.add_argument('-layers', type=int, default=3)
    parser.add_argument('-chunk', type=int, default=8)
    parser.add_argument('-dropout1', type=float, default=0.5)
    parser.add_argument('-dropout2', type=float, default=0.33)
    parser.add_argument('-epochs', type=int, default=80)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-freeze_epochs', type=int, default=50)
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-wd', type=float, default=1e-05, help='weight deacy')
    parser.add_argument('-gamma', type=float, default=0.8, help='weight deacy')
    parser.add_argument('-use_gpu', type=int, default=1)
    parser.add_argument('-device', type=int, default=0)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    logger = utils.get_logger(args.log_dir + 'Baseline_{}.txt'.format(time.
        strftime('%m-%d_%H-%M-%S')))
    logger.info(args)
    if args.use_gpu and paddle.device.cuda.device_count() >= 1:
        paddle.device.set_device(device='gpu:' + str(args.device))
    logger.info('Loading Data')
    train_set, dev_set, test_dataset, vocab = utils.load_data(True)
    train_loader, dev_loader = (paddle.io.DataLoader(dataset=dataset,
        batch_size=args.batch_size, collate_fn=utils.collate_fn, shuffle=i ==
        0, num_workers=0, drop_last=i == 0) for i, dataset in enumerate([
        train_set, dev_set]))
    logger.info('Loading Embedding')
    embeddings = None
    logger.info('Building Model')
    model = MRN(vocab_size=len(vocab), tok_emb_size=args.tok_emb_size,
        ner_emb_size=args.ner_emb_size, dis_emb_size=args.dis_emb_size,
        hid_size=args.hid_size, channels=args.channels, layers=args.layers,
        dropout1=args.dropout1, dropout2=args.dropout2, chunk=args.chunk,
        embeddings=embeddings)
    if args.use_gpu:
        model = model
    trainer = Trainer(model)
    if args.evaluate:
        with open('theta.txt') as fp:   
            input_theta = float(fp.readline().strip())
        test_loader = paddle.io.DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn, num_workers=0)
        trainer.load('model.pdparams')
        logger.info(input_theta)
        trainer.test(test_loader, input_theta)
    else:
        best_F1 = 0
        input_theta = 0
        for i in range(args.epochs):
            if embeddings is not None:
                if i >= args.freeze_epochs:
                    model.word_emb.weight.stop_gradient = not True
                else:
                    model.word_emb.weight.stop_gradient = not False
            print('Epoch: {}'.format(i))
            trainer.train(train_loader)
            f1, theta = trainer.eval(dev_loader)
            if f1 > best_F1:
                best_F1 = f1
                input_theta = theta
                trainer.save('model.pdparams')
                with open('theta.txt', 'w') as fp:
                    fp.write(str(theta))
        logger.info('Best NT F1: {:3.4f}'.format(best_F1))
        test_loader = paddle.io.DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn, num_workers=0)
        trainer.load('model.pdparams')
        logger.info(input_theta)
        trainer.test(test_loader, input_theta)
