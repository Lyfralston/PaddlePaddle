import os, sys, logging, tqdm, pprint, pickle
# import torch
import paddle
import numpy as np
from collections import namedtuple
# from transformers import RobertaTokenizer, AutoTokenizer, get_linear_schedule_with_warmup
from paddlenlp.transformers import RobertaTokenizer, AutoTokenizer, LinearDecayWithWarmup
# from torch.utils.data import DataLoader
from paddle.io import DataLoader, Dataset
# from torch.optim import AdamW
from paddle.optimizer import AdamW
from ..trainer import BasicTrainer
from .EAEmodel import RCEEEAEModel
from scorer import compute_EAE_scores, print_scores
import ipdb

logger = logging.getLogger(__name__)

def clip_grad_value_(parameters, clip_value):
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        paddle.clip(p.grad, min=-clip_value, max=clip_value)

def read_query_templates(normal_file, des_file):
    """Load query templates"""
    query_templates = dict()
    with open(normal_file, "r", encoding='utf-8') as f:
        for line in f:
            event_arg, query = line.strip().split(",")
            event_type, arg_name = event_arg.split("_")

            if event_type not in query_templates:
                query_templates[event_type] = dict()
            if arg_name not in query_templates[event_type]:
                query_templates[event_type][arg_name] = list()

            # 0 template arg_name
            query_templates[event_type][arg_name].append(arg_name)
            # 1 template arg_name + in trigger (replace [trigger] when forming the instance)
            query_templates[event_type][arg_name].append(arg_name + " in [trigger]")
            # 2 template arg_query
            query_templates[event_type][arg_name].append(query)
            # 3 arg_query + trigger (replace [trigger] when forming the instance)
            query_templates[event_type][arg_name].append(query[:-1] + " in [trigger]?")

    with open(des_file, "r", encoding='utf-8') as f:
        for line in f:
            event_arg, query = line.strip().split(",")
            event_type, arg_name = event_arg.split("_")
            # 4 template des_query
            query_templates[event_type][arg_name].append(query)
            # 5 template des_query + trigger (replace [trigger] when forming the instance)
            query_templates[event_type][arg_name].append(query[:-1] + " in [trigger]?")

    for event_type in query_templates:
        for arg_name in query_templates[event_type]:
            assert len(query_templates[event_type][arg_name]) == 6

    return query_templates

EAEBatch_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_tokens', 'batch_pieces', 'batch_token_lens', 'batch_token_num', 'batch_text', 'batch_trigger', 'batch_arguments']
EAEBatch = namedtuple('EAEBatch', field_names=EAEBatch_fields, defaults=[None] * len(EAEBatch_fields))

def EAE_collate_fn(batch):
    return EAEBatch(
        batch_doc_id=[instance["doc_id"] for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_tokens=[instance["tokens"] for instance in batch], 
        batch_pieces=[instance["pieces"] for instance in batch], 
        batch_token_lens=[instance["token_lens"] for instance in batch], 
        batch_token_num=paddle.to_tensor([instance["token_num"] for instance in batch]), 
        batch_text=[instance["text"] for instance in batch], 
        batch_trigger=[instance["trigger"] for instance in batch], 
        batch_arguments=[instance["arguments"] for instance in batch], 
    )

class CustomDataset(Dataset):
    def __init__(self, data):
        super(CustomDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
   
class CustomOptimizer:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def clear_grad(self):
        for optimizer in self.optimizers:
            optimizer.clear_grad()

class CustomScheduler:
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()
            
class RCEEEAETrainer(BasicTrainer):
    def __init__(self, config, type_set=None):
        super().__init__(config, type_set)
        self.tokenizer = None
        self.model = None
    
    def load_model(self, checkpoint=None):
        if checkpoint:
            logger.info(f"Loading model from {checkpoint}")
            with open(os.path.join(checkpoint, f"best_model_tokenizer.pkl"), "rb") as f:
                self.tokenizer = pickle.load(f)
            with open(os.path.join(checkpoint, f"best_model_type_set.pkl"), "rb") as f:
                self.type_set = pickle.load(f)
            self.model = RCEEEAEModel(self.config, self.tokenizer, self.type_set)
            self.model.set_state_dict(paddle.load(os.path.join(checkpoint, "best_model.pdparams")))
            
            # state = torch.load(os.path.join(checkpoint, "best_model.state"), map_location=f'cuda:{self.config.gpu_device}')
            # state = paddle.load(os.path.join(checkpoint, "best_model.state"))
            # self.tokenizer = state["tokenizer"]
            # self.type_set = state["type_set"]
            # self.model.load_state_dict(state['model'])  # torch
            # self.model.set_state_dict(state['model'])  # paddle
            # self.model.cuda(device=self.config.gpu_device)
            self.model.to(device=paddle.CUDAPlace(self.config.gpu_device))
        else:
            logger.info(f"Loading model from {self.config.pretrained_model_name}")
            if self.config.pretrained_model_name.startswith('roberta-'):
                self.tokenizer = RobertaTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir, do_lower_case=False, add_prefix_space=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir, do_lower_case=False, use_fast=False)
            self.model = RCEEEAEModel(self.config, self.tokenizer, self.type_set)
            # self.model.cuda(device=self.config.gpu_device)
            self.model.to(device=paddle.CUDAPlace(self.config.gpu_device))
    
    def process_data(self, data):
        assert self.tokenizer, "Please load model and tokneizer before processing data!"
        
        logger.info("Removing overlapping arguments and over-length examples")
        
        # greedily remove overlapping arguments
        n_total = 0
        new_data = []
        for dt in data:
            
            n_total += 1
            
            if len(dt["tokens"]) > self.config.max_length:
                continue
            
            trigger = dt["trigger"]
            no_overlap_flag = np.ones((len(dt["tokens"]), ), dtype=bool)
            new_arguments = []
            for argument in sorted(dt["arguments"]):
                start, end = argument[0], argument[1]
                if np.all(no_overlap_flag[start:end]):
                    new_arguments.append(argument)
                    no_overlap_flag[start:end] = False
            
            pieces = [self.tokenizer.tokenize(t, is_split_into_words=True) for t in dt["tokens"]]
            token_lens = [len(p) for p in pieces] 

            new_dt = {"doc_id": dt["doc_id"], 
                      "wnd_id": dt["wnd_id"], 
                      "tokens": dt["tokens"], 
                      "pieces": [p for w in pieces for p in w], 
                      "token_lens": token_lens, 
                      "token_num": len(dt["tokens"]), 
                      "text": dt["text"], 
                      "trigger": dt["trigger"], 
                      "arguments": new_arguments
                     }
            
            
            new_data.append(new_dt)
                
        logger.info(f"There are {len(new_data)}/{n_total} EAE instances after removing overlapping arguments and over-length examples")

        return new_data
    
    def train(self, train_data, dev_data, **kwargs):
        
        self.load_model()
        
        internal_train_data = self.process_data(train_data)
        internal_train_data = CustomDataset(internal_train_data)
        internal_dev_data = self.process_data(dev_data)
        internal_dev_data = CustomDataset(internal_dev_data)
        
        # param_groups = [
        #     {
        #         'params': [p for n, p in self.model.named_parameters() if n.startswith('base_model')],
        #         'lr': self.config.base_model_learning_rate, 'weight_decay': self.config.base_model_weight_decay
        #     },
        #     {
        #         'params': [p for n, p in self.model.named_parameters() if not n.startswith('base_model')],
        #         'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay
        #     },
        # ]
        # param_groups = [
        #     {
        #         'params': [p for n, p in self.model.named_parameters() if n.startswith('base_model')],
        #         'learning_rate': self.config.base_model_learning_rate, 'weight_decay': self.config.base_model_weight_decay
        #     },
        #     {
        #         'params': [p for n, p in self.model.named_parameters() if not n.startswith('base_model')],
        #         'learning_rate': self.config.learning_rate, 'weight_decay': self.config.weight_decay
        #     },
        # ]
        
        train_batch_num = len(internal_train_data) // self.config.train_batch_size + (len(internal_train_data) % self.config.train_batch_size != 0)
        # optimizer = AdamW(params=param_groups)
        # scheduler = get_linear_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=train_batch_num*self.config.warmup_epoch,
        #                                             num_training_steps=train_batch_num*self.config.max_epoch)
        
        base_model_params = [p for n, p in self.model.named_parameters() if n.startswith('base_model')]
        other_params = [p for n, p in self.model.named_parameters() if not n.startswith('base_model')]
        base_model_optimizer = AdamW(
            learning_rate=self.config.base_model_learning_rate,
            parameters=base_model_params,
            weight_decay=self.config.base_model_weight_decay
        )
        other_optimizer = AdamW(
            learning_rate=self.config.learning_rate,
            parameters=other_params,
            weight_decay=self.config.weight_decay
        )
        base_model_scheduler = LinearDecayWithWarmup(
            learning_rate=self.config.base_model_learning_rate,
            total_steps=train_batch_num * self.config.max_epoch,
            warmup=train_batch_num * self.config.warmup_epoch
        )

        other_scheduler = LinearDecayWithWarmup(
            learning_rate=self.config.learning_rate,
            total_steps=train_batch_num * self.config.max_epoch,
            warmup=train_batch_num * self.config.warmup_epoch
        )
        base_model_optimizer._learning_rate = base_model_scheduler
        other_optimizer._learning_rate = other_scheduler
        optimizer = CustomOptimizer([base_model_optimizer, other_optimizer])
        scheduler = CustomScheduler([base_model_scheduler, other_scheduler])
        # optimizer = AdamW(learning_rate=self.config.learning_rate, parameters=param_groups)
        # scheduler = LinearDecayWithWarmup(learning_rate=self.config.learning_rate, 
        #                                   total_steps=train_batch_num * self.config.max_epoch,
        #                                   warmup=train_batch_num * self.config.warmup_epoch)
        
        
        best_scores = {"argument_attached_cls": {"f1": 0.0}}
        best_epoch = -1
        
        for epoch in range(1, self.config.max_epoch+1):
            logger.info(f"Log path: {self.config.log_path}")
            logger.info(f"Epoch {epoch}")
            
            # training step
            progress = tqdm.tqdm(total=train_batch_num, ncols=100, desc='Train {}'.format(epoch))
            
            self.model.train()
            # optimizer.zero_grad()
            optimizer.clear_grad()
            cummulate_loss = []
            for batch_idx, paddle_batch in enumerate(DataLoader(internal_train_data, batch_size=self.config.train_batch_size // self.config.accumulate_step, 
                                                         shuffle=True, drop_last=False, collate_fn=EAE_collate_fn)):
                batch = EAEBatch(
                    batch_doc_id=paddle_batch[0],
                    batch_wnd_id=paddle_batch[1],
                    batch_tokens=paddle_batch[2], 
                    batch_pieces=paddle_batch[3], 
                    batch_token_lens=paddle_batch[4], 
                    batch_token_num=paddle_batch[5].tolist(), 
                    batch_text=paddle_batch[6], 
                    batch_trigger=paddle_batch[7], 
                    batch_arguments=paddle_batch[8], 
                )
                loss = self.model(batch)
                loss = loss * (1 / self.config.accumulate_step)
                # cummulate_loss.append(loss.item())
                cummulate_loss.append(float(loss))
                loss.backward()

                if (batch_idx + 1) % self.config.accumulate_step == 0:
                    progress.update(1)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clipping)
                    # paddle.nn.clip.clip_grad_norm_(self.model.parameters(), self.config.grad_clipping)
                    clip_grad_value_(self.model.parameters(), self.config.grad_clipping)
                    optimizer.step()
                    scheduler.step()
                    # optimizer.zero_grad()
                    optimizer.clear_grad()
                    
            progress.close()
            logger.info(f"Average training loss: {np.mean(cummulate_loss)}")
            print(f"Average training loss: {np.mean(cummulate_loss)}")
            
            # eval dev
            predictions = self.internal_predict(internal_dev_data, split="Dev")
            dev_scores = compute_EAE_scores(predictions, internal_dev_data, metrics={"argument_id", "argument_cls", "argument_attached_id", "argument_attached_cls"})

            # print scores
            print(f"Dev {epoch}")
            print_scores(dev_scores)
            
            if dev_scores["argument_attached_cls"]["f1"] >= best_scores["argument_attached_cls"]["f1"]:
                logger.info("Saving best model")
                # state = dict(model=self.model.state_dict(), tokenizer=self.tokenizer, type_set=self.type_set)
                # paddle.save(state, os.path.join(self.config.output_dir, "best_model.state"))
                paddle.save(self.model.state_dict(), os.path.join(self.config.output_dir, "best_model.pdparams"))
                with open(os.path.join(self.config.output_dir, f"best_model_tokenizer.pkl"), "wb") as f:
                    pickle.dump(self.tokenizer, f)
                with open(os.path.join(self.config.output_dir, f"best_model_type_set.pkl"), "wb") as f:
                    pickle.dump(self.type_set, f)
                # torch.save(state, os.path.join(self.config.output_dir, "best_model.state"))
                best_scores = dev_scores
                best_epoch = epoch
                
            logger.info(pprint.pformat({"epoch": epoch, "dev_scores": dev_scores}))
            logger.info(pprint.pformat({"best_epoch": best_epoch, "best_scores": best_scores}))
        
        
    def internal_predict(self, eval_data, split="Dev"):
        eval_batch_num = len(eval_data) // self.config.eval_batch_size + (len(eval_data) % self.config.eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        
        predictions = []
        for batch_idx, paddle_batch in enumerate(DataLoader(eval_data, batch_size=self.config.eval_batch_size, 
                                                     shuffle=False, collate_fn=EAE_collate_fn)):
            progress.update(1)
            batch = EAEBatch(
                batch_doc_id=paddle_batch[0],
                batch_wnd_id=paddle_batch[1],
                batch_tokens=paddle_batch[2], 
                batch_pieces=paddle_batch[3], 
                batch_token_lens=paddle_batch[4], 
                batch_token_num=paddle_batch[5].tolist(), 
                batch_text=paddle_batch[6], 
                batch_trigger=paddle_batch[7], 
                batch_arguments=paddle_batch[8], 
            )
            batch_pred_arguments = self.model.predict(batch)
            for doc_id, wnd_id, tokens, text, trigger, pred_arguments in zip(batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, batch.batch_text, 
                                                                             batch.batch_trigger, batch_pred_arguments):
                prediction = {"doc_id": doc_id,  
                              "wnd_id": wnd_id, 
                              "tokens": tokens, 
                              "text": text, 
                              "trigger": trigger, 
                              "arguments": pred_arguments
                             }
                
                predictions.append(prediction)
        progress.close()
        
        return predictions

    
    def predict(self, data, **kwargs):
        assert self.tokenizer and self.model
        internal_data = self.process_data(data)
        predictions = self.internal_predict(internal_data, split="Test")
        return predictions
