import paddle
import paddlenlp
# from allennlp.nn.util import batched_index_select
import numpy as np
import os
import json
import logging
logger = logging.getLogger('root')

def batched_index_select(target, indices):
    batch_size, num_indices = indices.shape
    target_dim = target.shape[-1]

    flat_indices = paddle.flatten(indices)
    flat_target = paddle.reshape(target, [-1, target.shape[-1]])

    selected_elements = paddle.gather(flat_target, flat_indices)

    result = paddle.reshape(selected_elements, [batch_size, num_indices, target_dim])

    return result


class MyFeedForward(paddle.nn.Layer):

    def __init__(self, input_dim, num_layers, hidden_dims, activations, dropout):
        super(MyFeedForward, self).__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers
        if not isinstance(activations, list):
            activations = [activations] * num_layers
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers
        self._activations = paddle.nn.LayerList(sublayers=activations)
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(paddle.nn.Linear(in_features=
                layer_input_dim, out_features=layer_output_dim))
        self._linear_layers = paddle.nn.LayerList(sublayers=linear_layers)
        dropout_layers = [paddle.nn.Dropout(p=value) for value in dropout]
        self._dropout = paddle.nn.LayerList(sublayers=dropout_layers)
        self._output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def forward(self, inputs):
        output = inputs
        for layer, activation, dropout in zip(self._linear_layers, self.
            _activations, self._dropout):
            output = dropout(activation(layer(output)))
        return output


class BertForEntity(paddlenlp.transformers.BertModel):
    def __init__(self, config, num_ner_labels, head_hidden_dim=150,
        width_embedding_dim=150, max_span_length=8):
        super().__init__(config)
        
        self.bert = paddlenlp.transformers.BertModel(config)
        self.hidden_dropout = paddle.nn.Dropout(p=config.hidden_dropout_prob)
        self.width_embedding = paddle.nn.Embedding(num_embeddings=max_span_length + 1, embedding_dim=width_embedding_dim)
        
        self.ner_classifier = paddle.nn.Sequential(
            MyFeedForward(input_dim=config.hidden_size * 2 + width_embedding_dim, 
                          num_layers=2,
                          hidden_dims=head_hidden_dim, 
                          activations=paddle.nn.GELU(),
                          dropout=0.2), 
            paddle.nn.Linear(in_features=head_hidden_dim, out_features=num_ner_labels))
        self.init_weights()

    def _get_span_embeddings(self, input_ids, spans, token_type_ids=None,
        attention_mask=None):
        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=
            token_type_ids, attention_mask=attention_mask)
        sequence_output = self.hidden_dropout(sequence_output)
        """
        spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        # spans_start = spans[:, :, 0].view(spans.shape[0], -1)
        spans_start = paddle.reshape(spans[:, :, 0], [spans.shape[0], -1])
        spans_start_embedding = batched_index_select(sequence_output, spans_start)
        
        # spans_end = spans[:, :, 1].view(spans.shape[0], -1)
        spans_end = paddle.reshape(spans[:, :, 1], [spans.shape[0], -1])
        spans_end_embedding = batched_index_select(sequence_output, spans_end)
        
        # spans_width = spans[:, :, 2].view(spans.shape[0], -1)
        spans_width = paddle.reshape(spans[:, :, 2], [spans.shape[0], -1])
        spans_width_embedding = self.width_embedding(spans_width)
        spans_embedding = paddle.concat(x=(spans_start_embedding, spans_end_embedding, spans_width_embedding), axis=-1)
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding

    def forward(self, input_ids, spans, spans_mask, spans_ner_label=None,
        token_type_ids=None, attention_mask=None):
        spans_embedding = self._get_span_embeddings(input_ids, spans,
            token_type_ids=token_type_ids, attention_mask=attention_mask)
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]
        if spans_ner_label is not None:
            loss_fct = paddle.nn.CrossEntropyLoss(reduction='sum')
            if attention_mask is not None:
                # active_loss = spans_mask.view(-1) == 1
                active_loss = paddle.to_tensor(spans_mask.numpy().reshape(-1) == 1)
                # active_logits = logits.view(-1, logits.shape[-1])
                active_logits = paddle.reshape(logits, [-1, logits.shape[-1]])
                # active_labels = paddle.where(condition=active_loss, 
                #                              x=spans_ner_label.view(-1),
                #                              y=paddle.to_tensor(data=loss_fct.ignore_index).astype(dtype=spans_ner_label.dtype))
                active_labels = paddle.where(condition=active_loss, 
                                             x=paddle.reshape(spans_ner_label, [-1]),
                                             y=paddle.to_tensor(data=loss_fct.ignore_index).astype(dtype=spans_ner_label.dtype))
                loss = loss_fct(active_logits, active_labels)
            else:
                # loss = loss_fct(logits.view(-1, logits.shape[-1]),
                #     spans_ner_label.view(-1))
                loss = loss_fct(paddle.reshape(logits, [-1, logits.shape[-1]]), 
                                paddle.reshape(spans_ner_label, [-1]))
            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding


class EntityModel():
    def __init__(self, args, num_ner_labels):
        super().__init__()
        bert_model_name = args.model
        vocab_name = bert_model_name
        if args.bert_model_dir is not None:
            bert_model_name = str(args.bert_model_dir) + '/'
            vocab_name = bert_model_name
            logger.info('Loading BERT model from {}'.format(bert_model_name))
        self.tokenizer = paddlenlp.transformers.BertTokenizer.from_pretrained(vocab_name)
        self.bert_model = BertForEntity.from_pretrained(bert_model_name,
                num_ner_labels=num_ner_labels, max_span_length=args.max_span_length) # , convert_from_torch=True)

    def _get_input_tensors(self, tokens, spans, spans_ner_label):
        start2idx = []
        end2idx = []
        bert_tokens = []
        bert_tokens.append(self.tokenizer.cls_token)
        for token in tokens:
            start2idx.append(len(bert_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens) - 1)
        bert_tokens.append(self.tokenizer.sep_token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = paddle.to_tensor(data=[indexed_tokens])
        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for
            span in spans]
        bert_spans_tensor = paddle.to_tensor(data=[bert_spans])
        spans_ner_label_tensor = paddle.to_tensor(data=[spans_ner_label])
        return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor

    def _get_input_tensors_batch(self, samples_list, training=True):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        sentence_length = []
        max_tokens = 0
        max_spans = 0
        for sample in samples_list:
            tokens = sample['tokens']
            spans = sample['spans']
            spans_ner_label = sample['spans_label']
            tokens_tensor, bert_spans_tensor, spans_ner_label_tensor = (self
                ._get_input_tensors(tokens, spans, spans_ner_label))
            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            assert bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1
                ]
            if tokens_tensor.shape[1] > max_tokens:
                max_tokens = tokens_tensor.shape[1]
            if bert_spans_tensor.shape[1] > max_spans:
                max_spans = bert_spans_tensor.shape[1]
            sentence_length.append(sample['sent_length'])
        sentence_length = paddle.to_tensor(data=sentence_length)
        final_tokens_tensor = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_mask_tensor = None
        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor in zip(
            tokens_tensor_list, bert_spans_tensor_list,
            spans_ner_label_tensor_list):
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = paddle.full(shape=[1, num_tokens],
                fill_value=1, dtype='int64')
            if tokens_pad_length > 0:
                pad = paddle.full(shape=[1, tokens_pad_length], fill_value=
                    self.tokenizer.pad_token_id, dtype='int64')
                tokens_tensor = paddle.concat(x=(tokens_tensor, pad), axis=1)
                attention_pad = paddle.full(shape=[1, tokens_pad_length],
                    fill_value=0, dtype='int64')
                attention_tensor = paddle.concat(x=(attention_tensor,
                    attention_pad), axis=1)
            num_spans = bert_spans_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = paddle.full(shape=[1, num_spans],
                fill_value=1, dtype='int64')
            if spans_pad_length > 0:
                pad = paddle.full(shape=[1, spans_pad_length,
                    bert_spans_tensor.shape[2]], fill_value=0, dtype='int64')
                bert_spans_tensor = paddle.concat(x=(bert_spans_tensor, pad
                    ), axis=1)
                mask_pad = paddle.full(shape=[1, spans_pad_length],
                    fill_value=0, dtype='int64')
                spans_mask_tensor = paddle.concat(x=(spans_mask_tensor,
                    mask_pad), axis=1)
                spans_ner_label_tensor = paddle.concat(x=(
                    spans_ner_label_tensor, mask_pad), axis=1)
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_mask_tensor = spans_mask_tensor
            else:
                final_tokens_tensor = paddle.concat(x=(final_tokens_tensor,
                    tokens_tensor), axis=0)
                final_attention_mask = paddle.concat(x=(
                    final_attention_mask, attention_tensor), axis=0)
                final_bert_spans_tensor = paddle.concat(x=(
                    final_bert_spans_tensor, bert_spans_tensor), axis=0)
                final_spans_ner_label_tensor = paddle.concat(x=(
                    final_spans_ner_label_tensor, spans_ner_label_tensor),
                    axis=0)
                final_spans_mask_tensor = paddle.concat(x=(
                    final_spans_mask_tensor, spans_mask_tensor), axis=0)
        return (final_tokens_tensor, final_attention_mask,
            final_bert_spans_tensor, final_spans_mask_tensor,
            final_spans_ner_label_tensor, sentence_length)

    def run_batch(self, samples_list, try_cuda=True, training=True):
        (tokens_tensor, attention_mask_tensor, bert_spans_tensor,
            spans_mask_tensor, spans_ner_label_tensor, sentence_length
            ) = self._get_input_tensors_batch(samples_list, training)
        output_dict = {'ner_loss': 0}
        if training:
            self.bert_model.train()
            ner_loss, ner_logits, spans_embedding = self.bert_model(input_ids=tokens_tensor,
                                                                    spans=bert_spans_tensor, 
                                                                    spans_mask=spans_mask_tensor, 
                                                                    spans_ner_label=spans_ner_label_tensor,
                                                                    attention_mask=attention_mask_tensor)
            output_dict['ner_loss'] = ner_loss.sum()
            output_dict['ner_llh'] = paddle.nn.functional.log_softmax(x=ner_logits, axis=-1)
        else:
            self.bert_model.eval()
            with paddle.no_grad():
                ner_logits, spans_embedding, last_hidden = self.bert_model(input_ids=tokens_tensor, 
                                                                           spans=bert_spans_tensor, 
                                                                           spans_mask=spans_mask_tensor,
                                                                           spans_ner_label=None, 
                                                                           attention_mask=attention_mask_tensor)
            _, predicted_label = ner_logits.max(2)
            predicted_label = predicted_label.cpu().numpy()
            last_hidden = last_hidden.cpu().numpy()
            predicted = []
            pred_prob = []
            hidden = []
            for i, sample in enumerate(samples_list):
                ner = []
                prob = []
                lh = []
                for j in range(len(sample['spans'])):
                    ner.append(predicted_label[i][j])
                    prob.append(ner_logits[i][j].cpu().numpy())
                    lh.append(last_hidden[i][j])
                predicted.append(ner)
                pred_prob.append(prob)
                hidden.append(lh)
            output_dict['pred_ner'] = predicted
            output_dict['ner_probs'] = pred_prob
            output_dict['ner_last_hidden'] = hidden
        return output_dict
