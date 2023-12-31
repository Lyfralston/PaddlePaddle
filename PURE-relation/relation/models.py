import paddle
import paddlenlp
# from allennlp.nn.util import batched_index_select
BertLayerNorm = paddle.nn.LayerNorm


def batched_index_select(target, indices):
    batch_size, num_indices = indices.shape
    target_dim = target.shape[-1]

    flat_indices = paddle.flatten(indices)
    flat_target = paddle.reshape(target, [-1, target.shape[-1]])

    selected_elements = paddle.gather(flat_target, flat_indices)

    result = paddle.reshape(selected_elements, [batch_size, num_indices, target_dim])

    return result


class BertForRelation(paddlenlp.transformers.BertModel):

    def __init__(self, config, num_rel_labels):
        super(BertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = paddlenlp.transformers.BertModel(config)
        self.dropout = paddle.nn.Dropout(p=config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = paddle.nn.Linear(in_features=config.hidden_size *
            2, out_features=self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
        labels=None, sub_idx=None, obj_idx=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids,
            attention_mask=attention_mask, output_hidden_states=False,
            output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]
        sub_output = paddle.concat(x=[a[i].unsqueeze(axis=0) for a, i in
            zip(sequence_output, sub_idx)])
        obj_output = paddle.concat(x=[a[i].unsqueeze(axis=0) for a, i in
            zip(sequence_output, obj_idx)])
        rep = paddle.concat(x=(sub_output, obj_output), axis=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape([-1, self.num_labels]), labels.reshape([-1]))
            return loss
        else:
            return logits