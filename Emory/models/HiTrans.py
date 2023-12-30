import paddle
import paddlenlp
from models.Trans import TransformerEncoder


class HiTrans(paddle.nn.Layer):

    def __init__(self, hidden_dim, emotion_class_num, d_model, d_ff, heads,
        layers, dropout=0, input_max_length=512):
        super(HiTrans, self).__init__()
        self.input_max_length = input_max_length
        self.bert = paddlenlp.transformers.AutoModel.from_pretrained('bert-base-uncased')
        self.encoder = TransformerEncoder(d_model, d_ff, heads, layers, 0.1)
        self.emo_output_layer = paddle.nn.Sequential(paddle.nn.Dropout(p=
            dropout), paddle.nn.Linear(in_features=hidden_dim, out_features
            =emotion_class_num))
        self.spk_output_layer = SpeakerMatchLayer(hidden_dim, dropout)

    def forward(self, dia_input, cls_index, mask):
        bert_outputs = []
        for i in range((dia_input.shape[1] - 1) // self.input_max_length + 1):
            cur_input = dia_input[:, i * self.input_max_length:(i + 1) * self.input_max_length]
            cur_mask = cur_input.not_equal(y=paddle.to_tensor(0))
            bert_output = self.bert(input_ids=cur_input, attention_mask=cur_mask)
            bert_outputs.append(bert_output[0])
        bert_outputs = paddle.concat(x=bert_outputs, axis=1)
        bert_outputs = bert_outputs[paddle.arange(end=bert_outputs.shape[0]
            ).unsqueeze(axis=1), cls_index.astype(dtype='int64')]
        bert_outputs = bert_outputs * mask[:, :, None].astype(dtype='float32')
        bert_outputs = self.encoder(bert_outputs, mask)
        emo_output = self.emo_output_layer(bert_outputs)
        spk_output = self.spk_output_layer(bert_outputs)
        return emo_output, spk_output


class SpeakerMatchLayer(paddle.nn.Layer):

    def __init__(self, hidden_dim, dropout):
        super(SpeakerMatchLayer, self).__init__()
        self.mpl1 = paddle.nn.Sequential(paddle.nn.Dropout(p=dropout),
            paddle.nn.Linear(in_features=hidden_dim, out_features=
            hidden_dim // 2), paddle.nn.LeakyReLU(negative_slope=0.1))
        self.mpl2 = paddle.nn.Sequential(paddle.nn.Dropout(p=dropout),
            paddle.nn.Linear(in_features=hidden_dim, out_features=
            hidden_dim // 2), paddle.nn.LeakyReLU(negative_slope=0.1))
        self.biaffine = Biaffine(hidden_dim // 2, 2)

    def forward(self, x):
        x1 = self.mpl1(x)
        x2 = self.mpl2(x)
        output = self.biaffine(x1, x2)
        return output


class Biaffine(paddle.nn.Layer):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = paddle.zeros(shape=(n_out, n_in + int(bias_x), n_in + int(
            bias_y)))
        init_XavierNormal = paddle.nn.initializer.XavierNormal()
        init_XavierNormal(weight)
        out_1 = paddle.create_parameter(shape=weight.shape, dtype=weight.
            numpy().dtype, default_initializer=paddle.nn.initializer.Assign
            (weight))
        out_1.stop_gradient = not True
        self.weight = out_1

    def extra_repr(self):
        s = f'n_in={self.n_in}, n_out={self.n_out}'
        if self.bias_x:
            s += f', bias_x={self.bias_x}'
        if self.bias_y:
            s += f', bias_y={self.bias_y}'
        return s

    def forward(self, x, y):
        if self.bias_x:
            x = paddle.concat(x=(x, paddle.ones_like(x=x[..., :1])), axis=-1)
        if self.bias_y:
            y = paddle.concat(x=(y, paddle.ones_like(x=y[..., :1])), axis=-1)
        s = paddle.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        s = s.transpose(perm=[0, 2, 3, 1])
        return s
