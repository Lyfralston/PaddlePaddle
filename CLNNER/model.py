import paddle
import paddlenlp


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


class LayerNorm(paddle.nn.Layer):

    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False, hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        P.S. In CLN of W2NER, input_dim and cond_dim are the same as the hidden size of output of LSTM.
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        if self.center:
            out_0 = paddle.create_parameter(shape=paddle.zeros(shape=input_dim).shape, dtype=paddle.zeros(shape=input_dim).numpy().dtype, default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=input_dim)))
            out_0.stop_gradient = not True
            self.beta = out_0
        if self.scale:
            out_1 = paddle.create_parameter(shape=paddle.ones(shape=input_dim).shape, dtype=paddle.ones(shape=input_dim).numpy().dtype, default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=input_dim)))
            out_1.stop_gradient = not True
            self.gamma = out_1
        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = paddle.nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias_attr=False)
            if self.center:
                self.beta_dense = paddle.nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias_attr=False)
            if self.scale:
                self.gamma_dense = paddle.nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias_attr=False)
        self.initialize_weights()

    def initialize_weights(self):
        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    init_Normal = paddle.nn.initializer.Normal()
                    init_Normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':
                    init_XavierUniform = paddle.nn.initializer.XavierUniform()
                    init_XavierUniform(self.hidden_dense.weight)
            if self.center:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(self.beta_dense.weight)
            if self.scale:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(self.gamma_dense.weight)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(axis=1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta  # 512(512) + 512
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma  # 512(512) + 512
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma
        outputs = inputs
        if self.center:
            mean = paddle.mean(x=outputs, axis=-1).unsqueeze(axis=-1)
            outputs = outputs - mean
        if self.scale:
            variance = paddle.mean(x=outputs ** 2, axis=-1).unsqueeze(axis=-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta
        return outputs


class ConvolutionLayer(paddle.nn.Layer):

    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = paddle.nn.Sequential(paddle.nn.Dropout2D(p=dropout), paddle.nn.Conv2D(in_channels=input_size, out_channels=channels, kernel_size=1), paddle.nn.GELU())
        self.convs = paddle.nn.LayerList(sublayers=[paddle.nn.Conv2D(in_channels=channels, out_channels=channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.transpose(perm=[0, 3, 1, 2])
        x = self.base(x)
        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = paddle.nn.functional.gelu(x=x)
            outputs.append(x)
        outputs = paddle.concat(x=outputs, axis=1)
        outputs = outputs.transpose(perm=[0, 2, 3, 1])
        return outputs


class Biaffine(paddle.nn.Layer):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = paddle.zeros(shape=(n_out, n_in + int(bias_x), n_in + int(bias_y)))
        init_XavierNormal = paddle.nn.initializer.XavierNormal()
        init_XavierNormal(weight)
        out_2 = paddle.create_parameter(shape=weight.shape, dtype=weight.numpy().dtype, default_initializer=paddle.nn.initializer.Assign(weight))
        out_2.stop_gradient = not True
        self.weight = out_2

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


class MLP(paddle.nn.Layer):

    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features=n_in, out_features=n_out)
        self.activation = paddle.nn.GELU()
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(paddle.nn.Layer):
    def __init__(self, cls_num, hid_size, biaffine_size, channels,
        ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)  # 512, 512
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)  # 512, 512
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)  # channels: 512, ffnn: 384
        self.linear = paddle.nn.Linear(in_features=ffnn_hid_size, out_features=cls_num)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)
        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2


class Model(paddle.nn.Layer):

    def __init__(self, config):
        super(Model, self).__init__()
        self.use_bert_last_4_layers = config.use_bert_last_4_layers
        self.lstm_hid_size = config.lstm_hid_size
        self.conv_hid_size = config.conv_hid_size
        lstm_input_size = 0
        self.bert = paddlenlp.transformers.AutoModel.from_pretrained(config.bert_name)
        lstm_input_size += config.bert_hid_size
        self.encoder = paddle.nn.LSTM(input_size=lstm_input_size, hidden_size=config.lstm_hid_size // 2, num_layers=1, time_major=not True, direction='bidirectional')
        conv_input_size = config.lstm_hid_size
        self.convLayer = ConvolutionLayer(conv_input_size, config.conv_hid_size, config.dilation, config.conv_dropout)
        self.dropout = paddle.nn.Dropout(p=config.emb_dropout)
        self.predictor = CoPredictor(config.label_num, config.lstm_hid_size, config.biaffine_size, config.conv_hid_size * len(config.dilation), config.ffnn_hid_size, config.out_dropout)
        self.cln = LayerNorm(config.lstm_hid_size, config.lstm_hid_size, conditional=True)

    def forward(self, bert_inputs, grid_mask2d, pieces2word, sent_length):
        """
        :param bert_inputs: [B, L']  # B: batch, L': transformed length (influenced by pieces)
        :param grid_mask2d: [B, L, L]  # L: sentence length (the number of tokens)
        :param pieces2word: [B, L, L']
        :param sent_length: [B]
        :return:
        """
        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.not_equal(y=paddle.to_tensor(0)).astype(dtype='float32'))
        if self.use_bert_last_4_layers:
            bert_embs = paddle.stack(x=bert_embs[2][-4:], axis=-1).mean(axis=-1)
        else:
            bert_embs = bert_embs[0]  # [B, L', 768]
            
        length = pieces2word.shape[1]
        
        min_value = paddle.min(x=bert_embs).item()
        
        _bert_embs = bert_embs.unsqueeze(axis=1).expand(shape=[-1, length, -1, -1])  # [B, L, L', 768]
        _bert_embs = masked_fill(_bert_embs, pieces2word.equal(y=0).unsqueeze(axis=-1), min_value)
        word_reps, _ = paddle.max(x=_bert_embs, axis=2), paddle.argmax(x=_bert_embs, axis=2)
        word_reps = self.dropout(word_reps)
        word_reps, (hidden, _) = self.encoder(word_reps)
        cln = self.cln(word_reps.unsqueeze(axis=2), word_reps)
        conv_inputs = masked_fill(cln, grid_mask2d.equal(y=0).unsqueeze(axis=-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = masked_fill(conv_outputs, grid_mask2d.equal(y=0).unsqueeze(axis=-1), 0.0)
        outputs = self.predictor(word_reps, word_reps, conv_outputs)
        return outputs
