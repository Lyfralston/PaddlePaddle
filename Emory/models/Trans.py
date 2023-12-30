import sys
sys.path.append('/home/lyf/paddle/HiTrans-pd/utils')
import paddle_aux
import paddle
import math


def gelu(x):
    return 0.5 * x * (1 + paddle.nn.functional.tanh(x=math.sqrt(2 / math.pi) * (x + 0.044715 * paddle.pow(x=x, y=3))))


class PositionwiseFeedForward(paddle.nn.Layer):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = paddle.nn.Linear(in_features=d_model, out_features=d_ff)
        self.w_2 = paddle.nn.Linear(in_features=d_ff, out_features=d_model)
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=d_model,
            epsilon=1e-06)
        self.actv = gelu
        self.dropout_1 = paddle.nn.Dropout(p=dropout)
        self.dropout_2 = paddle.nn.Dropout(p=dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(paddle.nn.Layer):

    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        self.linear_k = paddle.nn.Linear(in_features=model_dim,
            out_features=head_count * self.dim_per_head)
        self.linear_v = paddle.nn.Linear(in_features=model_dim,
            out_features=head_count * self.dim_per_head)
        self.linear_q = paddle.nn.Linear(in_features=model_dim,
            out_features=head_count * self.dim_per_head)
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.linear = paddle.nn.Linear(in_features=model_dim, out_features=
            model_dim)


    def forward(self, key, value, query, mask=None):
        batch_size = key.shape[0]
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.reshape([batch_size, -1, head_count, dim_per_head]).transpose([0, 2, 1, 3])

        def unshape(x):
            """  compute context """
            return x.transpose([0, 2, 1, 3]).reshape([batch_size, -1, head_count * dim_per_head])

        key = shape(self.linear_k(key))
        value = shape(self.linear_v(value))
        query = shape(self.linear_q(query))

        query = query / paddle.sqrt(paddle.to_tensor(dim_per_head, dtype='float32'))
        scores = paddle.matmul(query, key.transpose([0, 1, 3, 2]))

        if mask is not None:
            mask = paddle.unsqueeze(mask, axis=1).expand_as(scores)
            scores = paddle.where(mask, scores, paddle.to_tensor(-1e10))

        attn = self.softmax(scores)

        drop_attn = self.dropout(attn)
        context = paddle.matmul(drop_attn, value).transpose([0, 2, 1, 3])
        context = context.reshape([batch_size, -1, head_count * dim_per_head])
        output = self.linear(context)
        return output


class PositionalEncoding(paddle.nn.Layer):

    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = paddle.zeros(shape=[max_len, dim])
        position = paddle.arange(start=0, end=max_len).unsqueeze(axis=1)
        div_term = paddle.exp(x=paddle.arange(start=0, end=dim, step=2,
            dtype='float32') * -(math.log(10000.0) / dim))
        pe[:, 0::2] = paddle.sin(x=position.astype(dtype='float32') * div_term)
        pe[:, 1::2] = paddle.cos(x=position.astype(dtype='float32') * div_term)
        pe = pe.unsqueeze(axis=0)
        self.register_buffer(name='pe', tensor=pe)

    def forward(self, x):
        L = x.shape[1]
        pos_emb = self.pe[:, :L]
        x = x + pos_emb
        return x


class TransformerEncoderLayer(paddle.nn.Layer):

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=d_model,
            epsilon=1e-06)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, iter, query, inputs, mask):
        if iter != 0:
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs
        mask = mask.unsqueeze(axis=1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(paddle.nn.Layer):

    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = paddle.nn.LayerList(sublayers=[
            TransformerEncoderLayer(d_model, heads, d_ff, dropout) for _ in
            range(layers)])
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.pos_emb(x)
        x = self.dropout(x)
        for i in range(self.layers):
            x = self.transformer_inter[i](i, x, x, mask.equal(y=0))
        return x
