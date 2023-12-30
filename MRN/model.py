import sys
sys.path.append('/home/lyf/paddle/MRN-pd/utils')
import paddle_aux
import paddle
import paddle.nn.functional as F
import numpy as np
import math


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


class ConvAttentionLayer(paddle.nn.Layer):

    def __init__(self, hid_dim, n_heads, pre_channels, channels, groups,
        dropout=0.1):
        super(ConvAttentionLayer, self).__init__()
        assert hid_dim % n_heads == 0
        self.n_heads = n_heads
        input_channels = hid_dim * 2 + pre_channels
        self.groups = groups
        self.activation = paddle.nn.LeakyReLU(negative_slope=0.1)
        self.linear1 = paddle.nn.Linear(in_features=hid_dim, out_features=
            hid_dim, bias_attr=False)
        self.linear2 = paddle.nn.Linear(in_features=hid_dim, out_features=
            hid_dim, bias_attr=False)
        self.conv = paddle.nn.Sequential(paddle.nn.Dropout2D(p=dropout),
            paddle.nn.Conv2D(in_channels=input_channels, out_channels=
            channels, kernel_size=1), paddle.nn.LeakyReLU(negative_slope=
            0.1), paddle.nn.Conv2D(in_channels=channels, out_channels=
            channels, kernel_size=3, padding=1), paddle.nn.LeakyReLU(
            negative_slope=0.1))
        self.score_layer = paddle.nn.Conv2D(in_channels=channels,
            out_channels=n_heads, kernel_size=1, bias_attr=False)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x, y, pre_conv=None, mask=None, residual=True,
        self_loop=True):
        ori_x, ori_y = x, y
        B, M, _ = x.shape
        B, N, _ = y.shape
        fea_map = paddle.concat(x=[x.unsqueeze(axis=2).repeat_interleave(
            repeats=N, axis=2), y.unsqueeze(axis=1).repeat_interleave(
            repeats=M, axis=1)], axis=-1).transpose(perm=[0, 3, 1, 2])
        if pre_conv is not None:
            fea_map = paddle.concat(x=[fea_map, pre_conv], axis=1)
        fea_map = self.conv(fea_map)
        scores = self.activation(self.score_layer(fea_map))
        if mask is not None:
            mask = mask.unsqueeze(axis=1).expand_as(y=scores)
            scores = paddle.where(mask.equal(y=0), scores, paddle.to_tensor(-90000000000.0, dtype='float32'))
        x = self.linear1(self.dropout(x))
        y = self.linear2(self.dropout(y))
        
        out_x = paddle.matmul(F.softmax(scores, axis=-1), y.reshape([B, N, self.n_heads, -1]).transpose([0, 2, 1, 3]))
        out_x = out_x.transpose([0, 2, 1, 3]).reshape([B, M, -1])
        
        out_y = paddle.matmul(F.softmax(scores.transpose([0, 1, 3, 2]), axis=-1), x.reshape([B, M, self.n_heads, -1]).transpose([0, 2, 1, 3]))
        out_y = out_y.transpose([0, 2, 1, 3]).reshape([B, N, -1])
        if self_loop:
            out_x = out_x + x
            out_y = out_y + y
        out_x = self.activation(out_x)
        out_y = self.activation(out_y)
        if residual:
            out_x = out_x + ori_x
            out_y = out_y + ori_y
        return out_x, out_y, fea_map


class ConvAttention(paddle.nn.Layer):

    def __init__(self, hid_dim, n_heads, pre_channels, channels, layers,
        groups, dropout):
        super(ConvAttention, self).__init__()
        self.layers = paddle.nn.LayerList(sublayers=[ConvAttentionLayer(
            hid_dim, n_heads, pre_channels if i == 0 else channels,
            channels, groups, dropout=dropout) for i in range(layers)])

    def forward(self, x, y, fea_map=None, mask=None, residual=True,
        self_loop=True):
        fea_list = []
        for layer in self.layers:
            x, y, fea_map = layer(x, y, fea_map, mask, residual, self_loop)
            fea_list.append(fea_map)
        return x, y, fea_map.transpose(perm=[0, 2, 3, 1])


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


class MLP(paddle.nn.Layer):

    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features=n_in, out_features=n_out)
        self.activation = paddle.nn.LeakyReLU(negative_slope=0.1)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class MRN(paddle.nn.Layer):

    def __init__(self, vocab_size, tok_emb_size, ner_emb_size, dis_emb_size,
        hid_size, channels, layers, chunk, dropout1, dropout2, embeddings=None
        ):
        super(MRN, self).__init__()
        self.chunk = chunk
        if embeddings is None:
            self.word_emb = paddle.nn.Embedding(num_embeddings=vocab_size, embedding_dim=tok_emb_size)
        self.ner_embs = paddle.nn.Embedding(num_embeddings=7, embedding_dim
            =ner_emb_size)
        self.dis_embs = paddle.nn.Embedding(num_embeddings=20,
            embedding_dim=dis_emb_size)
        self.ref_embs = paddle.nn.Embedding(num_embeddings=3, embedding_dim
            =dis_emb_size)
        emb_size = tok_emb_size + ner_emb_size
        self.encoder = paddle.nn.LSTM(input_size=emb_size, hidden_size=
            hid_size // 2, num_layers=1, time_major=not True, direction=
            'bidirectional')
        self.dropout1 = paddle.nn.Dropout(p=dropout1)
        self.dropout2 = paddle.nn.Dropout(p=dropout2)
        self.men2men_conv_att = ConvAttention(hid_size, 1, dis_emb_size,
            channels, groups=1, layers=layers, dropout=dropout1)
        self.mlp_sub = MLP(n_in=hid_size, n_out=hid_size, dropout=dropout2)
        self.mlp_obj = MLP(n_in=hid_size, n_out=hid_size, dropout=dropout2)
        self.biaffine = Biaffine(n_in=hid_size, n_out=97, bias_x=True,
            bias_y=True)
        self.mlp_rel = MLP(channels, channels, dropout=dropout2)
        self.linear = paddle.nn.Linear(in_features=channels, out_features=97)

    def forward(self, doc_inputs, ner_inputs, dis_inputs, ref_inputs,
        doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask):
        length = doc_inputs.not_equal(y=paddle.to_tensor(0)).sum(axis=-1)
        length = length.to('cpu')
        tok_embs = self.word_emb(doc_inputs)
        ner_embs = self.ner_embs(ner_inputs)
        tok_embs = self.dropout1(paddle.concat(x=[tok_embs, ner_embs], axis=-1))
        packed_embs = tok_embs
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        outs = packed_outs
        max_e = doc2ent_mask.shape[1]
        max_m = doc2men_mask.shape[1]
        min_value = paddle.min(x=outs).item()
        _outs = outs.unsqueeze(axis=1).expand(shape=[-1, max_m, -1, -1])
        
        _outs = masked_fill(_outs, doc2men_mask.equal(y=0).unsqueeze(axis=-1), min_value)
        
        men_reps, _ = paddle.max(x=_outs, axis=2), paddle.argmax(x=_outs, axis=2)
        dis_emb = self.dis_embs(dis_inputs).transpose(perm=[0, 3, 1, 2])
        x, y, fea_maps = self.men2men_conv_att(men_reps, men_reps, dis_emb, men2men_mask)
        min_x_value = paddle.min(x=x).item()
        x = x.unsqueeze(axis=1).expand(shape=[-1, max_e, -1, -1])
        
        x = masked_fill(x, men2ent_mask.equal(y=0).unsqueeze(axis=-1), min_x_value)

        x, _ = paddle.max(x=x, axis=2), paddle.argmax(x=x, axis=2)
        min_y_value = paddle.min(x=y).item()
        y = y.unsqueeze(axis=1).expand(shape=[-1, max_e, -1, -1])
        
        y = masked_fill(y, men2ent_mask.equal(y=0).unsqueeze(axis=-1), min_y_value)

        y, _ = paddle.max(x=y, axis=2), paddle.argmax(x=y, axis=2)
        min_f_value = paddle.min(x=fea_maps).item()
        fea_list = []
        chunk = self.chunk
        fea_maps = paddle_aux.split(x=fea_maps, num_or_sections=chunk, axis=0)
        m2e_mask2 = paddle_aux.split(x=men2ent_mask, num_or_sections=chunk, axis=0)
        for fea_map, m2e_mask in zip(fea_maps, m2e_mask2):
            fea_map = fea_map.unsqueeze(axis=1).repeat(1, max_e, 1, 1, 1)
            
            fea_map = masked_fill(fea_map, m2e_mask.equal(y=0)[:, :, :, None, None], min_f_value)

            fea_map, _ = paddle.max(x=fea_map, axis=2), paddle.argmax(x=fea_map, axis=2)
            fea_map = fea_map.unsqueeze(axis=1).repeat(1, max_e, 1, 1, 1)
            
            fea_map = masked_fill(fea_map, m2e_mask.equal(y=0)[:, :, None, :, None], min_f_value)

            fea_map, _ = paddle.max(x=fea_map, axis=3), paddle.argmax(x=fea_map, axis=3)
            fea_list.append(fea_map)
        fea_maps = paddle.concat(x=fea_list, axis=0)
        ent_sub = self.dropout2(self.mlp_sub(x))
        ent_obj = self.dropout2(self.mlp_obj(y))
        rel_outputs1 = self.biaffine(ent_sub, ent_obj)
        fea_maps = self.dropout2(self.mlp_rel(fea_maps))
        rel_outputs2 = self.linear(fea_maps)
        return rel_outputs1 + rel_outputs2
