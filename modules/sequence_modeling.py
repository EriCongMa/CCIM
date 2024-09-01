import torch
import math
import copy
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class TransformerEncoder(Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output

class TransformerDecoder(Module):

    def __init__(self, decoder_layer, num_layers, opt, output_dim=None, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        if output_dim is None:
            output_dim = opt.num_class
        self.generator = nn.Linear(opt.hidden_size, output_dim)

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, is_train=True):
        if is_train:
            output = tgt
            for i in range(self.num_layers):
                output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)
            
            if self.norm:
                output = self.norm(output)
                
            output = self.generator(output)
            
            return output
        
        else:
            output = tgt
            for i in range(self.num_layers):
                output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)

            if self.norm:
                output = self.norm(output)
                
            output = self.generator(output)

            return output

class TransformerEncoderLayer(Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if len(tgt_mask.size()) > 2:
            tgt_mask = tgt_mask[0, :, :]

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class InteractiveDecodingTransformerDecoder(Module):

    def __init__(self, decoder_layer, num_layers, opt, src_output_dim=None, tgt_output_dim=None, src_norm=None, tgt_norm=None):
        super(InteractiveDecodingTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm_src = src_norm
        self.norm_tgt = tgt_norm
        if src_output_dim is None:
            src_output_dim = opt.src_num_class
        if tgt_output_dim is None:
            tgt_output_dim = opt.tgt_num_class
        self.generator_src = nn.Linear(opt.hidden_size, src_output_dim)
        self.generator_tgt = nn.Linear(opt.hidden_size, tgt_output_dim)

    def forward(self, opt, src, tgt, memory, lmd=1,  type='hierarchical', src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, is_train=True):
        lmd = opt.interactive_lambda
        src_output = src
        tgt_output = tgt
        for i in range(self.num_layers):
            src_output, tgt_output = self.layers[i](opt, src_output, tgt_output, memory, src_mask=src_mask, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    src_key_padding_mask=src_key_padding_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask, lmd=lmd, type=type)
        
        if self.norm_src:
            src_output = self.norm_src(src_output)
        if self.norm_tgt:
            tgt_output = self.norm_tgt(tgt_output)
            
        src_output = self.generator_src(src_output)
        tgt_output = self.generator_tgt(tgt_output)
        
        return src_output, tgt_output

class InteractiveDecodingTransformerDecoderLayer(Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(InteractiveDecodingTransformerDecoderLayer, self).__init__()
        self.self_attn_src = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_tgt = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_src_tgt = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_src_tgt2 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_src_mem = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_tgt_src = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_tgt_mem = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_tgt_src2 = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.dropout_src = Dropout(dropout)
        self.linear1_src = Linear(d_model, dim_feedforward)
        self.linear2_src = Linear(dim_feedforward, d_model)

        self.norm1_src = LayerNorm(d_model)
        self.norm2_src = LayerNorm(d_model)
        self.norm3_src = LayerNorm(d_model)
        self.dropout1_src = Dropout(dropout)
        self.dropout2_src = Dropout(dropout)
        self.dropout3_src = Dropout(dropout)

        self.dropout_tgt = Dropout(dropout)
        self.linear1_tgt = Linear(d_model, dim_feedforward)
        self.linear2_tgt = Linear(dim_feedforward, d_model)

        self.norm1_tgt = LayerNorm(d_model)
        self.norm2_tgt = LayerNorm(d_model)
        self.norm3_tgt = LayerNorm(d_model)
        self.dropout1_tgt = Dropout(dropout)
        self.dropout2_tgt = Dropout(dropout)
        self.dropout3_tgt = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, opt, src, tgt, memory, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, lmd=1, type='hierarchical'):

        src_self_hidden = self.self_attn_src(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        tgt_self_hidden = self.self_attn_tgt(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        
        #########################
        # Interactive decoding between self-attentin and cross-task attention
        if type == 'hierarchical':
            fusion_src_tgt_hidden = self.multihead_attn_src_tgt(src_self_hidden, tgt, tgt, attn_mask=opt.src_tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
            fusion_tgt_src_hidden = self.multihead_attn_tgt_src(tgt_self_hidden, src, src, attn_mask=opt.tgt_src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        
        elif type == 'weighted':
            src_tgt_hidden = self.multihead_attn_src_tgt(src, tgt, tgt, attn_mask=opt.src_tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
            tgt_src_hidden = self.multihead_attn_tgt_src(tgt, src, src, attn_mask=opt.tgt_src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
            
            assert src_self_hidden.shape == src_tgt_hidden.shape
            assert tgt_self_hidden.shape == tgt_src_hidden.shape

            fusion_src_tgt_hidden = src_self_hidden + lmd * src_tgt_hidden
            fusion_tgt_src_hidden = tgt_self_hidden + lmd * tgt_src_hidden
        
        else:
            raise 'Please input a valid interactive type.'
        
        src_hidden = src + self.dropout1_src(fusion_src_tgt_hidden)
        tgt_hidden = tgt + self.dropout1_tgt(fusion_tgt_src_hidden)

        src_hidden = self.norm1_src(src_hidden)
        tgt_hidden = self.norm1_tgt(tgt_hidden)

        fusion_src_mem_tgt_hidden = self.multihead_attn_src_mem(src_hidden, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        fusion_tgt_mem_src_hidden = self.multihead_attn_tgt_mem(tgt_hidden, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        src_hidden = src_hidden + self.dropout2_src(fusion_src_mem_tgt_hidden)
        tgt_hidden = tgt_hidden + self.dropout2_tgt(fusion_tgt_mem_src_hidden)

        src_hidden = self.norm2_src(src_hidden)
        tgt_hidden = self.norm2_tgt(tgt_hidden)

        if hasattr(self, "activation"):
            src_linear_hidden = self.linear2_src(self.dropout_src(self.activation(self.linear1_src(src_hidden))))
            tgt_linear_hidden = self.linear2_tgt(self.dropout_tgt(self.activation(self.linear1_tgt(tgt_hidden))))
        else:
            src_linear_hidden = self.linear2_src(self.dropout_src(F.relu(self.linear1_src(src_hidden))))
            tgt_linear_hidden = self.linear2_tgt(self.dropout_tgt(F.relu(self.linear1_tgt(tgt_hidden))))
        
        src_hidden = src_hidden + self.dropout3_src(src_linear_hidden)
        tgt_hidden = tgt_hidden + self.dropout3_tgt(tgt_linear_hidden)
        
        src_hidden = self.norm3_src(src_hidden)
        tgt_hidden = self.norm3_tgt(tgt_hidden)
        
        return src_hidden, tgt_hidden

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)
