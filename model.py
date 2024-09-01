
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import (
    BidirectionalLSTM, PositionalEncoding, TransformerEncoder, TransformerEncoderLayer,
    InteractiveDecodingTransformerDecoder, InteractiveDecodingTransformerDecoderLayer,
)

from modules.prediction import Attention

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info


class Pre_Encoder(nn.Module):

    def __init__(self, opt):
        super(Pre_Encoder, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.task == 'vtmt_os':
            self.FeatureExtraction = Embeddings(opt.hidden_size, opt.src_num_class)
            self.make_feature_dim = nn.Linear(opt.src_batch_max_length+1, 26)
            return
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'Textual':
            self.FeatureExtraction = Embeddings(opt.hidden_size, opt.src_num_class)
            self.make_feature_dim = nn.Linear(opt.src_batch_max_length+1, 26)
            return
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        
        self.cv_bi_lstm = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.cv_bi_lstm_output = opt.hidden_size    # Used to initialize later layer

    def forward(self, input, text, tgt_mask, is_train=True):

        """ Transformation stage """
        
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        if self.stages['Feat'] == 'Textual':
            visual_feature = self.FeatureExtraction(text)
            visual_feature = visual_feature.permute(0, 2, 1)
            visual_feature = self.make_feature_dim(visual_feature).permute(0, 2, 1)
        else:
            visual_feature = self.FeatureExtraction(input)

            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
            
            visual_feature = visual_feature.squeeze(3)
            
            visual_feature = self.cv_bi_lstm(visual_feature)
        return visual_feature

class Encoder(nn.Module):

    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}
        
        self.FeatureExtraction_output = opt.output_channel 

        """ Sequence modeling"""
        if opt.SequenceModeling == 'TransformerEncoder':
            
            self.SequenceModeling_input = opt.hidden_size
            self.SequenceModeling_output = opt.hidden_size

            self.EncoderPositionalEmbedding = PositionalEncoding(d_model=self.SequenceModeling_output, dropout = 0, max_len = max(opt.src_batch_max_length, opt.tgt_batch_max_length) + 2)
            self.Transformer_encoder_layer = TransformerEncoderLayer(d_model=self.SequenceModeling_input, nhead=8)
            self.SequenceModeling = TransformerEncoder(self.Transformer_encoder_layer, num_layers=6)

        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = opt.hidden_size


    def forward(self, visual_feature, input, text, tgt_mask, is_train=True):

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'TransformerEncoder':
            visual_feature = self.EncoderPositionalEmbedding(visual_feature)

            batch_mid_variable = visual_feature.permute(1, 0, 2)    # Make batch dimension in the middle

            contextual_feature = self.SequenceModeling(src=batch_mid_variable)

            contextual_feature = contextual_feature.permute(1, 0, 2)
        
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM
        return contextual_feature

class InteractiveDecoding_Decoder(nn.Module):

    def __init__(self, opt):
        super(InteractiveDecoding_Decoder, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Prediction """
        if opt.Prediction == 'TransformerDecoder':
            pass
            self.Prediction_input = opt.hidden_size
            self.Prediction_output = opt.hidden_size
            
            self.src_embedding = Embeddings(opt.hidden_size, opt.src_num_class)
            self.tgt_embedding = Embeddings(opt.hidden_size, opt.tgt_num_class)

            self.src_DecoderPositionalEmbedding = PositionalEncoding(d_model=self.Prediction_output, dropout = 0, max_len = opt.src_batch_max_length)
            self.tgt_DecoderPositionalEmbedding = PositionalEncoding(d_model=self.Prediction_output, dropout = 0, max_len = opt.tgt_batch_max_length)
            
            self.Transformer_decoder_layer = InteractiveDecodingTransformerDecoderLayer(d_model=opt.hidden_size, nhead=8)
            self.Prediction_TransformerDecoder = InteractiveDecodingTransformerDecoder(self.Transformer_decoder_layer, num_layers=6, opt = opt, src_output_dim = opt.src_num_class, tgt_output_dim = opt.tgt_num_class)

        else:
            raise Exception('Please set Prediction correctly')

    def forward(self, opt, contextual_feature, input, src_text, tgt_text, src_mask, tgt_mask, lmd, type, is_train=True):

        """ Prediction stage """
        if self.stages['Pred'] == 'TransformerDecoder':

            src_pred_feature = self.src_DecoderPositionalEmbedding(contextual_feature)
            tgt_pred_feature = self.tgt_DecoderPositionalEmbedding(contextual_feature)

            src_pred_feature = src_pred_feature.permute(1, 0, 2)    # Make batch dimension in the middle
            tgt_pred_feature = tgt_pred_feature.permute(1, 0, 2)    # Make batch dimension in the middle
            

            src_text_input = self.src_embedding(src_text).permute(1, 0, 2)
            tgt_text_input = self.tgt_embedding(tgt_text).permute(1, 0, 2)

            pred_feature, pred_feature2 = self.Prediction_TransformerDecoder(opt, src = src_text_input, tgt = tgt_text_input, memory = src_pred_feature, lmd=lmd,  type=type, src_mask = src_mask, tgt_mask = tgt_mask, is_train = is_train)

            pred_feature = pred_feature.permute(1, 0, 2)    # Make batch dimension in the top
            pred_feature2 = pred_feature2.permute(1, 0, 2)    # Make batch dimension in the top
            
            # Explanation of prediction: probability distribution at each step
            # Size of prediction: [batch_size x num_steps x num_classes]
            prediction = pred_feature
            prediction2 = pred_feature2
        
        return prediction, prediction2

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
        # Updated mask methods.
        # Generate mask without considering pad information
        tgt_mask = subsequent_mask(tgt.size(-1))
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        return Variable(tgt_mask.cuda(), requires_grad=False)

def dissym_subsequent_mask(size1, size2):
    attn_shape = (1, size1, size2)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_dissym_mask(size1, size2, pad):        
        # Updated mask methods.
        # Generate mask without considering pad information
        tgt_mask = dissym_subsequent_mask(size1, size2)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        return Variable(tgt_mask.cuda(), requires_grad=False)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

