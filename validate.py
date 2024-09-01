import os
import time
import string
import argparse
import re
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk_bleu import doc_bleu
from utils import Averager
from model import make_std_mask

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validation_interactive_decoding(model_list, criterion, evaluation_loader, src_converter, tgt_converter, opt):
    """ validation or evaluation """
    print('Now in evaluation of interactive decoding ...')
    n_correct = 0
    n_total = 0
    length_of_data = 0
    infer_time = 0
    ref_sents = []
    pred_sents = []
    valid_loss_avg = Averager()
    
    for i, (image_tensors, src_labels, tgt_labels) in enumerate(evaluation_loader):
        print('Now in batch {}'.format(i + 1))
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        src_length_for_pred = torch.IntTensor([opt.src_batch_max_length] * batch_size).to(device)
        tgt_length_for_pred = torch.IntTensor([opt.tgt_batch_max_length] * batch_size).to(device)
        length_for_pred = tgt_length_for_pred
        src_text_for_pred = torch.LongTensor(batch_size, opt.src_batch_max_length + 1).fill_(0).to(device)
        tgt_text_for_pred = torch.LongTensor(batch_size, opt.tgt_batch_max_length + 1).fill_(0).to(device)
        src_text_for_loss, src_length_for_loss = src_converter.encode(src_labels, opt.src_level, batch_max_length=opt.src_batch_max_length)
        tgt_text_for_loss, tgt_length_for_loss = tgt_converter.encode(tgt_labels, opt.tgt_level, batch_max_length=opt.tgt_batch_max_length)
        text_for_loss = tgt_text_for_loss
        length_for_loss = tgt_length_for_loss
        valid_tgt_mask = make_std_mask(tgt_text_for_loss[:, :-1], pad = 2)
        valid_src_mask = opt.src_mask
        valid_tgt_mask = opt.tgt_mask
        
        start_time = time.time()
            
        start_symbol = 0
        src_preds = src_text_for_pred
        tgt_preds = tgt_text_for_pred
        src_decoder_input = src_text_for_pred
        tgt_decoder_input = tgt_text_for_pred

        for i in range(opt.tgt_batch_max_length + 1):

            visual_feature = model_list[0](image, src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
            contextual_feature = model_list[1](visual_feature, image, src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
            src_preds, tgt_preds = model_list[2](opt, contextual_feature = contextual_feature, input = image, src_text = src_decoder_input.long(), tgt_text = tgt_decoder_input.long(), src_mask = valid_src_mask, tgt_mask = valid_tgt_mask, lmd=opt.interactive_lambda, type=opt.interactive_type, is_train=False)
        
            _, src_preds_index = src_preds.max(2)
            _, tgt_preds_index = tgt_preds.max(2)
            
            if i+1 < opt.src_batch_max_length + 1:
                src_decoder_input[:, i+1] = src_preds_index[:, i]
            if i+1 < opt.tgt_batch_max_length + 1:
                tgt_decoder_input[:, i+1] = tgt_preds_index[:, i]

        forward_time = time.time() - start_time
        
        src_preds = src_preds[:, :src_text_for_loss.shape[1] - 1, :]
        tgt_preds = tgt_preds[:, :tgt_text_for_loss.shape[1] - 1, :]
        
        src_target = src_text_for_loss[:, 1:]  # without [GO] Symbol
        tgt_target = tgt_text_for_loss[:, 1:]  # without [GO] Symbol
        src_cost = criterion(src_preds.contiguous().view(-1, src_preds.shape[-1]), src_target.contiguous().view(-1))
        tgt_cost = criterion(tgt_preds.contiguous().view(-1, tgt_preds.shape[-1]), tgt_target.contiguous().view(-1))

        # select max probabilty (greedy decoding) then decode index to character
        _, src_preds_index = src_preds.max(2)
        src_preds_str = src_converter.decode(src_preds_index, src_length_for_pred, opt.src_level)
        src_labels = src_converter.decode(src_text_for_loss[:, 1:], src_length_for_loss, opt.src_level)
        
        _, tgt_preds_index = tgt_preds.max(2)
        tgt_preds_str = tgt_converter.decode(tgt_preds_index, tgt_length_for_pred, opt.tgt_level)
        tgt_labels = tgt_converter.decode(tgt_text_for_loss[:, 1:], tgt_length_for_loss, opt.tgt_level)
        
        infer_time += forward_time
        valid_loss_avg.add(src_cost)
        valid_loss_avg.add(tgt_cost)

        # calculate accuracy & confidence score
        src_preds_prob = F.softmax(src_preds, dim=2)
        src_preds_max_prob, _ = src_preds_prob.max(dim=2)

        tgt_preds_prob = F.softmax(tgt_preds, dim=2)
        tgt_preds_max_prob, _ = tgt_preds_prob.max(dim=2)

        # confidence_score_list = []
        for gt, pred, pred_max_prob in zip(tgt_labels, tgt_preds_str, tgt_preds_max_prob):

            ref_sents.append(gt)
            pred_sents.append(pred)
            
            # Updated accuracy calculation
            # If the characters in gt occurs in pred, a positive support is given
            for item in gt:
                n_total += 1
                if item in pred:
                    n_correct += 1

    accuracy = n_correct / float(n_total) * 100

    tok_bleu, char_bleu = doc_bleu(ref_sents, pred_sents)

    return valid_loss_avg.val(), accuracy, tok_bleu, src_preds_str, tgt_preds_str, src_labels, tgt_labels, infer_time, length_of_data
