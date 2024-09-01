# coding: utf-8
import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset_3, AlignCollate_3, Batch_Balanced_Dataset_3
from model import (
    make_std_mask, make_dissym_mask,
    Pre_Encoder, Encoder, InteractiveDecoding_Decoder
)
from validate import validation_interactive_decoding

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print('Stopped! Because Only CPU could be available ...')
    exit()
else:
    print('Now is using device: {}'.format(device))


def interactive_decoding_train(opt):
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    print('-' * 80)
    print('Loading Train Dataset ...')
    train_dataset = Batch_Balanced_Dataset_3(opt)

    log = open(f'{opt.saved_model}/{opt.exp_name}/log_dataset.txt', 'a', encoding='utf-8')
    AlignCollate_valid = AlignCollate_3(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

    print('-' * 80)
    print('Loading Valid Dataset ...')
    valid_dataset, valid_dataset_log = hierarchical_dataset_3(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    print('Length of valid_dataset: {}'.format(len(valid_dataset)))

    """ model configuration """
    print('-' * 80)
    print('Now in model configuration')
    if 'CTC' in opt.Prediction:
        src_converter = CTCLabelConverter(opt.src_character)
        tgt_converter = CTCLabelConverter(opt.tgt_character)
    else:
        src_converter = AttnLabelConverter(opt.src_character)
        tgt_converter = AttnLabelConverter(opt.tgt_character)
    opt.src_num_class = len(src_converter.character)
    opt.tgt_num_class = len(tgt_converter.character)

    if opt.rgb:
        opt.input_channel = 3
    
    # Construct Model information 
    pre_encoder = Pre_Encoder(opt)
    encoder = Encoder(opt)
    decoder = InteractiveDecoding_Decoder(opt)
    model_list = [pre_encoder, encoder, decoder]

    # weight initialization
    print('-' * 80)
    print('Now in weight initialization')
    print('Print all name in model.named_parameters: ')
    for sub_model in model_list:
        for name, param in sub_model.named_parameters():
            print('=' * 50)
            print(name)
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'Transformer_encoder_layer' in name or 'Transformer_decoder_layer' in name \
                    or 'TransformerDecoder' in name or 'SequenceModeling' in name:
                    if param.dim() > 1:
                        print('init {} with xavier_uniform.'.format(name))
                        init.xavier_uniform_(param)
                        continue
            except:
                pass
            try:
                if 'bias' in name:
                    print('Constant init for {} bias'.format(name))
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    print('kaiming_normal_ init for {} weight'.format(name))
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    print('fill_(1) init for {} weight'.format(name))
                    param.data.fill_(1)
                continue

    # data parallel for multi-GPU
    pre_encoder = torch.nn.DataParallel(pre_encoder).to(device)
    encoder = torch.nn.DataParallel(encoder).to(device)
    decoder = torch.nn.DataParallel(decoder).to(device)
    
    pre_encoder.train()
    encoder.train()
    decoder.train()

    print('-' * 80)
    print("Model:")
    for sub_model in model_list:
        print(sub_model)
    

    """ setup loss """
    print('-' * 80)
    print('Now in setup loss')
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()
    src_loss_avg = Averager()
    tgt_loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for sub_model in model_list:
        for p in filter(lambda p: p.requires_grad, sub_model.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
    print('Trainable params num : {}'.format(sum(params_num)))

    # setup optimizer
    print('-' * 80)
    print('Now in setup optimizer')
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    print('-' * 80)
    print('Now in final options')
    with open(f'{opt.saved_model}/{opt.exp_name}/opt.txt', 'a', encoding='utf-8') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    print('-' * 80)
    print('Now in start training')

    start_time = time.time()
    best_accuracy = -1
    best_bleu = -1
    best_norm_ED = -1
    best_valid_loss = 1000000
    iteration = -1
    previous_best_accuracy_iter = 0
    previous_best_bleu_iter = 0
    previous_best_valid_iter = 0
    
    old_time = time.time()
    while(True):
        # train part
        iteration += 1
        image_tensors, src_labels, tgt_labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        src_text, _ = src_converter.encode(src_labels, opt.src_level,  batch_max_length=opt.src_batch_max_length)
        tgt_text, tgt_length = tgt_converter.encode(tgt_labels, opt.tgt_level,  batch_max_length=opt.tgt_batch_max_length)

        src_mask = make_std_mask(src_text[:, :-1], pad = 2)[0]
        tgt_mask = make_std_mask(tgt_text[:, :-1], pad = 2)[0]
        src_tgt_mask = make_dissym_mask(src_text[:, :-1].size(-1), tgt_text[:, :-1].size(-1), pad = 2)[0]
        tgt_src_mask = make_dissym_mask(tgt_text[:, :-1].size(-1), src_text[:, :-1].size(-1), pad = 2)[0]
        
        opt.src_mask = src_mask
        opt.tgt_mask = tgt_mask
        opt.src_tgt_mask = src_tgt_mask
        opt.tgt_src_mask = tgt_src_mask

        visual_feature = pre_encoder(input = image, text = src_text[:, :-1], tgt_mask = tgt_mask)
        contextual_feature = encoder(visual_feature, input = image, text = src_text[:, :-1], tgt_mask = tgt_mask)
        src_preds, tgt_preds = decoder(opt, contextual_feature = contextual_feature, input = image, src_text = src_text[:, :-1], tgt_text = tgt_text[:, :-1], src_mask = src_mask, tgt_mask = tgt_mask, lmd=opt.interactive_lambda, type=opt.interactive_type)
        
        # Save ground truth results
        src_target = src_text[:, 1:]  # without [GO] Symbol
        tgt_target = tgt_text[:, 1:]  # without [GO] Symbol

        # In Deep-text original code, using logit to calculate loss directly
        src_cost = criterion(src_preds.contiguous().view(-1, src_preds.shape[-1]), src_target.contiguous().view(-1))
        tgt_cost = criterion(tgt_preds.contiguous().view(-1, tgt_preds.shape[-1]), tgt_target.contiguous().view(-1))
        
        cost_weight = opt.Weight_lambda # cost_weight is used to weight the loss of OCR and OCR-MT
        cost = cost_weight * src_cost + (1.0 - cost_weight) * tgt_cost
        
        pre_encoder.zero_grad()
        encoder.zero_grad()
        decoder.zero_grad()

        cost.backward()
        torch.nn.utils.clip_grad_norm_(pre_encoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)

        optimizer.step()

        loss_avg.add(src_cost)
        loss_avg.add(tgt_cost)
        src_loss_avg.add(src_cost)
        tgt_loss_avg.add(tgt_cost)
        
        # print loss at each step ...
        duration_time = time.time() - old_time
        print_str=f'step = {iteration+1}, loss = {loss_avg.val():0.5f}, src_loss = {src_loss_avg.val():0.5f}, tgt_loss = {tgt_loss_avg.val():0.5f}, duration = {duration_time:0.2f}s'
        old_time = time.time()
        print(print_str)
        
        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            print('-' * 80)
            print('Now in validation on iteration {} ...'.format(iteration + 1))
            elapsed_time = time.time() - start_time
            with open(f'{opt.saved_model}/{opt.exp_name}/log_train.txt', 'a', encoding='utf-8') as log:
                pre_encoder.eval()
                encoder.eval()
                decoder.eval()

                with torch.no_grad():
                    valid_loss, current_accuracy, current_bleu, src_preds_str, tgt_preds_str, src_labels, tgt_labels, infer_time, length_of_data = validation_interactive_decoding(
                        model_list, criterion, valid_loader, src_converter, tgt_converter, opt)
                    
                pre_encoder.train()
                encoder.train()
                decoder.train()

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()
                
                current_model_log = f'{"Current_valid_loss":17s}: {valid_loss:0.5f}, {"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_bleu":17s}: {current_bleu:0.3f}'

                # keep best valid loss model (on valid dataset)
                print('Current valid_loss: {}'.format(valid_loss))
                print('Current best_valid_loss: {}'.format(best_valid_loss))
                if valid_loss <= best_valid_loss:
                    print('Saving best_valid_loss model ...')
                    best_valid_loss = valid_loss
                    torch.save(pre_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'pre_encoder' + '.pth')
                    torch.save(encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'encoder' + '.pth')
                    torch.save(decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'decoder' + '.pth')

                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'pre_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'pre_encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'decoder' + '.pth')
                    
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'pre_encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'decoder' + '.pth')

                    previous_best_valid_iter = iteration + 1
                
                # keep best accuracy model (on valid dataset)    
                print('Current accuracy: {}'.format(current_accuracy))
                print('Current best_accuracy: {}'.format(best_accuracy))
                if current_accuracy >= best_accuracy:
                    print('Saving best_accuracy model ...')
                    best_accuracy = current_accuracy
                    torch.save(pre_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'pre_encoder' + '.pth')
                    torch.save(encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'encoder' + '.pth')
                    torch.save(decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'decoder' + '.pth')

                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'pre_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'pre_encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'decoder' + '.pth')

                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'pre_encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'decoder' + '.pth')
                    
                    previous_best_accuracy_iter = iteration + 1
                
                # keep best bleu model (on valid dataset)    
                print('Current bleu: {}'.format(current_bleu))
                print('Current best_bleu: {}'.format(best_bleu))
                if current_bleu >= best_bleu:
                    print('Saving best_bleu model ...')
                    best_bleu = current_bleu
                    torch.save(pre_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'pre_encoder' + '.pth')
                    torch.save(encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'encoder' + '.pth')
                    torch.save(decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'decoder' + '.pth')

                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'pre_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'pre_encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'decoder' + '.pth')

                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'pre_encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'decoder' + '.pth')
                    
                    previous_best_bleu_iter = iteration + 1
                
                best_model_log = f'{"Best_valid":17s}: {best_valid_loss:0.5f}, {"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_bleu":17s}: {best_bleu:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')


        ######################################################################

        # save model per opt.saveInterval, deep-text originally 1e+5 iter
        if (iteration + 1) % opt.saveInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            print('-' * 80)
            print('Saving model on set step of {} ...'.format(iteration + 1))
            torch.save(pre_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'pre_encoder' + '.pth')
            torch.save(encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'encoder' + '.pth')
            torch.save(decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'decoder' + '.pth')
            
        
        # Final Step and offer information
        if (iteration + 1) == opt.num_iter:
            print('Remove iter_step_1_* model savings, which is just a model saving to see whether it could run normally.')
            os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'pre_encoder' + '.pth')
            os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'encoder' + '.pth')
            os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'decoder' + '.pth')
            print('end the training at step {}!'.format(iteration + 1))
            sys.exit()
        
