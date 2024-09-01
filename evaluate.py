import os
import string
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import time
from utils import AttnLabelConverter

from dataset import RawDataset, AlignCollate
from model import (
    make_std_mask, make_dissym_mask,
    Pre_Encoder, Encoder, InteractiveDecoding_Decoder
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

def interactive_decoding_eval(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        src_converter = AttnLabelConverter(opt.src_character)
        tgt_converter = AttnLabelConverter(opt.tgt_character)

    opt.src_num_class = len(src_converter.character)
    opt.tgt_num_class = len(tgt_converter.character)

    if opt.rgb:
        opt.input_channel = 3
    pre_encoder = Pre_Encoder(opt)
    encoder = Encoder(opt)
    decoder = InteractiveDecoding_Decoder(opt)
    model_list = [pre_encoder, encoder, decoder]
    
    pre_encoder = torch.nn.DataParallel(pre_encoder).to(device)
    encoder = torch.nn.DataParallel(encoder).to(device)
    decoder = torch.nn.DataParallel(decoder).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    pre_encoder.load_state_dict(torch.load(opt.saved_model + '_' + opt.saved_iter + '_' + 'pre_encoder' +'.pth', map_location=device))
    encoder.load_state_dict(torch.load(opt.saved_model + '_' + opt.saved_iter + '_' + 'encoder' +'.pth', map_location=device))
    decoder.load_state_dict(torch.load(opt.saved_model + '_' + opt.saved_iter + '_' + 'decoder' +'.pth', map_location=device))

    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    pre_encoder.eval()
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        idx = 1
        for image_tensors, image_path_list in demo_loader:
            print('Now decoding {}'.format(idx))
            idx += 1
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)

            src_length_for_pred = torch.IntTensor([opt.src_batch_max_length] * batch_size).to(device)
            tgt_length_for_pred = torch.IntTensor([opt.tgt_batch_max_length] * batch_size).to(device)
            src_text_for_pred = torch.LongTensor(batch_size, opt.src_batch_max_length + 1).fill_(0).to(device)
            tgt_text_for_pred = torch.LongTensor(batch_size, opt.tgt_batch_max_length + 1).fill_(0).to(device)

            valid_src_mask = src_mask = make_std_mask(src_text_for_pred[:, :], pad = 2)[0]
            valid_tgt_mask = tgt_mask = make_std_mask(tgt_text_for_pred[:, :], pad = 2)[0]
            src_tgt_mask = make_dissym_mask(src_text_for_pred[:, :].size(-1), tgt_text_for_pred[:, :].size(-1), pad = 2)[0]
            tgt_src_mask = make_dissym_mask(tgt_text_for_pred[:, :].size(-1), src_text_for_pred[:, :].size(-1), pad = 2)[0]
            opt.src_mask = src_mask
            opt.tgt_mask = tgt_mask
            opt.src_tgt_mask = src_tgt_mask
            opt.tgt_src_mask = tgt_src_mask
            
            start_symbol = 0

            src_preds = src_text_for_pred
            tgt_preds = tgt_text_for_pred
            src_decoder_input = src_text_for_pred
            tgt_decoder_input = tgt_text_for_pred

            for i in range(opt.batch_max_length + 1):
                print('Decoding position {} ...'.format(i))

                visual_feature = model_list[0](image, tgt_decoder_input.long(), tgt_mask = valid_tgt_mask, is_train=False)
                contextual_feature = model_list[1](visual_feature, image, tgt_decoder_input.long(), tgt_mask = valid_tgt_mask, is_train=False)
                src_preds, tgt_preds = model_list[2](opt, contextual_feature = contextual_feature, input = image, src_text = src_decoder_input.long(), tgt_text = tgt_decoder_input.long(), src_mask = src_mask, tgt_mask = tgt_mask, lmd=opt.interactive_lambda, type=opt.interactive_type)
                
                _, src_preds_index = src_preds.max(2)
                _, tgt_preds_index = tgt_preds.max(2)

                if i+1 < opt.src_batch_max_length + 1:
                    src_decoder_input[:, i+1] = src_preds_index[:, i]
                if i+1 < opt.tgt_batch_max_length + 1:
                    tgt_decoder_input[:, i+1] = tgt_preds_index[:, i]


                _, src_preds_index = src_preds.max(2)
                src_preds_str = src_converter.decode(src_preds_index, src_length_for_pred, opt.src_level)
                
                _, tgt_preds_index = tgt_preds.max(2)
                tgt_preds_str = tgt_converter.decode(tgt_preds_index, tgt_length_for_pred, opt.tgt_level)

            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            
            # Writting Source Results
            src_preds_prob = F.softmax(src_preds, dim=2)
            src_preds_max_prob, _ = src_preds_prob.max(dim=2)
            log = open(f'{opt.src_output}', 'a', encoding='utf-8')
            for img_name, pred, pred_max_prob in zip(image_path_list, src_preds_str, src_preds_max_prob):

                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

                print(f'{img_name:25s}\t{pred:25s}')
                log.write(f'{pred}\n')
            log.close()

            # Writting Target Results
            tgt_preds_prob = F.softmax(tgt_preds, dim=2)
            tgt_preds_max_prob, _ = tgt_preds_prob.max(dim=2)
            log = open(f'{opt.tgt_output}', 'a', encoding='utf-8')
            for img_name, pred, pred_max_prob in zip(image_path_list, tgt_preds_str, tgt_preds_max_prob):

                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

                print(f'{img_name:25s}\t{pred:25s}')
                log.write(f'{pred}\n')
            log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default=None, help="Task parameter: None for normal training; others for special tasks")
    
    parser.add_argument('--image_folder', default=None, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--src_batch_max_length', type=int, default=25, help='maximum-label-length of src')
    parser.add_argument('--tgt_batch_max_length', type=int, default=25, help='maximum-label-length of tgt')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--saved_iter', required=True, help="iter step when saving model")
    
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    
    parser.add_argument('--Transformation', type=str, required=False, default='TPS', help='Transformation stage.')
    parser.add_argument('--FeatureExtraction', type=str, required=False, default='ResNoLSTM', help='FeatureExtraction stage.')
    parser.add_argument('--SequenceModeling', type=str, required=False, default='TransformerEncoder', help='SequenceModeling stage.')
    parser.add_argument('--Prediction', type=str, required=False, default='TransformerDecoder', help='Prediction stage.')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--data_format', type=str, default="pic", help='pic or lmdb')

    parser.add_argument('--src_vocab', required=True, help='path to source vocab')
    parser.add_argument('--tgt_vocab', required=True, help='path to target vocab')
    
    parser.add_argument('--src_level', type=str, default="char", help='char or word of src')
    parser.add_argument('--tgt_level', type=str, default="char", help='char or word of tgt')

    parser.add_argument("--interactive_lambda", type = float, default = 0.5)
    parser.add_argument("--interactive_type", type = str, default = 'hierarchical')
    parser.add_argument("--input_modal_num", type = int, default = 3)
    parser.add_argument("--vocab_path", type = str, default = './')
    parser.add_argument("--batch_max_length", type = int, default = 80)
    parser.add_argument("--level", type = str, default = 'char')

    parser.add_argument('--src_output', help='output the source result log', default='log_test_result.txt')
    parser.add_argument('--tgt_output', help='output the target result log', default='log_test_result.txt')

    parser.add_argument('--test_data', required=True, help='path to test dataset')

    opt = parser.parse_args()

    def pic_file_texts(file):
        id=[]
        for line in open(file, 'r', encoding='utf-8'):
            line = line.replace("\n", "")
            id.append(line)
        return id

    if not opt.src_vocab:
    	print('not find src vocab!')
    src_dict_ = pic_file_texts(opt.src_vocab)
    opt.src_character = src_dict_

    if not opt.tgt_vocab:
    	print('not find tgt vocab!')
    tgt_dict_ = pic_file_texts(opt.tgt_vocab)
    opt.tgt_character = tgt_dict_

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    if opt.task == 'interactive_decoding':
        interactive_decoding_eval(opt)
    
    else:
        print('Input task {} is not defined. Please check your task name.'.format(opt.task))
        exit()
        