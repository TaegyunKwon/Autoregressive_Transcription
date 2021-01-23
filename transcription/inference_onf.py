
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .core import dataset as dataset_module
from .core import *
from .core import models, decoding, midi, representation
from .core.utils import draw_predictions_with_label
from .core.ece import calculate_acc_conf

eps = sys.float_info.epsilon


def inference(model, audio, max_step=16000*30 //HOP_LENGTH * HOP_LENGTH):
    with torch.no_grad():
        audio = audio.cuda()
        n_seg = (len(audio) - 1) // max_step + 1
        onsets = []
        offsets = []
        frames = []
        for n in range(n_seg):
            seg_end = min((n+1)*max_step, len(audio))
            if seg_end - n*max_step <= HOP_LENGTH:
                onsets.append(np.zeros((1,88)))
                offsets.append(np.zeros((1,88)))
                frames.append(np.zeros((1,88)))
                break
            onset_pred, offset_pred, _, frame_pred = model(audio[n*max_step : seg_end].reshape(1,-1))
            onsets.append(onset_pred.detach().cpu().numpy().squeeze()), offsets.append(offset_pred.detach().cpu().numpy().squeeze()), frames.append(frame_pred.detach().cpu().numpy().squeeze())
        return np.concatenate(onsets, axis=0), np.concatenate(offsets, axis=0), np.concatenate(frames, axis=0)


def infer_dataset(model_file, dataset, dataset_group, sequence_length, save_path, rep_type, n_class, no_recursive,
                  onset_threshold, frame_threshold, device='cuda', gt_condition=False, offset_bias=1, context_len=1,
                  ac_model_type='simple_conv', lm_model_type='lstm', bidirectional=False):
    if save_path == None:
      save_path = os.path.dirname(model_file)
    os.makedirs(save_path, exist_ok=True)
    recursive = not no_recursive
    dataset_class = getattr(dataset_module, dataset)
    kwargs = {'sequence_length': sequence_length}
    if dataset_group is not None:
        kwargs['groups'] = [dataset_group]
    dataset = dataset_class(**kwargs)

    model_state_path = model_file
    ckp = torch.load(model_state_path, map_location='cpu')
    model_name = ckp['model_name']

    if model_name =='ONF':
        model = models.OnsetsAndFrames(
            ckp['n_mels'], 
            ckp['n_fft'], 
            ckp['f_min'], 
            ckp['f_max'], 
            ckp['cnn_unit'],
            ckp['fc_unit'], 
            ckp['bi_lstm']) 
    model.load_state_dict(ckp['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in tqdm(loader):
        onsets, offsets, frames = inference(model, batch['audio'][0])
        basename = Path(save_path) / Path(batch['path'][0]).stem
        np.savez(str(basename) + f'_pred.npz', onsets, offsets, frames)

        p_est, i_est = decoding.simple_decoding_wrapper(onsets, frames)
        save_midi(str(basename) + f'_pred.midi', p_est, i_est, [64] * len(p_est))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('dataset', nargs='?', default='MAESTRO')
    parser.add_argument('dataset_group', nargs='?', default='test')
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--rep_type', default='base')
    parser.add_argument('--n_class', default=5, type=int)
    parser.add_argument('--ac_model_type', default='simple_conv', type=str)
    parser.add_argument('--lm_model_type', default='lstm', type=str)
    parser.add_argument('--bidirectional', default=False, type=bool)
    parser.add_argument('--context_len', default=1, type=int)
    parser.add_argument('--no_recursive', action='store_true')
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--gt_condition', action='store_true')
    parser.add_argument('--offset_bias', default=1, type=float)

    with torch.no_grad():
        infer_dataset(**vars(parser.parse_args()))
