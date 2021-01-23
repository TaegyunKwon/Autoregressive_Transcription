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


def inference(model, audio, n_class=5, max_step=10000, stateful=True):
    with torch.no_grad():
        if stateful:
            offset = 0
            step_len = (len(audio) - 1) // HOP_LENGTH + 1
            results = torch.zeros(step_len, 88, n_class)

            init_hidden = model.init_hidden()
            init_step = torch.zeros(1, 1, 88).long().cuda()
            hidden = init_hidden
            prev_step = init_step

            audio_len = len(audio)
            n_segs = ((step_len - 1)//max_step + 1)
            if 0 <= audio_len - (n_segs + 1)*max_step* HOP_LENGTH < 8192:
                n_segs -= 1
            seg_edges = [el*max_step for el in range(n_segs)]
            for n in range(step_len):
                if n in seg_edges:
                    offset = n
                    if n == seg_edges[-1]:
                        acoustic_out = model.acoustic_model(audio[offset * HOP_LENGTH : ].cuda())
                    else: 
                        acoustic_out = model.acoustic_model(audio[offset * HOP_LENGTH : (offset + max_step + 10) * HOP_LENGTH].cuda())

                step, hidden_out = model.lm_model_step(acoustic_out[:, n - offset:n - offset+1, :], hidden, prev_step)
                results[n] = step[0].permute(1, 2, 0).cpu().detach()
                del hidden, prev_step
                prev_step = step.argmax(dim=1)
                hidden = hidden_out
        else:
            step_len = (len(audio) - 1) // HOP_LENGTH + 1
            results = torch.zeros(step_len, 88, n_class)
            max_step = 1000
            n_seg = (step_len - 1) // max_step
            for n in range(n_seg):
                max_seg = min((n+1)*max_step , len(step_len))
                frame_pred = model(
                    audio=audio[n*max_step*HOP_LENGTH: max_seg*HOP_LENGTH].cuda(), prev_label=None, sampling_method='argmax', gt_ratio=1.0)
                results[n*max_step: max_seg] = frame_pred.detach().cpu()

    return results


def infer_dataset(model_file, dataset, dataset_group, sequence_length, save_path, rep_type, 
                  onset_threshold, frame_threshold, device='cuda', gt_condition=False, offset_bias=1, 
                  seg_len=16000*30 //HOP_LENGTH * HOP_LENGTH, stateful=True):
    if save_path == None:
      save_path = os.path.dirname(model_file)
    os.makedirs(save_path, exist_ok=True)
    dataset_class = getattr(dataset_module, dataset)
    kwargs = {'sequence_length': sequence_length}
    if dataset_group is not None:
        kwargs['groups'] = [dataset_group]
    dataset = dataset_class(**kwargs)

    model_state_path = model_file
    ckp = torch.load(model_state_path, map_location='cpu')
    model_name = ckp['model_name']

    model_class = getattr(models, model_name)

    if model_name == 'ARModel':
        model = models.ARModel(
            ckp['n_mels'], 
            ckp['n_fft'], 
            ckp['f_min'], 
            ckp['f_max'], 
            ckp['cnn_unit'],
            ckp['lstm_unit'],
            ckp['fc_unit'], 
            ckp['bi_lstm'], 
            recursive=ckp['recursive'], 
            n_class=ckp['n_class'], 
            win_fw=ckp['win_fw'],
            win_bw=ckp['win_bw'],
            ac_model_type=ckp['ac_model_type'], 
            lm_model_type=ckp['lm_model_type'])

    model.load_state_dict(ckp['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in tqdm(loader):
        result = inference(model, batch['audio'][0], ckp['n_class'], seg_len, stateful)
        preds = result.cpu().numpy()

        argmax_pred = np.argmax(preds, -1)
        onsets, offsets, frames = representation.convert2onsets_and_frames(torch.from_numpy(argmax_pred), rep_type)
        onsets = onsets.numpy()
        offsets = offsets.numpy()
        frames = frames.numpy()

        basename = Path(save_path) / Path(batch['path'][0]).stem
        np.save(str(basename) + f'_pred.npy', preds)

        p_est, i_est = decoding.simple_decoding_wrapper(onsets, frames)
        save_midi(str(basename) + f'_pred.midi', p_est, i_est, [64] * len(p_est))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('dataset', nargs='?', default='MAESTRO')
    parser.add_argument('dataset_group', nargs='?', default='test')
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--rep_type', default='base')
    parser.add_argument('--seg_len', default=16000*30//HOP_LENGTH*HOP_LENGTH, type=int)
    parser.add_argument('--stateful', default=True, type=bool)
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--gt_condition', action='store_true')
    parser.add_argument('--offset_bias', default=1, type=float)

    with torch.no_grad():
        infer_dataset(**vars(parser.parse_args()))
