import os
from string import digits
import argparse
from pathlib import Path
import numpy as np
import torch as th
import tensorflow as tf
from collections import defaultdict
from transcription.evaluate import framewise_eval
from tqdm import tqdm
import json
from transcription.core import decoding, representation
from transcription import evaluate
from transcription.core.constants import DATA_PATH, META_JSON
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
import magenta.models.onsets_frames_transcription.metrics as magenta_metrics


def cal_metric(target_folder, dataset, rep_type='base', sampling_method='argmax', delay=1, no_onehot=False, save_path=None):
    if dataset == 'MAESTRO':
        metadata = json.load(open(os.path.join(DATA_PATH, META_JSON)))
        maestro = DATA_PATH
    files = sorted([row['audio_filename'] for row in metadata if row['split'] == 'test'])

    if save_path == None:
        save_path = target_folder
    Path(save_path).mkdir(exist_ok=True)

    pred_lists = []
    label_lists = list([str(Path(maestro) / el.replace('.wav', '_ae_512.pt')) for el in files])
    for filename in files:
        basename = filename.split('/')[1].replace('.wav', '')
        if rep_type == 'onf':
            pred_npy = list(Path(target_folder).glob(basename + '*_pred*.npz'))
        else:
            pred_npy = list(Path(target_folder).glob(basename + '*_pred*.npy'))
        pred_lists.append(pred_npy[0])

    midi_lists = list([str(Path(maestro) / el.replace('.wav', '.midi')) for el in files])

    metric = defaultdict(list)
    for n in tqdm(range(len(pred_lists))):
        pred = np.load(pred_lists[n])
        if rep_type == 'onf':

            onsets = pred['arr_0'] > 0.5
            offsets = pred['arr_1'] > 0.5
            frames = pred['arr_2'] > 0.5
        else:
            if no_onehot:
                argmax_pred = pred
            else:
                argmax_pred = np.argmax(pred, -1)

            if delay != 1:
                argmax_pred = np.pad(argmax_pred[delay-1:], ((0, delay-1), (0,0)))

            onsets, offsets, frames = representation.convert2onsets_and_frames(th.from_numpy(argmax_pred), rep_type)
            onsets = onsets.numpy()
            offsets = offsets.numpy()
            frames = frames.numpy()

        label = th.load(label_lists[n])['label']

        # th_pred = th.from_numpy(argmax_pred)
        # th_label = representation.convert_representation(label, rep_type)

        # point_result = framewise_eval(th_pred, th_label)
        seq_pred = decoding.magenta_decoding(onsets, frames, offsets)
        seq_label = evaluate.midi_to_seq(midi_lists[n])
        frame_label = (label.numpy() >= 1 ).astype(int)

        frame_results_raw_tf = magenta_metrics.calculate_frame_metrics(
            frame_labels=frame_label,
            frame_predictions=frames)
        frame_results_raw = defaultdict(list)
        for key, value in frame_results_raw_tf.items():
            frame_results_raw[key] = value[0].numpy()
        
        frame_results = evaluate.magenta_frame_eval(seq_pred, frame_label)
        
        magenta_result = evaluate.magenta_note_eval(seq_pred, seq_label)
        for n in range(21, 108):
            metric[f'magenta_note_{n}'].append(evaluate.magenta_note_eval(seq_pred, seq_label, restrict_to_pitch=n))
   

        # metric['point'].append(point_result)
        metric['frame'].append(frame_results)
        metric['frame_raw'].append(frame_results_raw)
        metric['magenta_note'].append(magenta_result)

    
    metric_txt = open(Path(save_path) / f'metric_results_{sampling_method}.txt', 'w')
    pitchwise_evals = np.asarray([metric[f'magenta_note_{el}'] for el in range(21, 108)])
    np.save(Path(save_path) / f'magenta_pitchwise_notes_{sampling_method}.npy', pitchwise_evals)
    for key, value in metric.items():
        remove_digits = str.maketrans('', '', digits)
        if key.translate(remove_digits) == 'magenta_note_':
            pass
        else:
            np.save(Path(save_path) / f'{key}_{sampling_method}.npy', value)
        if key not in ['frame', 'frame_raw', 'point']:
            print(key)
            result_str = f'{key}: {[np.mean(el) for el in zip(*value)]}'
            print(result_str)
            metric_txt.write(result_str + '\n')
        else:
            print(key)
            key2 = value[0].keys()
            for name in key2:
                result_str = f'{name}: {np.mean([el[name] for el in value])}'
                print(result_str)
                metric_txt.write(result_str + '\n')
    metric_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target_folder', type=Path)
    parser.add_argument('dataset', nargs='?', default='MAESTRO')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--no_onehot', action='store_true')
    parser.add_argument('--rep_type', default='base', type=str)
    parser.add_argument('--sampling_method', default='argmax', type=str)
    parser.add_argument('--delay', default=1, type=int)
    args = parser.parse_args()

    cal_metric(**vars(args))
