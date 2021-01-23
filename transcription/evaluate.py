import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
import tensorflow as tf
import mir_eval
import pretty_midi
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from tqdm import tqdm
from torch.utils.data import DataLoader
from magenta.music import sequences_lib
import magenta.models.onsets_frames_transcription.metrics as magenta_metrics
from magenta.music import midi_io

import transcription.core.dataset as dataset_module
from transcription.core import *
from transcription.core.ece import calculate_acc_conf
from transcription.core import models, representation
from transcription.core.utils import LabelSmoothingLoss, draw_predictions_with_label, NLLLoss
eps = sys.float_info.epsilon


def evaluate(batch, model, device, save_path=None, criterion=None, sampling_method='argmax', rep_type='base', plot_example=False, recursive=True, detail_eval=False, delay=1):
    # TODO: input: prediction & label. output: metric
    metrics = defaultdict(list)
    acc_conf = []
    if sampling_method == 'argmax':
        gt_ratio = 0.0
    elif sampling_method == 'gt':
        gt_ratio = 1.0
    else:
        gt_ratio = 0.0
    with th.no_grad():
        preds, losses = models.run_on_batch(
            model, batch, device[0], sampling_method=sampling_method, gt_ratio=gt_ratio, criterion=criterion, rep_type=rep_type, recursive=recursive, delay=delay)
    losses = losses.cpu().numpy()
    metrics['loss'].extend(list(np.atleast_1d(losses)))

    for n in range(preds.shape[0]):
        label = dict()
        pred = preds[n]
        argmax_pred = pred.argmax(dim=0)
        for key in batch:
            label[key] = batch[key][n]

        if detail_eval:
            acc_conf.append(calculate_acc_conf(pred.cpu().numpy().transpose((1, 2, 0)),
                                        label['shifted_label'][delay:].cpu().numpy()))
        else:
            acc_conf.append(None)

        onset_ref, offset_ref, frame_ref = representation.base2onsets_and_frames(label['shifted_label'][delay:])
        onsets, offsets, frames = representation.convert2onsets_and_frames(argmax_pred, rep_type)

        
        p_ref, i_ref, v_ref = extract_notes(onset_ref, frame_ref)
        p_est, i_est, v_est = extract_notes(onsets, frames)

        t_ref, f_ref = notes_to_frames(p_ref, i_ref, frame_ref.shape)
        t_est, f_est = notes_to_frames(p_est, i_est, frames.shape)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [np.array([midi_to_hz(MIN_MIDI + midi)
                            for midi in freqs]) for freqs in f_ref]
        t_est = t_est.astype(np.float64) * scaling
        f_est = [np.array([midi_to_hz(MIN_MIDI + midi)
                            for midi in freqs]) for freqs in f_est]

        p, r, f, o = evaluate_notes(
            i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        metrics['metric/frame/f1'].append(hmean(
            [frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

        for key, value in frame_metrics.items():
            metrics['metric/frame/' + key.lower().replace(' ', '_')].append(value)
        
        
        if plot_example:
            pred = pred.cpu().numpy().transpose(1, 2, 0)
            label = label['shifted_label'][delay:].cpu().numpy()
            os.makedirs(save_path, exist_ok=True)
            basename = Path(save_path) / Path(batch['path'][n]).stem

            np.save(str(basename) + f'_label.npy', label)
            np.save(str(basename) + f'_pred_{sampling_method}.npy', pred)

            draw_predictions_with_label(str(basename) + f'_pred.png',
                                        pred,
                                        label)
            # midi_path = str(basename) + f'_pred_{global_step}.mid'
            # save_midi(midi_path, p_est, i_est, v_est)

    return metrics, acc_conf


def evaluate_onf(batch, model, device, save_path=None, criterion=None, sampling_method='argmax', rep_type='base', plot_example=False, recursive=True, detail_eval=False, delay=1):
    metrics = defaultdict(list)
    with th.no_grad():
        preds, losses = models.run_on_batch_onf(model, batch, device[0])
    losses = losses.cpu().numpy()
    metrics['loss'].extend([losses])

    for n in range(preds['frame'].shape[0]):
        label = dict()
        for key in batch:
            label[key] = batch[key][n]

        onset_ref, offset_ref, frame_ref = representation.base2onsets_and_frames(label['shifted_label'][delay:])
        onsets = preds['onset'][n] > 0.5
        offsets = preds['offset'][n] > 0.5
        frames = preds['frame'][n] > 0.5

        p_ref, i_ref, v_ref = extract_notes(onset_ref, frame_ref)
        p_est, i_est, v_est = extract_notes(onsets, frames)

        t_ref, f_ref = notes_to_frames(p_ref, i_ref, frame_ref.shape)
        t_est, f_est = notes_to_frames(p_est, i_est, frames.shape)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [np.array([midi_to_hz(MIN_MIDI + midi)
                            for midi in freqs]) for freqs in f_ref]
        t_est = t_est.astype(np.float64) * scaling
        f_est = [np.array([midi_to_hz(MIN_MIDI + midi)
                            for midi in freqs]) for freqs in f_est]

        p, r, f, o = evaluate_notes(
            i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        metrics['metric/frame/f1'].append(hmean(
            [frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

        for key, value in frame_metrics.items():
            metrics['metric/frame/' + key.lower().replace(' ', '_')].append(value)
        
        
    return metrics, None


def framewise_eval(argmax_pred, label):
    '''
    evaluate frame-wise (point-wise) evaluation
    argmax_pred: torch.tensor shape of (frame, pitch)
    label: torch.tensor shape of (frame, pitch)
    '''
    frame_metrics = defaultdict(list)

    n_class = label.max() - label.min() + 1
    for n in range(int(n_class)):
        tp = th.sum((label == n) * (argmax_pred == n))
        fn = th.sum((label == n) * (argmax_pred != n))
        fp = th.sum((label != n) * (argmax_pred == n))
        
        pr = tp / float(tp + fp)
        re = tp / float(tp + fn)
        f1 = 2 * pr * re / float(pr + re)
       
        frame_metrics[f'class_{n}/precision'] = pr
        frame_metrics[f'class_{n}/recall'] = re
        frame_metrics[f'class_{n}/f1'] = f1
    
    frame_metrics['accuracy'] = th.sum(argmax_pred == label) / float(label.numel())
    return frame_metrics


def sequence_to_valued_intervals(note_sequence,
                                 min_midi_pitch=21,
                                 max_midi_pitch=108,
                                 restrict_to_pitch=None):
  """Convert a NoteSequence to valued intervals."""
  intervals = []
  pitches = []
  velocities = []

  for note in note_sequence.notes:
    if restrict_to_pitch and restrict_to_pitch != note.pitch:
      continue
    if note.pitch < min_midi_pitch or note.pitch > max_midi_pitch:
      continue
    # mir_eval does not allow notes that start and end at the same time.
    if note.end_time == note.start_time:
      continue
    intervals.append((note.start_time, note.end_time))
    pitches.append(note.pitch)
    velocities.append(note.velocity)

  # Reshape intervals to ensure that the second dim is 2, even if the list is
  # of size 0. mir_eval functions will complain if intervals is not shaped
  # appropriately.
  return (np.array(intervals).reshape((-1, 2)), np.array(pitches),
          np.array(velocities))
 

def magenta_note_eval(pred_seq, label_seq, onset_tolerance=0.05, restrict_to_pitch=None):
    note_density = len(pred_seq.notes) / pred_seq.total_time

    est_intervals, est_pitches, est_velocities = (
        sequence_to_valued_intervals(
            pred_seq, restrict_to_pitch=restrict_to_pitch))

    ref_intervals, ref_pitches, ref_velocities = (
        sequence_to_valued_intervals(
            label_seq, restrict_to_pitch=restrict_to_pitch))
        
    note_precision, note_recall, note_f1, _ = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            pretty_midi.note_number_to_hz(ref_pitches),
            est_intervals,
            pretty_midi.note_number_to_hz(est_pitches),
            onset_tolerance=onset_tolerance,
            offset_ratio=None))
    '''
    (note_with_velocity_precision, note_with_velocity_recall,
    note_with_velocity_f1, _) = (
        mir_eval.transcription_velocity.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=pretty_midi.note_number_to_hz(ref_pitches),
            ref_velocities=ref_velocities,
            est_intervals=est_intervals,
            est_pitches=pretty_midi.note_number_to_hz(est_pitches),
            est_velocities=est_velocities,
            offset_ratio=None))
    '''
    (note_with_offsets_precision, note_with_offsets_recall, note_with_offsets_f1,
    _) = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, pretty_midi.note_number_to_hz(ref_pitches),
            est_intervals, pretty_midi.note_number_to_hz(est_pitches),
            onset_tolerance=onset_tolerance)
    )
    '''
    (note_with_offsets_velocity_precision, note_with_offsets_velocity_recall,
    note_with_offsets_velocity_f1, _) = (
        mir_eval.transcription_velocity.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=pretty_midi.note_number_to_hz(ref_pitches),
            ref_velocities=ref_velocities,
            est_intervals=est_intervals,
            est_pitches=pretty_midi.note_number_to_hz(est_pitches),
            est_velocities=est_velocities))
    '''
    return (note_precision, note_recall, note_f1, note_with_offsets_precision,
            note_with_offsets_recall, note_with_offsets_f1)


def magenta_frame_eval(pred_seq, frame_labels):
    processed_frame_predictions = sequences_lib.sequence_to_pianoroll(
        pred_seq,
        frames_per_second=16000/512,
        min_pitch=21,
        max_pitch=108).active

    if processed_frame_predictions.shape[0] < frame_labels.shape[0]:
        # Pad transcribed frames with silence.
        pad_length = frame_labels.shape[0] - processed_frame_predictions.shape[0]
        processed_frame_predictions = np.pad(processed_frame_predictions,
                                            [(0, pad_length), (0, 0)], 'constant')
    elif processed_frame_predictions.shape[0] > frame_labels.shape[0]:
        # Truncate transcribed frames.
        processed_frame_predictions = (
            processed_frame_predictions[:frame_labels.shape[0], :])

    frame_metrics = magenta_metrics.calculate_frame_metrics(
        frame_labels=frame_labels,
        frame_predictions=processed_frame_predictions)

    results = defaultdict(list)
    for key, value in frame_metrics.items():
        results[key] = value[0].numpy()
    return results


def midi_to_seq(midi):
    seq = midi_io.midi_file_to_note_sequence(midi)
    seq = sequences_lib.apply_sustain_control_changes(seq)
    return seq


def adjust_length(pred, label):
    pred_len = pred.shape[0]
    label_len = label.shape[0]
    
    if pred_len < label_len:
        return F.pad(pred, (0, 0, 0, label_len-pred_len))
    elif pred_len > label_len:
        return pred[:label_len]
    else:
        return pred

