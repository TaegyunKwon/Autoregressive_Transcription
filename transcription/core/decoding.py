import numpy as np
import torch
from magenta.models.onsets_frames_transcription.infer_util import predict_sequence
from magenta.models.onsets_frames_transcription import configs
from magenta.music import sequences_lib
from mir_eval.util import midi_to_hz


def extract_notes(onsets, frames, velocity=None, onset_threshold=0.5, frame_threshold=0.5, defalut_velocity=64, vel_width=5):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    onsets = (onsets > onset_threshold).type(torch.int).cpu()
    frames = (frames > frame_threshold).type(torch.int).cpu()
    onset_diff = torch.cat(
        [onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    if velocity is None:
        velocity = torch.ones_like(onsets) * defalut_velocity

    pitches = []
    intervals = []
    velocities = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
                velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == vel_width:
                break
            if offset == onsets.shape[0]:
                break
            if (offset != onset) and onsets[offset, pitch].item():
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)

    return np.array(pitches), np.array(intervals), np.array(velocities)


def notes_to_frames(pitches, intervals, shape):
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs


def simple_decoding_wrapper(onset_probs, frame_probs):
    th_onset_probs = torch.from_numpy(onset_probs)
    th_frame_probs = torch.from_numpy(frame_probs) 
    p_ref, i_ref, v_ref = extract_notes(
        th_onset_probs, th_frame_probs)

    scaling = 512 / 16000

    i_ref = (i_ref * scaling).reshape(-1, 2)
    p_ref = np.array([midi_to_hz(21 + midi) for midi in p_ref])
    return p_ref, i_ref


def magenta_decoding(onset_prob, frame_prob, offset_prob, threshold=0.5, viterbi=False):
    config_map=configs.CONFIG_MAP
    config = config_map['onsets_frames']
    hparams = config.hparams
    if viterbi:
        hparams.viterbi_decoding=True
    seq = predict_sequence(frame_prob,
                           onset_prob,
                           frame_prob > threshold,
                           onset_prob > threshold,
                           offset_prob > 0.0,
                           # offset_prob > threshold,
                           velocity_values=None,
                           hparams=hparams,
                           min_pitch=21)
    return seq


def seq_to_mireval_form(seq):
    i_est = []
    p_est = []
    for note in seq.notes:
        i_est.append([note.start_time, note.end_time])
        p_est.append(midi_to_hz(note.pitch))
    i_est = np.asarray(i_est)
    p_est = np.asarray(p_est)

    return p_est, i_est


def seq_to_pianoroll(seq):
    return sequences_lib.sequence_to_pianoroll(
        seq,
        frames_per_second=16000/512,
        min_pitch=21,
        max_pitch=108).active
