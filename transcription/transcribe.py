from collections import defaultdict
from pathlib import Path
import argparse
import tempfile
import shutil
import subprocess
import math

import torch as th
import torch.nn.functional as F
import numpy as np
import soundfile
import librosa
import os

from .inference import inference
from .core import models, representation, decoding
from .core.midi import save_midi

def load_audio(audiofile):
    try:
        audio, sr = soundfile.read(audiofile)
        if audio.shape[1] != 1:
            audio = librosa.to_mono(audio.T)
        if sr != 16000:
            audio = librosa.resample(audio, sr, 16000)
    except:
        path_audio = Path(audiofile)
        filetype = path_audio.suffix
        assert filetype in ['.mp3', '.ogg', '.flac', '.wav', '.m4a', '.mp4'], filetype
        with tempfile.TemporaryDirectory() as tempdir:
            tempwav = Path(tempdir) / (path_audio.stem + '_temp' + '.flac')
            command = ['ffmpeg', '-i', audiofile, '-af', 'aformat=s16:16000', '-ac', '1', tempwav] 
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            audio, sr = soundfile.read(tempwav)
    return audio

def transcribe(audio, model, args, save_name=None, save=True):
    """
    TODO
    """
    t_audio = th.tensor(audio).to(th.float)
    pad_len = math.ceil(len(t_audio) / 512) * 512 - len(t_audio)
    t_audio = F.pad(t_audio, (0, pad_len))
    result = inference(model, t_audio, args.n_class)
    preds = result.cpu().numpy()

    argmax_pred = np.argmax(preds, -1)
    onsets, offsets, frames = representation.convert2onsets_and_frames(th.from_numpy(argmax_pred), args.rep_type)
    onsets = onsets.numpy()
    offsets = offsets.numpy()
    frames = frames.numpy()


    if save:
        np.save(save_name, preds)

    p_est, i_est = decoding.simple_decoding_wrapper(onsets, frames)
    save_midi(str(basename) + f'.midi', p_est, i_est, [64] * len(p_est))

    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('audio_file', type=str)
    parser.add_argument('--rep_type', default='base')
    parser.add_argument('--save_path', default=None)
    args = parser.parse_args()
    with th.no_grad():
        model_state_path = args.model_file
        ckp = th.load(model_state_path, map_location='cpu')
        model_name = ckp['model_name']
        model_class = getattr(models, model_name)
        n_class = ckp['n_class'] # TODO: fix
        args.n_class = n_class
        if model_name == 'FlexibleModel':
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
                context_len=ckp['context_len'],
                ac_model_type=ckp['ac_model_type'], 
                lm_model_type=ckp['lm_model_type'])

        model.load_state_dict(ckp['model_state_dict'])
        model.eval()
        model = model.cuda()

        audio = load_audio(args.audiofile)
        basename = Path(args.audiofile).stem
        save_name = Path(args.audiofile).parent / (str(basename) + f'_pred.npy')

        transcribe(audio, model, args, save_name)
