import json
import os
from abc import abstractmethod
from glob import glob

import math
import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F

from .constants import *
from .midi import parse_midi, parse_pedal


class PianoSampleDataset(Dataset):
    def __init__(self, path, groups=None, sample_length=16000*5, seed=1000, load_mode='ram', random_sample=True, delay=1):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        print(self.groups, self.available_groups())
        assert all(group in self.available_groups() for group in self.groups)
        self.sample_length = sample_length // HOP_LENGTH * HOP_LENGTH if sample_length is not None else None
        self.random = np.random.RandomState(seed)
        self.data_path = []
        self.load_mode = load_mode
        self.random_sample = random_sample
        self.delay = delay

        self.file_list = dict()
        for group in groups:
          self.file_list[group] = self.files(group)
          for input_pair in self.file_list[group]:
            self.data_path.append(input_pair)

        if self.load_mode == 'ram':
            self.data = []
            print('Loading %d group%s of %s at %s' % (len(groups), 's'[:len(groups) - 1], self.__class__.__name__, path))
            for group in groups:
              for input_files in tqdm(self.file_list[group], desc='Loading group %s' % group):
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        # TODO: velocity label is not shifted
        if self.load_mode == 'ram':
          data = self.data[index]
        else:
          data = self.load(*self.data_path[index])
        result = dict(path=data['path'])

        if self.sample_length is not None:
            audio_length = len(data['audio'])
            if self.random_sample:
              step_begin = self.random.randint(audio_length - self.sample_length) // HOP_LENGTH
            else:
              step_begin = 0
            n_steps = self.sample_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sample_length

            result['audio'] = data['audio'][begin:end]
            if step_begin > self.delay - 1:
              result['shifted_label'] = data['label'][step_begin - self.delay:step_end, :]
            else:
              result['shifted_label'] = F.pad(data['label'][step_begin:step_end, :], (0,0,self.delay,0))
            result['velocity'] = data['velocity'][step_begin:step_end, :]
            result['time'] = begin / SAMPLE_RATE

        else:
            audio = data['audio']
            pad_len = math.ceil(len(audio) / HOP_LENGTH) * HOP_LENGTH - len(audio)
            result['audio'] = F.pad(audio, (0, pad_len))
            result['shifted_label'] = F.pad(data['label'], (0,0,self.delay,0))
            result['velocity'] = data['velocity']
            result['time'] = data['time']

        result['audio'] = result['audio'].float().div_(32768.0)
        result['shifted_label'] = result['shifted_label'].long()
        result['velocity'] = result['velocity'].float().div_(128.0)
        return result

    def __len__(self):
        return len(self.data_path)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        """
        saved_data_path = audio_path.replace('.flac', '_ae_512.pt').replace('.wav', '_ae_512.pt')
        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)

        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE

        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = left + 1
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + 1)

            # off->off :0, on -> off :1, off->onset :2, on -> on :3, on -> onset :4,
            f = int(note) - MIN_MIDI
            if left > 0 and label[left-1, f] <= 1:
              label[left:onset_right, f] = 2
            else:
              label[left:onset_right, f] = 4
            label[onset_right:frame_right, f] = 3
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity, time=0)
        torch.save(data, saved_data_path)
        return data


class MAESTRO(PianoSampleDataset):
    def __init__(self, path=DATA_PATH, meta_json=META_JSON, groups=None, sequence_length=None, seed=42, load_mode='ram', random_sample=True, delay=1):
        self.meta_json = meta_json
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, load_mode=load_mode, random_sample=random_sample, delay=delay)
        
    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test', 'debug']

    def files(self, group):
        metadata = json.load(open(os.path.join(self.path, self.meta_json)))

        if group == 'debug':
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if
                            row['split'] == 'train'])
            files = files[:50]
        else:
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in files]

        result = []

        first_tsv = files[0][1].replace('.midi', '.tsv').replace('.mid', '.tsv')
        if not os.path.exists(first_tsv):
          for audio_path, midi_path in tqdm(files, desc='Converting midi to tsv group %s' % group):
              tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
              midi = parse_midi(midi_path)
              np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
              pedal = parse_pedal(midi_path)
              np.savetxt(tsv_filename.replace('.tsv', '_pedal.tsv'), pedal, fmt='%.6f', delimiter='\t', header='onset,offset')
              result.append((audio_path, tsv_filename))
        else:
          for audio_path, midi_path in files:
              tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
              result.append((audio_path, tsv_filename))
        return result

        
class MAPS(PianoSampleDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42, load_mode='ram', random_sample=True):
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, load_mode=load_mode, random_sample=random_sample)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
        tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]

        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return sorted(zip(flacs, tsvs))
