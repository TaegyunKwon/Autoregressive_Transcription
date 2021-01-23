import torch
import torch.nn.functional as F
from torch import nn
import random
import torchaudio.transforms as transforms

from .constants import *
from .inverse_sigmoid import inverse_sigmoid
from .representation import convert_representation


class ConvStack(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(cnn_unit, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(cnn_unit, cnn_unit * 2, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((cnn_unit * 2) * (n_mels // 4), fc_unit),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class LstmLM(nn.Module):
    def __init__(self, input_features, hidden_units, n_class, bidirectional=False):
        super().__init__()
        self.n_class = n_class
        self.lstm = nn.LSTM(
            input_features, hidden_units, num_layers=2, batch_first=True, bidirectional=bidirectional)
        self.lstm.flatten_parameters()
        if bidirectional:
            self.output = nn.Linear(hidden_units * 2, 88 * n_class)
        else:
            self.output = nn.Linear(hidden_units, 88 * n_class)
            
    def forward(self, x, hidden=None):
        x, hidden_out = self.lstm(x, hidden)  # x: (batch, time, hidden)
        x = self.output(x)  # x: (batch, time, 88 * n_class)
        x = x.view(x.shape[0], x.shape[1], 88, self.n_class)
        return x, hidden_out


class ARModel(nn.Module):
    def __init__(self, n_mels, n_fft, f_min, f_max, cnn_unit, lstm_unit,
        fc_unit, bi_lstm, recursive, ac_model_type, lm_model_type, 
        n_class, win_fw, win_bw):
        super().__init__()
        self.recursive = recursive
        self.n_class = n_class
        self.win_fw = win_fw
        self.win_bw = win_bw
        self.context_len = self.win_fw + self.win_bw + 1
        context_len = self.context_len
        self.lm_model_type = lm_model_type
        self.ac_model_type = ac_model_type

        self.lstm_unit = lstm_unit

        self.melspectrogram = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=n_fft,
            hop_length=HOP_LENGTH, f_min=f_min, f_max=f_max, n_mels=n_mels, normalized=False)
        self.embedding = nn.Embedding(n_class, 2)

        if self.ac_model_type == 'simple_conv':
            self.conv_stack = ConvStack(n_mels, cnn_unit, fc_unit)
        else:
            raise KeyError(f'undefined ac_model_type:{ac_model_type}')
        
        rc_dim = 88*2 if recursive else 0
        if self.lm_model_type == 'lstm':
            self.sequence_model = LstmLM(fc_unit * context_len + rc_dim, lstm_unit, self.n_class)
        else:
            raise KeyError(f'undefined lm_model_type: {self.lm_model_type}')

    def forward(self, audio, prev_label=None, init_hidden=None, init_label=None, sampling_method='GT', gt_ratio=1, offset_bias=1):
        batch_size = audio.shape[0]
        mel = self.melspectrogram(
            audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
        mel = torch.log(torch.clamp(mel, min=1e-9))
        acoustic_out = self.conv_stack(mel)  # (B x T x C)
        if self.win_fw + self.win_bw != 1:
            acoustic_out = acoustic_out.transpose(1, 2)
            acoustic_out = F.pad(acoustic_out, (self.win_bw, self.win_fw)).unsqueeze(3)
            acoustic_out = F.unfold(acoustic_out, (self.win_bw + self.win_fw + 1, 1)) # (B x (C*W) x T)
            acoustic_out = acoustic_out.transpose(1, 2) # (B x T x (C*W))
        if not self.recursive:
            assert prev_label is None
            if self.lm_model_type in ['lstm', 'bi-lstm', 'lstm_small', 'pitchwise_lstm']:
                sequence_out = self.sequence_model(acoustic_out)[0]
            else:
                sequence_out = self.sequence_model(acoustic_out)
            # to shape (batch, class, time, pitch)
            sequence_out = sequence_out.view(batch_size, -1, 88, self.n_class).permute(0, 3, 1, 2)
            return F.log_softmax(sequence_out, dim=1)
        else:
            if sampling_method in ['random', 'gt'] or gt_ratio == 1.0:
                if sampling_method == 'random':
                    prev_label = sampling.random_modification(
                        prev_label, gt_ratio=gt_ratio)
                prev_embedded = self.embedding(
                    prev_label)  # (B x T x pitch x embed_size)
                combined_out = torch.cat([prev_embedded.flatten(-2), acoustic_out], dim=-1)
                if self.lm_model_type in 'lstm':
                    sequence_out = self.sequence_model(combined_out)[0]
                elif self.lm_model_type == 'pitchwise_lstm' or self.lm_model_type =='rnn_exp12':
                    sequence_out = self.sequence_model(acoustic_out, prev_embedded)[0]
                elif self.lm_model_type == 'pitchwise_dnn':
                    sequence_out = self.sequence_model(acoustic_out, prev_embedded)
                elif self.lm_model_type.translate({ord(ch): None for ch in '0123456789'}) in ['rnn_exp']:
                    sequence_out = self.sequence_model(acoustic_out, prev_embedded)[0]
                else:
                    raise KeyError(self.lm_model_type)
                # to shape (batch, class, time, pitch)
                sequence_out = sequence_out.view(batch_size, -1, 88, self.n_class).permute(0, 3, 1, 2)
                return F.log_softmax(sequence_out, dim=1)

            elif sampling_method in ['argmax', 'dist']:
                time_steps = mel.shape[1]
                frame_pred = torch.zeros((batch_size, self.n_class, time_steps, 88)).cuda()
                prev_step = init_label
                for n in range(time_steps):
                    prev_embedded = self.embedding(
                        prev_step)  # (B x 1 x C)
                    combined_pred = torch.cat(
                        [prev_embedded.flatten(-2), acoustic_out[:, n: n+1, :]], dim=-1)
                    if self.lm_model_type in ['lstm']:
                        if n == 0:
                            x, hidden = self.sequence_model(combined_pred)
                        else:
                            x, hidden = self.sequence_model(
                                combined_pred, hidden)  # (B x 1 x model_size)
                    else:
                        raise KeyError(self.lm_model_type)
                    step_pred = x  # (b x 1 x 88 x 5)
                    step_pred = step_pred.view(
                        step_pred.shape[0], 1, 88, self.n_class).permute(0, 3, 1, 2)  # (b, 5, 1, 88)
                    step_pred = F.log_softmax(step_pred, dim=1)  # (b, 5, 1, 88)
                    frame_pred[:, :, n, :] = step_pred.squeeze()
                    #  frame-based sampling
                    if random.random() <= gt_ratio:
                        prev_step = prev_label[:, n: n+1, :]
                    else:
                        step_pred = torch.exp(step_pred.detach())
                        if offset_bias != 1:
                            step_pred[:, 1] *= offset_bias
                        if sampling_method == 'argmax':
                            prev_step = step_pred.argmax(dim=1).detach()
                        elif sampling_method == 'dist':
                            flatten = step_pred.permute(0, 3, 2, 1).flatten(
                                0, 2)
                            prev_step = torch.multinomial(flatten, 1).view(-1, 1, step_pred.shape[-1]).detach()
                            del flatten, step_pred
                if self.lm_model_type in ['lstm', 'lstm_small']:
                    self.sequence_model.lstm.flatten_parameters()
                return frame_pred
        
    def acoustic_model(self, audio):
        batch_size = audio.shape[0]
        mel = self.melspectrogram(
            audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        acoustic_out = self.conv_stack(mel)  # (B x T x C)
        if self.context_len != 1:
            # acoustic_out = F.pad(acoustic_out, (0, 0, 0, self.context_len - 1))
            acoustic_out = acoustic_out.transpose(1, 2)
            acoustic_out = F.pad(acoustic_out, (0, self.context_len - 1)).unsqueeze(3)
            acoustic_out = F.unfold(acoustic_out, (self.context_len, 1))
            acoustic_out = acoustic_out.transpose(1, 2)
        return acoustic_out


    def lm_model_step(self, acoustic_out, hidden, prev_out):
        '''
        acoustic_out: tensor, shape of (B x T(1) x C)
        prev_out: tensor, shape of (B x T(1) x pitch)
        '''
        batch_size = acoustic_out.shape[0]
        prev_embedded = self.embedding(
            prev_out)  # (B x T x pitch x embed_size)
        if not self.recursive:
            sequence_out, hidden_out = self.sequence_model(acoustic_out, hidden)
            step_pred=sequence_out
        else:
            combined_out = torch.cat([prev_embedded.flatten(-2), acoustic_out], dim=-1)
            if self.lm_model_type == 'lstm':
                x, hidden = self.sequence_model(
                    combined_out, hidden)  # (B x 1 x model_size)
            else:
                x = self.sequence_model(combined_out)
            step_pred = x  # (b x 1 x 88*5)
        step_pred = step_pred.view(
            step_pred.shape[0], -1, 88, self.n_class).permute(0, 3, 1, 2)  # (b, 5, 1, 88)
        return F.log_softmax(step_pred, dim=1), hidden

    def init_hidden(self):
        if self.lm_model_type == 'lstm':
            return (torch.zeros(2, 1, self.lstm_unit).cuda(), torch.zeros(2, 1, self.lstm_unit).cuda())

class LSTM(nn.Module):
    def __init__(self, input_features, recurrent_features, bidirectional=True):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        return self.rnn(x)[0]

class OnsetsAndFrames(nn.Module):
    def __init__(self, n_mels, n_fft, f_min, f_max, cnn_unit=48, fc_unit=768, bidirectional=True):
        super().__init__()

        unit_multiplier = 2 if bidirectional else 1

        self.melspectrogram = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=n_fft,
            hop_length=HOP_LENGTH, f_min=f_min, f_max=f_max, n_mels=n_mels, normalized=False)
        
        self.onset_stack = nn.Sequential(
            ConvStack(n_mels, cnn_unit, fc_unit),
            LSTM(fc_unit, 256, bidirectional),
            nn.Linear(256*unit_multiplier, 88),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(n_mels, cnn_unit, fc_unit),
            LSTM(fc_unit, 256, bidirectional),
            nn.Linear(256*unit_multiplier, 88),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(n_mels, cnn_unit, fc_unit),
            nn.Linear(fc_unit, 88),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            LSTM(88 * 3, 256, bidirectional),
            nn.Linear(256*unit_multiplier, 88),
            nn.Sigmoid()
        )

    def forward(self, audio):
        mel = self.melspectrogram(
            audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
        mel = torch.log(torch.clamp(mel, min=1e-9))
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        return onset_pred, offset_pred, activation_pred, frame_pred

def run_on_batch_onf(model, batch, device):
    audio = batch['audio'].to(device)
    label = batch['shifted_label'][:,1:].to(device)

    onset_label = ((label == 2) + (label == 4)).to(torch.float)
    offset_label = (label == 1).to(torch.float)
    frame_label = (label >= 1).to(torch.float)

    onset_pred, offset_pred, _, frame_pred = model(audio)

    predictions = {
        'onset': onset_pred.reshape(*onset_label.shape),
        'offset': offset_pred.reshape(*offset_label.shape),
        'frame': frame_pred.reshape(*frame_label.shape),
    }

    losses = {
        'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
        'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
        'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
    }
    
    loss = sum(losses.values())
    return predictions, loss

def run_on_batch(model, batch, device, sampling_method, criterion, rep_type='base', recursive=True, gt_ratio=1, delay=1):
    audio_label = batch['audio'].to(device)
    shifted_label = batch['shifted_label'].to(device)
    shifted_label = convert_representation(shifted_label, rep_type)

    if recursive:
        if gt_ratio != 1.0 and sampling_method in ['argmax', 'dist']:
            init_step = shifted_label[:, 0:1, :].to(device)
            frame_pred = model(audio_label, prev_label=shifted_label[:, :-delay], init_label=init_step,
                            sampling_method=sampling_method, gt_ratio=gt_ratio)
        else:
            frame_pred = model(
                audio=audio_label, prev_label=shifted_label[:, :-delay], sampling_method=sampling_method, gt_ratio=gt_ratio)
    else:
        frame_pred = model(audio=audio_label)

    label = shifted_label[:, delay:]
    losses = criterion(frame_pred, label)

    return frame_pred, losses


def inference(acoustic_model, language_model, batch):
    with torch.no_grad():
        assert(batch.shape[0] == 1)  # take one sequence at a time
        device = acoustic_model.device
        init_step = torch.zeros((1, 88)).to(device)


def velocity_loss(velocity_pred, velocity_label, onset_label):
    denominator = onset_label.sum()
    if denominator.item() == 0:
        return denominator
    else:
        return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator
