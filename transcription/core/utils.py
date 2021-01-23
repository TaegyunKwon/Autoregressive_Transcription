import sys
from functools import reduce
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.modules.module import _addindent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from . import inverse_sigmoid
from . import focal_loss
import torch.nn.functional as F


def cycle(iterable):
    while True:
        for item in iterable:
            yield item

def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    '''
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        # file.flush()
    '''
    print(string)
    return count

def draw_onsets(save_path, onsets, frames, onset_threshold=0.5, frame_threshold=0.5, zoom=2):
    '''
    draw onsets and frames
    onsets are notated as 2, and frames are notated as 1.
    '''
    if Path(save_path).suffix != '.png':
        save_path += '.png'
    onsets = ((onsets.t() > onset_threshold).type(torch.int)).cpu()
    frames = ((frames.t() > frame_threshold).type(torch.int)).cpu()
    both = ((onsets + frames) >= 1).type(torch.int)
    height = 5*zoom
    width_scale = 4
    width = int((onsets.shape[1] / onsets.shape[0]) * height//width_scale)
    plt.figure(figsize=(width, height))
    plt.imshow(both + onsets, aspect='auto', origin='bottom', interpolation='nearest', cmap='gray', vmin=0, vmax=1)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def draw_predictions2(save_path, prob, zoom=2):
    '''
    draw piano-roll like figure with multi-class probability 'prob'
    prob: numpy array of shape (frames, pitch, class), with 0 <= value <= 1
    prob class index :
        0: off
        1: onset
        2: frame
        3: frame->onset
        4: offset
    draw class index:
        0: onset
        1: frame->onset
        2: frame
        3: offset
        4: off
    '''
    if Path(save_path).suffix != '.png':
        save_path += '.png'
    prob = prob[:, :, [1, 3, 2, 4, 0]]
    for n in range(1, 5):
        prob[:, :, n] += n
    height = 5*zoom
    width = int((prob.shape[0] / (prob.shape[1]*prob.shape[2])) * height)

    cmap = np.zeros((64*5, 4))
    colors = ['Reds', 'Reds', 'Greens', 'Purples', 'gray']
    for n, color in enumerate(colors):
        tmp_cmap = cm.get_cmap(color, 64)
        print(tmp_cmap(range(64)).shape)
        cmap[n*64:(n+1)*64, :] = tmp_cmap(range(64))
    
    cmap = ListedColormap(cmap)
    plt.figure(figsize=(width, height))
    plt.imshow(prob.T, aspect='auto', origin='bottom',
               interpolation='nearest', cmap=cmap, vmin=0, vmax=5)
    for i in range(1, prob.shape[1]//5):
        plt.axhline(i*5)
    plt.colorbar()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def draw_predictions(save_path, prob, zoom=2):
    '''
    draw piano-roll like figure with multi-class probability 'prob'
    prob: numpy array of shape (frames, pitch, class), with 0 <= value <= 1
    prob class index :
        0: off    (black)
        1: onset  (red)
        2: frame  (yellow)
        3: reonset(red)
        4: offset (blue)
    '''
    if Path(save_path).suffix != '.png':
        save_path += '.png'

    prob = prob[:, :, [2, 4, 3, 1, 0]]
    max_class = prob.argmax(axis=-1)
    for n in range(1, 5):
        prob[:, :, n] += n
    height = 5*zoom
    width_scale = 4
    width = int((prob.shape[0] / (prob.shape[1]) * height / width_scale))

    new_prob = np.zeros((prob.shape[0], prob.shape[1]))
    for i in range(new_prob.shape[0]):
        for j in range(new_prob.shape[1]):
            new_prob[i, j] = prob[i, j, max_class[i, j]]
    cmap = np.zeros((64*5, 4))
    # colors = ['Reds', 'Reds', 'Greens', 'Purples', 'gray']
    colors = ['Reds', 'Reds', 'Greens', 'Purples', 'Greys']
    for n, color in enumerate(colors):
        tmp_cmap = cm.get_cmap(color, 64)
        cmap[n*64:(n+1)*64, :] = tmp_cmap(range(63, -1, -1))
    cmap = ListedColormap(cmap)

    n_fig = (width - 1) // (2**16) + 1
    for n in range(n_fig):
        prob_seg = new_prob[n//new_prob.shape[0]: n+1//new_prob.shape[0]]

        plt.figure(figsize=(width//n_fig, height))
        plt.imshow(prob_seg.T, aspect='auto', origin='bottom',
                interpolation='nearest', cmap=cmap, vmin=0, vmax=5)
        plt.colorbar()
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', f'_{n}.png'), bbox_inches='tight')
        plt.close()

def draw_predictions_with_label(save_path, prob, label, zoom=2):
    '''
    draw piano-roll predictions probability with label
    prob: numpy array shape of (frame, pitch, class)
    label: numpy array shape of (frame, pitch)
    '''
    if Path(save_path).suffix != '.png':
        save_path += '.png'
    n_class = prob.shape[-1]
    max_class = prob.argmax(axis=-1)
    for n in range(1, n_class):
        prob[:, :, n] += n
    label = label +0.9
    height = n_class*zoom
    width_scale = 4
    width = int((prob.shape[0] / (prob.shape[1]) * height) / width_scale)
    new_prob = np.zeros((prob.shape[0], prob.shape[1]))
    '''
    for i in range(new_prob.shape[0]):
        for j in range(new_prob.shape[1]):
            new_prob[i, j] = prob[i, j, max_class[i, j]]
    '''
    new_prob = max_class + 0.5

    concat = np.concatenate((new_prob, label), axis=1)
    cmap = np.zeros((64*n_class, 4))
    colors = ['Greys', 'Purples', 'Reds', 'Greens', 'Reds']
    colors = colors[:n_class]
    for n, color in enumerate(colors):
        tmp_cmap = cm.get_cmap(color, 64)
        # cmap[n*64:(n+1)*64, :] = tmp_cmap(range(63, -1, -1))
        cmap[n*64:(n+1)*64, :] = tmp_cmap(range(64))
    
    cmap = ListedColormap(cmap)
    n_fig = (width * 100 - 1) // (2**16) + 1  # 100: dpi

    for n in range(n_fig):
        prob_seg = concat[n * concat.shape[0] // n_fig: (n + 1) * concat.shape[0] // n_fig]

        plt.figure(figsize=(width//n_fig, height))
        plt.imshow(prob_seg.T, aspect='auto', origin='bottom',
                interpolation='nearest', cmap=cmap, vmin=0, vmax=n_class)
        plt.colorbar()
        plt.savefig(save_path.replace('.png', f'_{n}.png'), bbox_inches='tight')
        plt.close()
               
def deprecated_save_pianoroll(path, onsets, frames, onset_threshold=0.5, frame_threshold=0.5, zoom=4):
    """
    Saves a piano roll diagram

    Parameters
    ----------
    path: str
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    zoom: int

    not working because PIL is broken
    """
    onsets = (1 - (onsets.t() > onset_threshold).type(torch.int)).cpu()
    frames = (1 - (frames.t() > frame_threshold).type(torch.int)).cpu()
    both = (1 - (1 - onsets) * (1 - frames))
    image = torch.stack([onsets, frames, both], dim=2).flip(0).mul(255).numpy()
    image = Image.fromarray(image, 'RGB')
    image = image.resize((image.size[0], image.size[1] * zoom))
    image.save(path)

def teacher_ratio(step, max_step, e_start, e_end, curve):
    '''
    calculate teacher forcing ratio at step.
    ratio starts from e_start when step==0, and decrease to e_end when step==max_step
    if curve=='i_sigmoid', use inversed sigmoid function in [-3, 3] as template
    '''
    if step >= max_step:
        return e_end
    else:
        if curve == 'i_sigmoid':
            return inverse_sigmoid.inverse_sigmoid(step, max_step, e_start, e_end)
        elif curve == 'linear':
            return (step / max_step) * (e_start - e_end) + e_end
        elif curve == 'preheat':
            if step < max_step // 2:
                return 1.0
            else:  
                return (step / max_step) * (e_start - e_end) + e_end
        else:
            raise KeyError(curve)

class LabelSmoothingLoss(torch.nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186
    """
    def __init__(self, label_smoothing, class_size):
        assert 0.0 <= label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (class_size - 1)
        one_hot = torch.full((class_size,), smoothing_value)
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): [Batch x n_class x *]
        target (LongTensor): [Batch x *]
        """
        model_prob = self.one_hot.repeat(target.size() + (1,))
        model_prob = model_prob.to(device=target.device)
        model_prob.scatter_(-1, target.unsqueeze(-1), self.confidence)

        return F.kl_div(output.permute(0, 2, 3, 1), model_prob, reduction='none')       

class NLLLoss(torch.nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, prob, target):
        """
        prob (FloatTensor): [batch x n_class, time x pitch]
        target (LongTensor): [batch x time x pitch]
        """

        return F.nll_loss(prob, target, reduction='none')
