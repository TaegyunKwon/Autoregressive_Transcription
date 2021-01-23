import os
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import heapq

from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import csv
from kornia.losses import DiceLoss

from sacred import Experiment, SETTINGS
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
from sacred.utils import apply_backspaces_and_linefeeds

from .core import *
from .core import utils, models, focal_loss
from .evaluate import evaluate, evaluate_onf

SETTINGS.CAPTURE_MODE = 'sys'

def remove_progress(captured_out):
    lines = (line for line in captured_out.splitlines() if ('it/s]' not in line) and ('s/it]' not in line))
    return '\n'.join(lines)

ex = Experiment('onset_frame')
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def config():
    device = list(range(torch.torch.cuda.device_count())) if torch.cuda.is_available() else []
    iterations = 200000
    resume_iteration = None
    checkpoint_interval = 5000
    load_mode = 'ram'
    num_workers = 0
    
    debug = False


@ex.config
def model_config():
    model_name = 'ARModel'

    n_fft = 2048
    n_mels = 229
    f_min = 30
    f_max = 8000

    recursive = True

    rep_type = 'base'
    if rep_type == 'base':
        n_class = 5
    elif rep_type in ['four', 'three_re']:
        n_class = 4
    elif rep_type == 'three':
        n_class = 3
    elif rep_type == 'binary':
        n_class = 2
    delay = 1

    win_bw = 0
    win_fw = 0
    ac_model_type = 'simple_conv'
    lm_model_type = 'lstm'
    cnn_unit = 48
    lstm_unit = 768 
    fc_unit = 768
    bi_lstm = False

    model_variables = ['model_name', 'recursive', 'rep_type', 'n_class', 
        'delay', 'context_len', 'ac_model_type', 'lm_model_type', 'cnn_unit', 
        'lstm_unit', 'fc_unit', 'bi_lstm']

@ex.config
def training_config(device, debug):
    dataset_name = 'MAESTRO'
    batch_size = 32
    assert batch_size % len(device) == 0
  
    sequence_length = SAMPLE_RATE * 10 # will be clipped as N // HOP_LENGTH * HOP_LENGTH
   
    learning_rate = 0.0006
    learning_rate_decay_steps = 5000
    learning_rate_decay_rate = 0.95
    clip_gradient_norm = 3
    weight_decay = 1e-7

    validation_interval = 5000
    validation_length = sequence_length

    sampling_method = 'gt' # one of ['gt', dist', 'argmax', 'random']
    # only valid when sampling_method != 'gt'
    sampling_curve = 'i_sigmoid' # one of ['i_sigmoid', 'linear']
    e_start = 1.0
    e_end = 0.0
    schedule_steps = 10000
    
    criterion_type = 'nllloss'
    assert criterion_type in ['kl_div', 'nllloss', 'focalloss', 'diceloss']

    smoothing_alpha = 0.0
    if smoothing_alpha != 0.0 and criterion_type != 'kl_div':
        raise ValueError('Lable smoothing need kl_div')
    gamma = 2.0

@ex.config
def log_config(model_name, sampling_method, criterion_type, learning_rate):
    logdir = Path('runs') / \
      ('_'.join([model_name, sampling_method, criterion_type, str(learning_rate)]) 
      + '_' + datetime.now().strftime('%y%m%d-%H%M%S'))
    ex.observers.append(FileStorageObserver.create(logdir))
    n_keep = 5
    

@ex.config
def debug_config(debug):
    if debug:
        validation_interval = 2
    else:
        pass

@ex.capture
def get_model(model_name, n_mels, n_fft, f_min, f_max, cnn_unit, lstm_unit, 
    fc_unit, bi_lstm, recursive, ac_model_type, lm_model_type, 
    n_class, win_bw, win_fw):
    if model_name == 'ARModel':
        model = models.ARModel(
            n_mels, n_fft, f_min, f_max, cnn_unit, lstm_unit, fc_unit, 
            bi_lstm, recursive=recursive, n_class=n_class, win_bw=win_bw, win_fw=win_fw, 
            ac_model_type=ac_model_type, lm_model_type=lm_model_type)
    elif model_name == 'ONF':
        model = models.OnsetsAndFrames(n_mels, n_fft, f_min, f_max, cnn_unit, fc_unit, bidirectional=bi_lstm)

    else:
        raise KeyError(f'invalid model name:{model_name}')
    return model

@ex.capture
def save_model(model, optimizer, iteration, device, logdir, model_name, n_mels, n_fft, f_min, f_max, cnn_unit, lstm_unit, 
    fc_unit, bi_lstm, recursive, ac_model_type, lm_model_type, n_class, win_bw, win_fw):
    # because of torch.nn.dataparallel
    state_dict = model.module.state_dict() if len(device) >= 2 else model.state_dict()
    torch.save({'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_name' : model_name,
                'iteration' : iteration,
                'n_class' : n_class,
                'win_fw' : win_fw,
                'win_bw' : win_bw,
                'n_mels' : n_mels,
                'n_fft' : n_fft,
                'f_min' : f_min,
                'f_max' : f_max,
                'cnn_unit' : cnn_unit,
                'lstm_unit' : lstm_unit,
                'fc_unit' : fc_unit,
                'bi_lstm' : bi_lstm,
                'recursive' : recursive,
                'ac_model_type' : ac_model_type,
                'lm_model_type' : lm_model_type},
                os.path.join(logdir, f'model-{iteration}.pt'))


def clean_up_checkpoints(logdir, n_keep, order='lower'): 
    ckpts = list(Path(logdir).glob('*.pt'))
    if len(ckpts) <= n_keep:
        pass
    else:
        with open(Path(logdir) / 'checkpoint.csv', "r") as f:
            reader = csv.reader(f, delimiter=',')
            data = [(el[0], float(el[1])) for el in list(reader)]
            lastest = np.argmax([int(el[0]) for el in data])
            if order == 'lower':
                idx = heapq.nsmallest(n_keep, range(len(data)), lambda x: data[x][1])
            elif order == 'higher':
                idx = heapq.nsmallest(n_keep, range(len(data)), lambda x: data[x][1])
            top_n = [data[i] for i in idx]
        for ckpt in ckpts:
            if ckpt.name not in [f'model-{el[0]}.pt' for el in top_n] and ckpt.name != f'model-{data[lastest][0]}.pt':
                ckpt.unlink()

        with open(Path(logdir) / 'checkpoint.csv', "w") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows([(el[0], el[1]) for el in top_n])

@ex.automain
def train(device, logdir, model_name, sequence_length, load_mode, debug, batch_size, 
          num_workers, validation_length, resume_iteration, n_keep, learning_rate, 
          weight_decay, learning_rate_decay_steps, learning_rate_decay_rate, 
          criterion_type, gamma, iterations, smoothing_alpha, schedule_steps, e_start, e_end, sampling_curve,
          sampling_method, clip_gradient_norm, validation_interval, checkpoint_interval,
          n_class, rep_type, recursive, delay,
          dataset_name):

    logdir = Path(logdir)      
    print_config(ex.current_run)
    assert all([el < torch.cuda.device_count() for el in device])
    default_device = 'cpu' if len(device) == 0 else f'cuda:{device[0]}'
    sequence_length = sequence_length // HOP_LENGTH * HOP_LENGTH

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir / 'train')
    valid_writer_gt = SummaryWriter(logdir / 'valid_gt')
    valid_writer_argmax = SummaryWriter(logdir / 'valid_argmax')

    if debug:
        if dataset_name == 'MAESTRO':
            dataset = MAESTRO(groups=['debug'], sequence_length=sequence_length, load_mode=load_mode, random_sample=True, delay=delay)
        validation_dataset = dataset
    else:
        if dataset_name == 'MAESTRO':
            dataset = MAESTRO(groups=['train'], sequence_length=sequence_length, load_mode=load_mode, random_sample=True, delay=delay)
            validation_dataset = MAESTRO(groups=['validation'], sequence_length=validation_length, load_mode=load_mode, 
                                        random_sample=False, delay=delay)
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)
    
    model = get_model()
    if resume_iteration is None:
        if len(device) >= 2:
          model = torch.nn.DataParallel(
              model, device_ids=device).to(default_device)
        else:
          model = model.to(default_device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
        resume_iteration = 0
    else:
        model_state_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        checkpoint = torch.load(model_state_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if len(device) == 1:
          model = model.to(default_device)
        elif len(device) >= 2:
          model = torch.nn.DataParallel(
              model, device_ids=device).to(default_device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay, eps=1e-6)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    if criterion_type == 'nllloss':
        criterion = utils.NLLLoss()
    elif criterion_type == 'kl_div':
        criterion = utils.LabelSmoothingLoss(label_smoothing=smoothing_alpha, class_size=n_class)
    else:
      raise KeyError(f'invalid criterion type:{criterion_type}')

    for i, batch in zip(loop, cycle(loader)):
        optimizer.zero_grad()

        gt_ratio = utils.teacher_ratio(i, schedule_steps, e_start, e_end, sampling_curve)
        if model_name == 'ARModel':
            _, loss = models.run_on_batch(model, batch, device[0], sampling_method=sampling_method, gt_ratio=gt_ratio, criterion=criterion, rep_type=rep_type, recursive=recursive, delay=delay)
        elif model_name == 'ONF':
            _, loss = models.run_on_batch_onf(model, batch, device[0])
        loss.mean().backward()

        if clip_gradient_norm:
            for parameter in model.parameters():
                clip_grad_norm_([parameter], clip_gradient_norm)

        optimizer.step()
        scheduler.step()
        loop.set_postfix_str("loss: {:.3e}".format(loss.mean()))
        writer.add_scalar('loss', loss.mean(), global_step=i)
        
        if i % validation_interval == 0 or i % checkpoint_interval == 0:
            if i % checkpoint_interval == 0:
                save_model(model, optimizer, i, device)
            print('')
            model.eval()
            with torch.no_grad():
                loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
                metrics_argmax = defaultdict(list)
                for j, batch in enumerate(loader):
                    if model_name == 'ARModel':
                        valid_result_argmax, _ = evaluate(batch, model, device, criterion=criterion, sampling_method='argmax',
                                                    rep_type=rep_type, recursive=recursive, delay=delay)
                    elif model_name == 'ONF':
                        valid_result_argmax, _ = evaluate_onf(batch, model, device, criterion=criterion, sampling_method='argmax',
                                                    rep_type=rep_type, recursive=recursive, delay=delay)
                    for key, value in valid_result_argmax.items():
                        metrics_argmax[key].extend(value)
                print('argmax metric')
                for key, value in metrics_argmax.items():
                    if key[-2:] == 'f1' or key == 'loss':
                        print(f'{key} : {np.mean(value)}')
                    valid_writer_argmax.add_scalar(key.replace(' ', '_'), np.mean(value), global_step=i)

                model.train()
                if i % checkpoint_interval == 0:
                    with open(Path(logdir) / 'checkpoint.csv', "a") as f:
                        csv_writer = csv.writer(f, delimiter=',')
                        csv_writer.writerow([i, float(np.mean(metrics_argmax['loss']))])
                    clean_up_checkpoints(logdir, n_keep=n_keep)
                                