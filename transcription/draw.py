import plotly.graph_objects as go
from scipy.stats import entropy
import numpy as np
import torch as th
from plotly.subplots import make_subplots
from mir_eval.transcription import match_notes
from mir_eval.util import hz_to_midi

from core.beam import find_candidate
from core import decoding




def draw_multi_dim_pred(arr):
    '''
    arr: numpy array shape of (time, pitch, class)
    return plotly.Heatmap of array, flatten last dimension 
    '''

    arr = arr[:, :, 1:]
    flatten = arr.reshape(arr.shape[0], -1)
    return go.Heatmap(z=flatten.T)


def draw_annotated_pred(pred, label):
    '''
    pred: numpy array shape of (time, pitch, class)
        class[0, 1, 2, 3, 4] -> [off, offset, onset, frame, reonset]
    label: numpy array shape of (time, pitch), elements are index of 
        ground truth class
    '''
    
    binary_pred = (np.argmax(pred, axis=-1) >= 1)
    binary_label = (label >= 1)
    return draw_annotated_pred_binary(binary_pred, binary_label)


def draw_annotated_pred_binary(binary_pred, binary_label):
    '''
    draw difference(frame based) with plotly heatmap.
    colorcode={
        hit: blue,
        false positive: yellow,
        false negetive: red
    }
    '''
    posit = np.logical_and(binary_pred, binary_label)
    fp = np.logical_and(binary_pred, 1 - binary_label)
    fn = np.logical_and(1 - binary_pred, binary_label)
    assert(np.sum(posit * fp * fn) == 0)
    
    annotated_arr = posit + 2*fp + 3*fn

    dcolorsc = discrete_colorscale([0, 1, 2, 3, 4], ['#FFFFFF', '#0035FF', '#FFFF00', '#FF0000'])
    return go.Heatmap(z=annotated_arr.T, colorscale=dcolorsc, zmin=0, zmax=4)
 

def notewise_diff_roll(argmax_pred, label):
    onsets = ((argmax_pred == 2) + (argmax_pred == 4)).astype(np.int)
    offsets = (argmax_pred == 1).astype(np.int)
    frames = ((argmax_pred >= 1)).astype(np.int)
    
    p_est, i_est = decoding.simple_decoding_wrapper(onsets, frames)
    p_ref, i_ref = decoding.simple_decoding_wrapper((label == 2) + (label == 4).astype(np.int), (label >=1).astype(np.int))

    matching = match_notes(i_ref, p_ref, i_est,
                           p_est, onset_tolerance=0.05,
                           pitch_tolerance=50.0,
                           offset_ratio=None,
                           offset_min_tolerance=0.05,
                           strict=False)
    matched_ref, matched_est = zip(*matching)

    roll_shape = label.shape

    def _intervals_to_annotated_roll(pitch_list, interval_list, matched_index_list, shape):
        roll = np.zeros(shape)
        for n, (pitch, (onset, offset)) in enumerate(zip(pitch_list, interval_list)):
            hit = True if n in matched_index_list else False
            pitch = int(round(hz_to_midi(pitch) - 21))
            onset = int(round(16000/512 * onset))
            offset = int(round(16000/512 * offset))
            roll[onset:offset, pitch] = 1 if hit else 4
            roll[onset, pitch] = 2
        return roll

    return _intervals_to_annotated_roll(p_est, i_est, matched_est, roll_shape), \
        _intervals_to_annotated_roll(p_ref, i_ref, matched_ref, roll_shape)


def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors)+1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)     
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values
    
    dcolorscale = [] #discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale


def beam_candidiate_plot(pred, label):
    pred_entropy = np.apply_along_axis(entropy, 2, pred)
    candidates = find_candidate(pred, label)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes=True, row_heights=[10,10,10], vertical_spacing=0.01)
    fig.add_trace(go.Heatmap(z=pred_entropy[:500].T), row=1, col=1)
    fig.add_trace(go.Heatmap(z=candidates[:500].T), row=2, col=1)
    fig.add_trace(draw_annotated_pred(pred[:500], label[:500]), row=3, col=1)
    fig.update_layout(
        autosize=False,
        margin=dict(l=20, r=20, t=0, b=0),
        width=1000,
        height=800)
    fig['layout']['yaxis'].update(matches='y3')
    fig['layout']['yaxis2'].update(matches='y3')
    fig.show()


def calculate_entropy(pred):
    raise NotImplementedError