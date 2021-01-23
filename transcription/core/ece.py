import numpy as np
from tqdm import tqdm
from . import representation


def calculate_ece_list(pred_lists, label_lists, rep_type, n_bin=20):
    '''
    calculate expected calibration error(ECE)
    pred_lists: list of raw predictions.
                predictions: np.array shape of (frame, pitch, class)
    label_lists: list of labels.
                 labels: np.array shape of (frame, pitch)
    rep_type: states notation type
    bin: number of bins to calibrate errors
    '''
    
    n_class = pred_lists[0].shape[-1]

    cum_results = np.zeros((n_class, n_bin, 4))
    for pred, label in tqdm(zip(pred_lists, label_lists)):
        label = representation.convert_representation_np(label, rep_type)
        for n in range(n_class):
            class_pred = pred[:, :, n]
            positives = (label == n)
            for bin_idx in range(n_bin):
                bin_range = (bin_idx / n_bin, (bin_idx + 1) / n_bin)
                pred_in_bin = np.logical_and(bin_range[0] <= class_pred, class_pred < bin_range[1])
                n_in_bin = np.sum(pred_in_bin)
                n_p = np.sum(np.logical_and(pred_in_bin, positives))
                n_n = n_in_bin - n_p
                conf_sum = np.sum(class_pred[pred_in_bin])
                cum_results[n, bin_idx, :] += np.array([n_in_bin, n_p, n_n, conf_sum])
    
    acc = cum_results[:, :, 1] / cum_results[:, :, 0]
    conf = cum_results[:, :, 3] / cum_results[:, :, 0]
    acc_and_conf = np.stack((acc, conf, cum_results[:, :, 0]), axis=-1)
    ece = np.abs(acc_and_conf[:, :, 0] - acc_and_conf[:, :, 1]) * acc_and_conf[:, :, 2]
    ece = np.nansum(ece) / np.nansum(acc_and_conf[:, :, 2])
    return acc_and_conf, ece


def calculate_acc_conf(pred, label, n_bin=20):
    '''
    calculate accuracy, confidence and number of element in bins
    to get expected calibration error(ECE)
    pred: np.array shape of (frame, pitch, class)
    label: np.array shape of (frame, pitch)
    bin: number of bins to calibrate errors
    '''

    
    n_class = pred.shape[-1]

    acc_and_conf = np.zeros((n_class, n_bin, 3))

    for n in range(n_class):
        class_pred = pred[:, :, n]
        positives = (label == n)
        # print(np.sum(positives))
        for bin_idx in range(n_bin):
            bin_range = (bin_idx / n_bin, (bin_idx + 1) / n_bin)
            pred_in_bin = np.logical_and(bin_range[0] <= class_pred, class_pred < bin_range[1])
            n_in_bin = np.sum(pred_in_bin)
            n_p = np.sum(np.logical_and(pred_in_bin, positives))
            # n_n = n_in_bin - n_p
            conf_sum = np.sum(class_pred[pred_in_bin])
            acc_and_conf[n, bin_idx, :] = np.array([n_p / n_in_bin, conf_sum / n_in_bin, n_in_bin])
    
    return acc_and_conf

