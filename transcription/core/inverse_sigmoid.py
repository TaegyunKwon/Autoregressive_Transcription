import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def inverse_sigmoid(step, max_step, v_max, v_min, clip_range=3):
    '''
    calculate value of inversed sigmoid. use [-clip_range, clip_range]
    of simoid function as template, and transform into inversed sigmoid form,
    which have f(0) = v_max, f(max_step) = v_min
    '''
    assert v_max > v_min
    assert 0 <= step <= max_step
    assert max_step >= 1
    assert clip_range > 0

    return v_max - (v_max - v_min) * (sigmoid(step/max_step * clip_range * 2 - clip_range) - sigmoid(-clip_range)) / (sigmoid(clip_range)- sigmoid(-clip_range))
