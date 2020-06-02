"""

 2020 (c) piteren

    some wrappers for hpmser

"""

from inspect import getfullargspec
from multiprocessing import Queue

from ptools.mpython.mpdecor import qproc

# func wrap with interface
def inner_func_H(
        func,
        device,
        spoint,
        est_score,
        s_time,
        **kwargs):

    # eventually add device/devices to func kwargs (if func accepts device)
    pms = getfullargspec(func).args
    for k in ['device','devices']:
        if k in pms: kwargs[k] = device

    res = func(**spoint, **kwargs) # call

    if type(res) is dict:   score = res['score']
    else:                   score = res

    return {
        'device':       device,
        'spoint':       spoint,
        'est_score':    est_score,
        's_time':       s_time,
        'score':        score}

wrap_que = Queue()  # results always will be returned via que (loop architecture)

# func wrap with interface
def interface_wrap_H(**kwargs):
    result = inner_func_H(**kwargs)
    wrap_que.put(result)

# func wrap with interface and MP
@qproc(wrap_que)
def interface_wrap_MP_H(**kwargs):
    return inner_func_H(**kwargs)
