"""

 2020 (c) piteren

    mrun - multi run function

    assuming there is a callable func, that we want to run with a list of kwargs in parallel on devices
    there is mrun that will do it for you

"""

from inspect import getfullargspec
from multiprocessing import Queue, cpu_count
from queue import Empty
import random
import time
from typing import Callable, List

from ptools.lipytools.decorators import timing
from ptools.mpython.mpdecor import qproc


# multi-running function
@timing
def mrun(
        func :Callable,             # function to run
        kwargsL :List[dict],        # list of function kwargs (dicts) for runs
        devices=            None,   # devices to use
        use_all_cores=      True,   # True: when devices is None >> uses all cores, otherwise as set by devices
        subprocess=         True,   # True: runs func in subprocesses, otherwise in this process
        verb=               1):

    # manage devices
    if not subprocess: use_all_cores = False
    if devices is None: devices = [None] * (cpu_count() if use_all_cores else 1) # manage case of None for devices
    assert subprocess or (not subprocess and len(devices)==1), 'ERR: cannot use many devices without subprocess'

    if verb > 0: print(f'\nmrun starting for {len(kwargsL)} task for {func.__name__} on {len(devices)} devices')

    # func wrap with interface
    def inner_func(
            kwargs :dict,
            kwIX :int,
            device,
            s_time):

        # eventually add device/devices to func kwargs (if func accepts device)
        pms = getfullargspec(func).args
        for k in ['device','devices']:
            if k in pms: kwargs[k] = device

        res = func(**kwargs) # call

        return {
            'result':       res,
            'kwIX':         kwIX,
            'device':       device,
            's_time':       s_time}

    que = Queue() # results always will be returned via que (loop architecture)

    # func wrap with interface
    def interface_wrap(**kwargs):
        result = inner_func(**kwargs)
        que.put(result)

    # func wrap with interface and MP
    @qproc(que)
    def interface_wrap_MP(**kwargs):
        return inner_func(**kwargs)

    loop_func = interface_wrap_MP if subprocess else interface_wrap

    kwargsD = {ix: kwargsL[ix] for ix in range(len(kwargsL))} # change list to dict

    resultsD = {}
    runIX = 0
    while len(resultsD) < len(kwargsD):

        if verb>1: print(f' > got {len(devices)} devices at {runIX} loop start')
        # use all available devices
        while devices and runIX < len(kwargsD):
            loop_func(
                kwargs=     kwargsD[runIX],
                kwIX=       runIX,
                device=     devices.pop(0),
                s_time=     time.time())
            runIX += 1

        # flush que
        resL = [que.get()] # at least one
        while True:
            try:            resL.append(que.get_nowait())
            except Empty:   break
        if verb > 1: print(f' > got {len(resL)} results in {runIX} loop')

        # manage results
        ts_len = len(str(len(kwargsL)))
        for res in resL:
            devices.append(res['device']) # return device
            resultsD[res['kwIX']] = res['result']
            if verb > 0: print(f' > #{res["kwIX"]:{ts_len}d} processed ({time.time() - res["s_time"]:.1f}s)')

    return [resultsD[ix] for ix in range(len(resultsD))]


def example_mrun(
        n_proc=     10,
        verb=       1):

    import time

    def some_func(
            name :str,
            device= None,
            wait=   2,
            verb=   0):

        time.sleep(random.random()*wait)
        if verb>0 :print(f'... {name} calculated on {device}')
        return f'done {name}'

    names = ['ala','bala','gila','deryl'] *10
    kwL = [{'name': names[ix]} for ix in range(len(names))]

    resL = mrun(
        func=       some_func,
        kwargsL=    kwL,
        devices=    [None]*n_proc,
        verb=       verb)
    print(resL)


if __name__ == '__main__':

    example_mrun()