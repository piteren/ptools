"""

 2020 (c) piteren

    hpmser - hyperparameters searching function
        > searches hyperparameters space to MAXIMIZE the score

    parameters:

        func - input function:
            > may have optional ‘device’ or ‘devices’ parameter - for NN, to run on given device
            > case of stochastic score for given hpms must be implemented of func (calculate many and average)
            > returns: a dict with ‘score’ key or a single value (score)
        psd - dictionary with parameters space to search in, check PaSpa @putils.neuralmess.params_dict
        def_kwargs - dict with other parameters of func
        devices - devices to use by hpmser, for syntax check @putils.neuralmess.dev_manager

"""

from inspect import getfullargspec
from multiprocessing import Queue, cpu_count
from queue import Empty
import os
import pandas as pd
import plotly.express as px
import random
import time
from typing import Callable, List

from ptools.lipytools.little_methods import stamp, w_pickle, r_pickle
from ptools.pms.paspa import PaSpa
from ptools.mpython.mpdecor import qproc

# single search result
class SeRes:

    def __init__(
            self,
            id: int,
            point: dict,
            score=      None):

        self.id = id
        self.point = point
        self.score = score
        self.smooth_score = None

# calculates smooth for given list of SeRes
def _smooth_RL(
        search_RL: List[SeRes], # list of search results
        paspa :PaSpa,           # parameters space
        rad: float=     0.5):   # smoothing radius

    # returns smoth score for given point in relation to other in search_RL
    def smooth_score(point :dict):

        tot_weight = 0
        s_val = 0
        for zix in range(len(search_RL)):
            dst = paspa.dist(point, search_RL[zix].point)
            weight = (rad-dst) / rad
            if weight < 0: weight = 0
            s_val += search_RL[zix].score * weight
            tot_weight += weight

        if tot_weight > 0: s_val /= tot_weight
        # no neighbour points >> average of all
        else:
            accs = [sr.score for sr in search_RL]
            s_val = sum(accs) / len(accs)
        return s_val

    for six in range(len(search_RL)):
        search_RL[six].smooth_score = smooth_score(search_RL[six].point)

# returns sample closer to local maximum
def _get_opt_sample(
        paspa :PaSpa,
        search_RL :List[SeRes]= None, # ...should be smoothed and sorted!
        ax_rrad=                None,
        space_prob :float=      0.5):

    max_point = search_RL[0].point if search_RL and random.random() > space_prob else None
    return paspa.sample_point(max_point, ax_rrad)

# prepares nice string of results
def _nice_results_str(
        name,
        search_RL,
        paspa):
    results = f'Search run {name} finished, {len(search_RL)} results by smooth_score:\n\n{paspa}\n\n'
    results += '  - smooth [local] id {params...} -\n'
    for sr in search_RL: results += f'{sr.smooth_score:9.5f} [{sr.score:9.5f}] {sr.id:4d}: {PaSpa.point_2str(sr.point)}\n'
    return results

# writes 3D graph to html with plotly
def _write_graph(
        name,
        search_RL :List[SeRes],
        hpmser_FD :str,
        silent=     True):

    columns = sorted(list(search_RL[0].point.keys())) + ['score','smooth_score']
    axes_data = {
        'x':    columns[0],
        'y':    columns[1],
        'z':    columns[2],
        'c':    columns[-1]}

    if not silent:
        print('\nResults got data for axes:')
        for ix in range(len(columns)): print('  %d: %s'%(ix,columns[ix]))
        axes = ['x', 'y', 'z', 'c']
        print('\nenter data for axes:')
        for ix in range(4):
            ax = axes[ix]
            v = input('%s:'%ax)
            try:                v = int(v)
            except ValueError:  v = None
            if v is not None: axes_data[ax] = columns[v]

    valLL = [[res.point[key] for key in columns[:-2]] + [res.score, res.smooth_score] for res in search_RL]
    df = pd.DataFrame(
        data=       valLL,
        columns=    columns)
    std = df[axes_data['c']].std()
    mean = df[axes_data['c']].mean()
    off = 2*std
    cr_min = mean - off
    cr_max = mean + off

    fig = px.scatter_3d(
        data_frame=     df,
        title=          f'{name} {len(df[columns[0]])} results',
        x=              axes_data['x'],
        y=              axes_data['y'],
        z=              axes_data['z'],
        color=          axes_data['c'],
        range_color=    [cr_min,cr_max],
        opacity=        0.7,
        width=          700,
        height=         700)

    file = f'{hpmser_FD}/{name}/{name}_results.html'
    fig.write_html(file, auto_open=False if os.path.isfile(file) else True)

# updates list, sorts, writes
def _update_and_save(
        name,
        search_RL :List[SeRes], # ...should be smoothed
        paspa :PaSpa,
        rad :float,
        new_SR :SeRes or List[SeRes]=   None,
        hpmser_FD :str=                 None):

    if new_SR:
        if type(new_SR) is not list: new_SR = [new_SR]
        search_RL += new_SR
    _smooth_RL(search_RL, paspa, rad)  # update smooth
    search_RL = sorted(search_RL, key=lambda x: x.smooth_score, reverse=True)  # sort
    if hpmser_FD:
        w_pickle((search_RL, paspa), f'{hpmser_FD}/{name}/{name}_results.srl')
        _write_graph(
            name=       name,
            search_RL=  search_RL,
            hpmser_FD=  hpmser_FD)
    return search_RL

# load results, show graph and save, print results
def show_hpmser_resuls(
        hpmser_FD :str):

    results_FDL = sorted(os.listdir(hpmser_FD))
    if len(results_FDL):
        rIX = -1
        if len(results_FDL) > 1:
            print(f'\nThere are {len(results_FDL)} searches, choose one (default last):')
            for ix in range(len(results_FDL)):
                print(f' > {ix:2d}: {results_FDL[ix]}')
            rIX = input('> ')
            try:
                rIX = int(rIX)
                if rIX > len(results_FDL): rIX = -1
            except ValueError:
                rIX = -1
        name = results_FDL[rIX]

        search_RL, paspa = r_pickle(f'{hpmser_FD}/{name}/{name}_results.srl')
        _write_graph(name, search_RL, hpmser_FD, silent=False)
        print(_nice_results_str(name, search_RL, paspa))

# hpms searching function
def hpmser(
        func :Callable,                     # function which parameters need to be optimized
        psd :dict,                          # dictionary defining the space of parameters
        name :str=                  None,   # for None stamp will be used
        rad :float=                 0.5,    # radius for smoothing
        ax_rrad :float=             0.3,    # relative distance on axis of space for sampling, should be < 1
        space_prob :float=          0.5,    # probability of sampling whole space (exploration)
        def_kwargs :dict=           None,   # func kwargs
        devices=                    None,   # devices to use for search
        use_all_cores=              True,   # True: when devices is None >> uses all cores, otherwise as set by devices
        subprocess=                 True,   # True: runs func in subprocesses, otherwise in this process
        n_loops=                    None,   # limit for number of search loops
        hpmser_FD : str or bool=    None,   # folder, where save search results and html, for None does not save, for True uses default
        verb=                       1):

    if not name: name = stamp()
    if verb > 0:
        print(f'\n*** hpmser *** {name} started for {func.__name__} ...')
        print(f'    rad {rad}, ax_rrad {ax_rrad}, space_prob {space_prob}')

    paspa = PaSpa(
        psd=    psd,
        seed=   None,
        verb=   verb-1)
    if verb>0: print(f'\n{paspa}\n')

    # set folders, create if needed
    if hpmser_FD:
        if hpmser_FD is True: hpmser_FD = 'hpmser'
        if not os.path.isdir(hpmser_FD): os.mkdir(hpmser_FD)
        os.mkdir(f'{hpmser_FD}/{name}')

    # manage devices
    if not subprocess: use_all_cores = False
    if devices is None: devices = [None] * (cpu_count() if use_all_cores else 1) # manage case of None for devices
    assert subprocess or (not subprocess and len(devices)==1), 'ERR: cannot use many devices without subprocess'

    # func wrap with interface
    def inner_func(
            device,
            spoint,
            s_time,
            kwargs):

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
            's_time':       s_time,
            'score':        score}

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

    # manage def_kwargs
    if not def_kwargs: def_kwargs = {}
    for k in psd:
        if k in def_kwargs: def_kwargs.pop(k)

    search_RL = []  # results list
    max_SR = None
    try:
        runIX = 0
        while True:

            if verb>1: print(f' > got {len(devices)} devices at {runIX} loop start')
            # use all available devices
            while devices:
                spoint = _get_opt_sample(
                    paspa=      paspa,
                    search_RL=  search_RL,
                    ax_rrad=    ax_rrad * random.random(), # use rad randomly decreased
                    space_prob= space_prob)
                loop_func(
                    device=     devices.pop(0),
                    spoint=     spoint,
                    s_time=     time.time(),
                    kwargs=     def_kwargs)

            # flush que
            resL = [que.get()] # at least one
            while True:
                try:            resL.append(que.get_nowait())
                except Empty:   break
            if verb > 1: print(f' > got {len(resL)} results in {runIX} loop')

            # manage results
            new_SRL = []
            for res in resL:
                devices.append(res['device'])   # return device
                new_SR = SeRes(           # add new point with results
                    id=     runIX,
                    point=  res['spoint'],
                    score=  res['score'])
                new_SRL.append(new_SR)
                runIX += 1

                if verb > 0:
                    print(f'R:{new_SR.id} {new_SR.score:6.3f} ', end='')
                    if max_SR: print(f'[{paspa.dist(max_SR.point, new_SR.point):.2f}] {max_SR.id} (max: {max_SR.smooth_score:6.3f}/{max_SR.score:6.3f}) ', end='')
                    print(f'{PaSpa.point_2str(new_SR.point)} {time.time() - res["s_time"]:.1f}s')

            if new_SRL:
                search_RL = _update_and_save(
                    name=       name,
                    search_RL=  search_RL,
                    paspa=      paspa,
                    rad=        rad,
                    new_SR=     new_SRL,
                    hpmser_FD=  hpmser_FD)

            max_SR = search_RL[0]

            if n_loops and runIX >= n_loops:
                if verb>0: print(f'...n_loops ({n_loops}) done!')
                break

    except KeyboardInterrupt:
        if verb>0: print('...interrupted')

    search_RL = _update_and_save(
        name=       name,
        search_RL=  search_RL,
        paspa=      paspa,
        rad=        rad,
        hpmser_FD=  hpmser_FD)

    results = _nice_results_str(name, search_RL, paspa)
    if hpmser_FD:
        with open( f'{hpmser_FD}/{name}_results.txt', 'w') as file: file.write(results)

    if verb>0: print(results)

    return search_RL


def example_hpmser(
        n_proc=     10,
        av_time=    1, # avg num seconds function calculates
        verb=       1):

    import time
    import math

    def some_func(
            name :str,
            device,
            a :int,
            b :float,
            c: float,
            d: float,
            wait=   0.1,
            verb=   0):

        val = math.sin(b-a*c) - abs(a+3.1)/(d+0.5) - pow(b/2,2)/12
        time.sleep(random.random()*wait)
        if verb>0 :print(f'... {name} calculated on {device} ({a}) ({b}) >> ({val})')
        return val

    psd = {
        'a':    [-5,5],
        'b':    [-5.0,5.0],
        'c':    [-2.0,2],
        'd':    [0.0,5]}

    hpmser(
        func=       some_func,
        psd=        psd,
        rad=        0.2,
        ax_rrad=    0.2,
        space_prob= 0.5,
        def_kwargs= {'name':'pio', 'a':3, 'wait':av_time*2, 'verb':verb-1},
        devices=    [None]*n_proc,
        #subprocess= False,
        hpmser_FD=  True,
        verb=       verb)


if __name__ == '__main__':

    #example_hpmser()
    show_hpmser_resuls('hpmser')