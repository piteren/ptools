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

from multiprocessing import cpu_count
from queue import Empty
import os
import pandas as pd
import plotly.express as px
import random
import time
from typing import Callable, List

from ptools.lipytools.decorators import timing
from ptools.lipytools.little_methods import stamp, w_pickle, r_pickle
from ptools.pms.paspa import PaSpa
from ptools.pms.hpmser_wraps import interface_wrap_H, interface_wrap_MP_H, wrap_que


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


# number of points in search_RL closer than dst to given point
def _num_of_close(
        point :dict,
        search_RL :List[SeRes],
        paspa :PaSpa,
        dst: float):            # distance to classify point as close to another
    n_close = 0
    for p in search_RL:
        if paspa.dist(p.point, point) <= dst: n_close += 1
    return n_close

# returns smooth score for given point in relation to other in search_RL
def _smooth_score(
        point :dict,
        search_RL :List[SeRes],
        paspa :PaSpa,
        dst_smth: float):

    tot_weight = 0
    s_val = 0
    for zix in range(len(search_RL)):
        dst = paspa.dist(point, search_RL[zix].point)
        weight = (dst_smth-dst) / dst_smth
        if weight < 0: weight = 0
        s_val += search_RL[zix].score * weight
        tot_weight += weight

    if tot_weight > 0: s_val /= tot_weight
    # no neighbour points >> average of all
    else:
        accs = [sr.score for sr in search_RL]
        s_val = sum(accs) / len(accs)
    return s_val

# smooths search_RL (updates .smooth_score)
def _smooth_RL(
        search_RL: List[SeRes],
        paspa :PaSpa,
        dst_smth: float):

    for six in range(len(search_RL)):
        search_RL[six].smooth_score = _smooth_score(
            point=      search_RL[six].point,
            search_RL=  search_RL,
            paspa=      paspa,
            dst_smth=   dst_smth)

# smooths & sorts
def _smooth_and_sort(
        search_RL :List[SeRes],
        paspa :PaSpa,
        dst_smth :float):
    _smooth_RL(
        search_RL=  search_RL,
        paspa=      paspa,
        dst_smth=   dst_smth)
    search_RL.sort(key=lambda x: x.smooth_score, reverse=True)

# updates list, smooths, sorts, writes
def _update_and_save(
        name,
        search_RL :List[SeRes],
        paspa :PaSpa,
        dst_smth :float,
        new_SR :SeRes or List[SeRes]=   None,
        hpmser_FD :str=                 None):

    if new_SR:
        if type(new_SR) is not list: new_SR = [new_SR]
        search_RL += new_SR

    _smooth_and_sort(
        search_RL=  search_RL,
        paspa=      paspa,
        dst_smth=   dst_smth)

    if hpmser_FD:
        w_pickle((search_RL, paspa), f'{hpmser_FD}/{name}/{name}_results.srl')
        _write_graph(
            name=       name,
            search_RL=  search_RL,
            hpmser_FD=  hpmser_FD)
    return search_RL

# returns sample closer to local maximum
def _get_opt_sample(
        paspa :PaSpa,
        search_RL :List[SeRes], # ...should be smoothed and sorted!
        dst_smpl,
        prob_max,               # probability of sampling from max smooth sample area
        prob_top,               # probability of weighted sampling from area of n_top samples
        n_top):

    if random.random() < prob_max or not search_RL:
        max_point = search_RL[0].point if search_RL else None
        spoint = paspa.sample_point(
            ref_point=  max_point,
            ax_dst=     dst_smpl)
    else:
        prob_top = prob_top / (1 - prob_max) # adjust
        if random.random() < prob_top:
            if n_top > len(search_RL): n_top = len(search_RL)
            spoints = []
            for ix in range(n_top):
                ref_point = search_RL[ix].point
                spoint = paspa.sample_point(
                    ref_point=  ref_point,
                    ax_dst=     dst_smpl)
                spoints.append(spoint)
            spw = [_smooth_score(sp, search_RL, paspa, dst_smth=dst_smpl) for sp in spoints] # list of scores
            min_spw = min(spw)
            spw = [w-min_spw for w in spw] # subtract min
            spoint = random.choices(spoints, weights=spw, k=1)[0]
        else: spoint = paspa.sample_point(ref_point=None) # sample from whole space

    return spoint

def _get_clusters(
        search_RL: List[SeRes], # should be sorted
        paspa: PaSpa,
        dst :float):            # distance to classify point as close to another
    clusters = {}
    for srIX in range(len(search_RL)):
        sr = search_RL[srIX]
        in_cluster = False
        for ix in clusters:
            if paspa.dist(sr.point, search_RL[ix].point) <= dst:
                clusters[ix][1].append(srIX)
                in_cluster = True
        if not in_cluster: clusters[srIX] = (sr.smooth_score, [srIX])
    return clusters

# prepares nice string of results
def _nice_results_str(
        name,
        search_RL :List[SeRes], # should be sorted
        paspa :PaSpa,
        dst :float,       # distance to classify point as close to another
        n_clusters=     30):
    results = f'Search run {name} - {len(search_RL)} results\n\n{paspa}\n'

    clusters = _get_clusters(search_RL, paspa, dst)
    sorted_clusters = list(clusters.keys())
    sorted_clusters.sort(key= lambda x : clusters[x][0], reverse=True)
    if len(clusters) < n_clusters: n_clusters = len(clusters) // 2

    results += f'\nGot {len(clusters)} clusters, top {n_clusters} clusters:\n'
    results += '  smooth [   local]   id(nicl)   max   min   dif {params...}\n'
    top_srIX = []
    for cIX in range(n_clusters):
        srIX = sorted_clusters[cIX]
        top_srIX.append(srIX)
        sr = search_RL[srIX]
        cl = clusters[srIX]
        scores = [search_RL[sIX].score for sIX in clusters[srIX][1]]
        maxs = max(scores)
        mins = min(scores)
        results += f'{sr.smooth_score:8.5f} [{sr.score:8.5f}] {sr.id:4d}({len(cl[1]):4d}) {maxs:.3f} {mins:.3f} {maxs-mins:.3f} {PaSpa.point_2str(sr.point)}\n'

    results += '\n   x '
    for srIX in top_srIX: results += f'{search_RL[srIX].id:4d} '
    results += '\n'
    for srIX in top_srIX:
        results += f'{search_RL[srIX].id:4d} '
        for srIXB in top_srIX:
            if srIX != srIXB: results += f'{paspa.dist(search_RL[srIX].point, search_RL[srIXB].point):.2f} '
            else: results += '   - '
        results += '\n'

    avg_score = sum([p.score for p in search_RL]) / len(search_RL)
    results +=f'\nResults by smooth_score (avg_smooth: {avg_score:8.5f}):\n'
    results += '  smooth [   local]   id(n_cl) {params...}\n'
    for sr in search_RL:
        n_close = _num_of_close(sr.point, search_RL, paspa, dst)
        results += f'{sr.smooth_score:8.5f} [{sr.score:8.5f}] {sr.id:4d}({n_close:4d}) {PaSpa.point_2str(sr.point)}\n'
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
        print('\nenter data for axes (default 0,1,2,last):')
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

# load results, show graph and save, print results
def show_hpmser_resuls(
        hpmser_FD :str,
        dst_smth :float):

    results_FDL = sorted(os.listdir(hpmser_FD))
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
    _smooth_and_sort(search_RL, paspa, dst_smth=dst_smth)

    _write_graph(name, search_RL, hpmser_FD, silent=False)
    print(f'\n{_nice_results_str(name, search_RL, paspa, dst=dst_smth)}')
    return name, search_RL, paspa

# hpms searching function
@timing
def hpmser(
        func :Callable,                     # function which parameters need to be optimized
        psd :dict,                          # dictionary defining the space of parameters
        continue_last=              False,  # flag to continue last search from hpmser_FD
        name :str=                  None,   # for None stamp will be used
        add_stamp=                  True,   # adds short stamp to name, when name given
        dst_smth=                   0.1,    # smoothing distance (L1N) (also distance for sampling)
        prob_max=                   0.1,    # probability of sampling from max area
        prob_top=                   0.5,    # probability of weighted sampling from n_weighted samples
        n_top=                      10,
        def_kwargs :dict=           None,   # func kwargs
        devices=                    None,   # devices to use for search
        use_all_cores=              True,   # True: when devices is None >> uses all cores, otherwise as set by devices
        subprocess=                 True,   # True: runs func in subprocesses, otherwise in this process
        n_loops=                    None,   # limit for number of search loops
        hpmser_FD : str or bool=    None,   # folder, where save search results and html, for None does not save, for True uses default
        verb=                       1):

    # manage hpmser_FD, create if needed
    if hpmser_FD is True: hpmser_FD = 'hpmser' # default for True
    if not os.path.isdir(hpmser_FD):
        os.mkdir(hpmser_FD)
        continue_last = False

    search_RL = []
    max_SR = None
    paspa = None

    if continue_last:
        results_FDL = sorted(os.listdir(hpmser_FD))
        if len(results_FDL):
            if len(results_FDL) > 1:
                print(f'\nThere are {len(results_FDL)} searches:')
                for ix in range(len(results_FDL)): print(f' > {ix:2d}: {results_FDL[ix]}')
            print(f'will continue with the last one')
            name = results_FDL[-1] # take last

            search_RL, paspa = r_pickle(f'{hpmser_FD}/{name}/{name}_results.srl')

            search_RL = sorted(search_RL, key=lambda x: x.smooth_score, reverse=True)  # sort
            max_SR = search_RL[0]
    else:
        if not name: name = stamp()
        elif add_stamp: name = f'{stamp(letters=0)}_{name}'

        # create subfolder if needed
        subfolder = f'{hpmser_FD}/{name}'
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)
            continue_last = False

    if verb > 0:
        print(f'\n*** hpmser *** {name} started for {func.__name__} ...')
        print(f'    dst_smth {dst_smth}, prob_max {prob_max}, prob_top {prob_top}')
        if continue_last and search_RL: print(f'    search will continue with {len(search_RL)} results...')

    if not paspa:
        paspa = PaSpa(
            psd=    psd,
            seed=   None,
            verb=   verb-1)
    if verb>0: print(f'\n{paspa}\n')

    # manage devices
    if not subprocess: use_all_cores = False
    if devices is None: devices = [None] * (cpu_count() if use_all_cores else 1) # manage case of None for devices
    assert subprocess or (not subprocess and len(devices)==1), 'ERR: cannot use many devices without subprocess'

    loop_func = interface_wrap_MP_H if subprocess else interface_wrap_H

    # manage def_kwargs
    if not def_kwargs: def_kwargs = {}
    for k in psd:
        if k in def_kwargs: def_kwargs.pop(k)

    runIX = len(search_RL)
    try:
        while True:

            if verb>1: print(f' > got {len(devices)} devices at {runIX} loop start')
            # use all available devices
            while devices:
                spoint = _get_opt_sample(
                    paspa=          paspa,
                    search_RL=      search_RL,
                    dst_smpl=       dst_smth,
                    prob_max=       prob_max,
                    prob_top=       prob_top,
                    n_top=          n_top)
                loop_func(
                    func=       func,
                    device=     devices.pop(0),
                    spoint=     spoint,
                    s_time=     time.time(),
                    **def_kwargs)

            print('got here also...')
            # flush que
            resL = [wrap_que.get()] # at least one
            while True:
                try:            resL.append(wrap_que.get_nowait())
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
                    dst_smth=   dst_smth,
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
        dst_smth=   dst_smth,
        hpmser_FD=  hpmser_FD)

    results = _nice_results_str(name, search_RL, paspa, dst=dst_smth)
    if hpmser_FD:
        with open( f'{hpmser_FD}/{name}/{name}_results.txt', 'w') as file: file.write(results)

    if verb>0: print(results)

    return search_RL


def example_hpmser_smpl(
        n_samples,
        hpmser_FD,
        dst_smth,
        prob_max,
        prob_top,
        n_top):

    name, search_RL, paspa = show_hpmser_resuls(hpmser_FD, dst_smth=dst_smth)

    num_0 = 0
    sum_n = 0
    avg_dist = 0
    avg_sc = 0
    for ix in range(n_samples):
        spoint = _get_opt_sample(
            paspa=          paspa,
            search_RL=      search_RL,
            dst_smpl=       dst_smth,
            prob_max=       prob_max,
            prob_top=       prob_top,
            n_top=          n_top)

        n_close = _num_of_close(spoint, search_RL, paspa, dst_smth)
        dist = paspa.dist(search_RL[0].point, spoint)
        smooth_score = _smooth_score(spoint, search_RL, paspa, dst_smth)

        if n_close == 0: num_0 += 1
        sum_n += n_close
        avg_dist += dist
        avg_sc += smooth_score

        print(f'{ix} point got {n_close} n_close, {dist:4.2f} {smooth_score:8.5f}')

    print(f'sampling stats:')
    print(f'n0:{num_0}, nALL:{sum_n}, avg_dist:{avg_dist/n_samples:.3f}, avg_sc:{avg_sc/n_samples}')


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
    example_hpmser_smpl(
        n_samples=      1000,
        hpmser_FD=      '_hpmser',
        dst_smth=       0.05,
        prob_max=       0.1,
        prob_top=       0.5,
        n_top=          10)