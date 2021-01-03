"""

 2020 (c) piteren

    hpmser - hyperparameters searching function
        > searches hyperparameters space to MAXIMIZE the SCORE for func

        MAXIMIZE the SCORE == find points cluster with:
         - high num of points in a small distance (dst)
         - high max smooth_score
         - high lowest score

         policy of sampling the space is crucial (determines the speed, top result and convergence)
         - fully random sampling is slow, wastes a lot of time and computing power
         - too aggressive sampling may undersample the space and miss the MAX

    parameters:

        func - input function:
            > may have optional ‘device’ or ‘devices’ parameter - for NN, to run on given CUDA device
            > returns: a dict with ‘score’ key or a single value (score)
        psd - dictionary with parameters space to search in, check PaSpa @putils.neuralmess.params_dict
        func_defaults - dict with other parameters of func (not to be optimized)
        devices - devices to use by hpmser, (syntax: check @ptools.neuralmess.dev_manager)

"""

#TODO:
# - add TB to hpmser
# - add GX to hpmser
# - add graphs as reports


from multiprocessing import cpu_count
import os
import pandas as pd
import plotly.express as px
import random
import sys, select
import time
from typing import Callable, List

from ptools.lipytools.decorators import timing
from ptools.lipytools.little_methods import stamp, w_pickle, r_pickle
from ptools.neuralmess.dev_manager import nestarter
from ptools.pms.paspa import PaSpa
from ptools.pms.hpmser_wraps import interface_wrap_H, interface_wrap_MP_H, wrap_que

NP_SMOOTH = [2,3,5,9] # numbers of points for smoothing

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
        self.smooth_score = score # default, to be updated


# for given point of space calculates
# - its smooth score calculated with np_smooth(int) closest points from search_RL
# - avg_dst: average distance between given point and its np_smooth closest points
# - all scores: scores aof all np_smooth points
def _smooth_score(
        point :dict,
        search_RL :List[SeRes],
        paspa :PaSpa,
        np_smooth :int):

    # case: no points in search_RL
    ss_np = 0
    avg_dst = 1
    all_scores = [0]

    if search_RL:

        sd = [[search_RL[srIX].score, paspa.dist(point, search_RL[srIX].point)] for srIX in range(len(search_RL))] # [[score,dst],..]
        sd.sort(key= lambda x: x[1]) # sort by distance to this point
        sd_np = sd[:np_smooth+1] # trim to np_smooth points (+1 point for reference)

        # one/two points case
        if len(sd_np) < 3:
            ss_np = sd_np[0][0] # closest point score
            all_scores = [ss_np]
        else:
            all_scores, all_dst = zip(*sd_np) # scores, distances

            max_dst = all_dst[-1] # distance of last(reference) point

            # trim them
            all_dst = all_dst[:-1]
            all_scores = all_scores[:-1]

            weights = [(max_dst-d)/max_dst for d in all_dst] # <1;0)
            wall_scores = [all_scores[ix]*weights[ix] for ix in range(len(all_scores))] # weighted scores

            ss_np = sum(wall_scores) / sum(weights)
            avg_dst = sum(all_dst) / len(all_dst)

    return ss_np, avg_dst, all_scores

# smooths search_RL (updates .smooth_score)
def _smooth_RL(
        search_RL: List[SeRes],
        paspa :PaSpa,
        np_smooth :int):

    avg_dst = []
    for six in range(len(search_RL)):
        search_RL[six].smooth_score, ad, _ = _smooth_score(
            point=      search_RL[six].point,
            search_RL=  search_RL,
            paspa=      paspa,
            np_smooth=  np_smooth)
        avg_dst.append(ad)
    return sum(avg_dst) / len(avg_dst)

# smooths & sorts
def _smooth_and_sort(
        search_RL :List[SeRes],
        paspa :PaSpa,
        np_smooth :int):
    avg_dst = _smooth_RL(
        search_RL=  search_RL,
        paspa=      paspa,
        np_smooth=  np_smooth)
    search_RL.sort(key=lambda x: x.smooth_score, reverse=True)
    return avg_dst

# updates list, smooths, sorts, writes with backup
def _update_and_save(
        name,
        search_RL :List[SeRes],
        paspa :PaSpa,
        np_smooth :int,
        new_SR :SeRes or List[SeRes]=   None,
        hpmser_FD :str=                 None):

    # add new
    if new_SR:
        if type(new_SR) is not list: new_SR = [new_SR]
        search_RL += new_SR

    avg_dst = _smooth_and_sort(
        search_RL=  search_RL,
        paspa=      paspa,
        np_smooth=  np_smooth)

    if hpmser_FD:

        # backup copy previous
        old_res = r_pickle(f'{hpmser_FD}/{name}/{name}_results.srl')
        if old_res: w_pickle(old_res, f'{hpmser_FD}/{name}/{name}_results.srl.backup')

        w_pickle((search_RL, paspa), f'{hpmser_FD}/{name}/{name}_results.srl')
        _write_graph(
            name=       name,
            search_RL=  search_RL,
            hpmser_FD=  hpmser_FD)

    return search_RL, avg_dst

# string from list of weights
def _str_weights(all_w :list):
    ws = '['
    if len(all_w) > 5:
        for w in all_w[:3]: ws += f'{w:.4f} '
        ws += '.. '
        for w in all_w[-2:]: ws += f'{w:.4f} '
    else:
        for w in all_w: ws += f'{w:.4f} '
    return f'{ws[:-1]}]'

# returns sample with policy and estimated score
def _get_opt_sample(
        paspa :PaSpa,
        search_RL :List[SeRes], # ...should be smoothed and sorted!
        np_smooth :int,
        prob_opt,               # probability of optimized sample
        n_opt,                  # number of optimized samples
        prob_top,               # probability of sample from area of top
        n_top,                  # number of top samples
        avg_dst):               # distance for sample from area of top

    prob_rnd = 1 - prob_opt - prob_top
    if random.random() < prob_rnd or len(search_RL) < 10:
        sample = paspa.sample_point() # rnd sample
    else:
        if random.random() < prob_top/(prob_top+prob_opt):
            n_top += 1 # last for reference
            if n_top > len(search_RL): n_top = len(search_RL)
            points = [paspa.sample_point(search_RL[ix].point, ax_dst=avg_dst) for ix in range(n_top)] # top points
        else:
            points = [paspa.sample_point() for _ in range(n_opt+1)] # opt points (last for reference)

        scores = [_smooth_score(p, search_RL, paspa, np_smooth)[0] for p in points]

        all_pw = list(zip(points, scores))
        all_pw.sort(key=lambda x: x[1], reverse=True)
        maxs = all_pw[0][1]
        subs = all_pw.pop(-1)[1]
        mins = all_pw[-1][1]

        all_p, all_w = zip(*all_pw)
        all_w = [w - subs for w in all_w]
        all_p = list(all_p)
        sample = random.choices(all_p, weights=all_w, k=1)[0]
        print(f'   % sampled #{all_p.index(sample)}/{len(all_p)} from: {maxs:.4f}-{mins:.4f} {_str_weights(all_w)}')

    est_score, _, _ =  _smooth_score(sample, search_RL, paspa, np_smooth)

    return sample, est_score

# prepares nice string results
def _nice_results_str(
        name,
        search_RL :List[SeRes],
        paspa :PaSpa,
        n_top=                  20,
        all_nps :int or None=   3):

    re_str = ''
    if all_nps: re_str += f'Search run {name} - {len(search_RL)} results (by smooth_score):\n\n{paspa}\n\n'

    if len(search_RL) < n_top: n_top = len(search_RL)
    for nps in NP_SMOOTH:
        avg_dst = _smooth_and_sort(search_RL, paspa, nps)

        re_str += f'TOP {n_top} results for NPS {nps} (avg_dst:{avg_dst:.3f}):'
        if NP_SMOOTH.index(nps) == 0: re_str += ' /// id smooth [ local] [  max-min  ] avg_dst {params...}\n'
        else: re_str += '\n'

        for srIX in range(n_top):
            sr = search_RL[srIX]
            ss_np, avg_dst, all_scores = _smooth_score(sr.point, search_RL, paspa, nps)
            re_str += f'{sr.id:4d} {ss_np:.4f} [{sr.score:.4f}] [{max(all_scores):.4f}-{min(all_scores):.4f}] {avg_dst:.3f} {paspa.point_2str(sr.point)}\n'

    if all_nps and len(search_RL) > n_top:
        avg_dst = _smooth_and_sort(search_RL, paspa, all_nps)
        re_str += f'\nALL results for NPS {all_nps} (avg_dst:{avg_dst:.3f}):\n'
        for sr in search_RL:
            ss_np, avg_dst, all_scores = _smooth_score(sr.point, search_RL, paspa, all_nps)
            re_str += f'{sr.id:4d} {ss_np:.4f} [{sr.score:.4f}] [{max(all_scores):.4f}-{min(all_scores):.4f}] {avg_dst:.3f} {paspa.point_2str(sr.point)}\n'

    return re_str

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

# loads results, shows graph and saves, prints results
def show_hpmser_resuls(hpmser_FD :str):

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

    _write_graph(name, search_RL, hpmser_FD, silent=False)

    print(f'\n{paspa}')
    print(f'\n{_nice_results_str(name, search_RL, paspa)}')
    return name, search_RL, paspa

# reads/updates config file
def update_config(
        hpmser_FD,
        name,
        np_smooth :int,
        prob_opt :float,
        n_opt :int,
        prob_top :float,
        n_top :int):

    cf_path = f'{hpmser_FD}/{name}/hpms.conf'
    write_file = True

    if os.path.isfile(cf_path):
        with open(cf_path) as cfile:
            vals = cfile.read()
            vals = vals.split()
            new_np_smooth = int(vals[0])
            new_prob_opt =  float(vals[1])
            new_n_opt =     int(vals[2])
            new_prob_top =  float(vals[3])
            new_n_top =     int(vals[4])

        if  new_np_smooth == np_smooth and \
            new_prob_opt  == prob_opt and \
            new_n_opt     == n_opt and \
            new_prob_top  == prob_top and \
            new_n_top     == n_top:
            write_file = False
        else:
            np_smooth = new_np_smooth
            prob_opt = new_prob_opt
            n_opt = new_n_opt
            prob_top = new_prob_top
            n_top = new_n_top

    if write_file:
        with open(cf_path, 'w') as cfile:
            cfile.write(f'{np_smooth} {prob_opt} {n_opt} {prob_top} {n_top}')

    return {
        'np_smooth':    np_smooth,
        'prob_opt':     prob_opt,
        'n_opt':        n_opt,
        'prob_top':     prob_top,
        'n_top':        n_top}

# hpms searching function
@timing
def hpmser(
        func :Callable,                             # function which parameters need to be optimized, has to return score or {'score': score}
        psd :dict,                                  # function params space for search
        func_defaults :dict=        None,           # function defaults
        name :str=                  None,           # for None stamp will be used
        add_stamp=                  True,           # adds short stamp to name, when name given
            # sampling process parameters
        np_smooth :int=             3,              # number of points used for smoothing
        prob_opt=                   0.5,            # probability of sampling from estimated space (optimized)
        n_opt=                      50,             # number of points taken from estimated space
        prob_top=                   0.0,            # probability of sampling from area of top points
        n_top=                      20,             # number of points taken from area of top points
        use_config=                 True,           # uses config file to set/log settings
            # process envy options
        devices=                    None,           # devices to use for search
        use_all_cores=              True,           # True: when devices is None >> uses all cores, otherwise as set by devices
        subprocess=                 True,           # True: runs func in subprocesses, otherwise in this process
        n_loops=                    None,           # limit for number of search loops
        hpmser_FD : str or bool=    None,           # folder, where save search results and html, for None does not save, for True uses default
        top_show_freq=              20,
        verb=                       1):

    # manage hpmser_FD, create if needed
    if hpmser_FD is True: hpmser_FD = 'hpmser' # default for True
    if not os.path.isdir(hpmser_FD): os.mkdir(hpmser_FD)

    # defaults
    if not name: name = stamp()
    elif add_stamp: name = f'{stamp(letters=0)}_{name}'
    search_RL = []
    paspa = None

    # check for continuation
    results_FDL = sorted(os.listdir(hpmser_FD))
    if len(results_FDL):
        print(f'\nThere are {len(results_FDL)} searches in hpmser_FD, do you want to continue with the last one ({results_FDL[-1]}) ..waiting 10 sec (y/n, n-default)?')
        i, o, e = select.select([sys.stdin], [], [], 10)
        if i and sys.stdin.readline().strip() == 'y':
            name = results_FDL[-1]  # take last
            try:    search_RL, paspa = r_pickle(f'{hpmser_FD}/{name}/{name}_results.srl')
            except: search_RL, paspa = r_pickle(f'{hpmser_FD}/{name}/{name}_results.srl.backup')
            _smooth_and_sort(search_RL, paspa, np_smooth)

    subfolder = f'{hpmser_FD}/{name}'
    if not os.path.isdir(subfolder): os.mkdir(subfolder)

    nestarter(log_folder=subfolder, custom_name=name, devices=False)

    if verb > 0:
        print(f'\n*** hpmser *** {name} started for: {func.__name__}, conf: {np_smooth} {prob_opt:.1f} {n_opt} {prob_top:.1f} {n_top}')
        if search_RL: print(f' search will continue with {len(search_RL)} results...')

    if not paspa:
        paspa = PaSpa(
            psd=    psd,
            verb=   verb-1)
    if verb>0: print(f'\n{paspa}\n')

    # manage devices
    if not subprocess: use_all_cores = False
    if devices is None: devices = [None] * (cpu_count() if use_all_cores else 1) # manage case of None for devices
    assert subprocess or (not subprocess and len(devices)==1), 'ERR: cannot use many devices without subprocess'

    loop_func = interface_wrap_MP_H if subprocess else interface_wrap_H

    # manage func_defaults, remove psd keys from func_defaults
    if not func_defaults: func_defaults = {}
    for k in psd:
        if k in func_defaults: func_defaults.pop(k)

    cpa, cpb = paspa.sample_corners()
    avg_dst = 1

    cr_ID = len(search_RL)
    sample_num = cr_ID

    max_run_ID = None if not cr_ID else search_RL[0].id
    prev_max_run_ID = None
    try:
        while True:

            if use_config:
                config = update_config(hpmser_FD, name, np_smooth, prob_opt, n_opt, prob_top, n_top)
                np_smooth = config['np_smooth']
                prob_opt =  config['prob_opt']
                n_opt =     config['n_opt']
                prob_top =  config['prob_top']
                n_top =     config['n_top']

            # use all available devices
            while devices:
                if verb > 1: print(f' > got {len(devices)} devices at {cr_ID} loop start')
                spoint = None
                est_score = 0
                if sample_num == 0: spoint = cpa # use corner point a
                if sample_num == 1: spoint = cpb # use corner point b
                if sample_num > 1:
                    spoint, est_score = _get_opt_sample(
                        paspa=          paspa,
                        search_RL=      search_RL,
                        np_smooth=      np_smooth,
                        prob_opt=       prob_opt,
                        n_opt=          n_opt,
                        prob_top=       prob_top,
                        n_top=          n_top,
                        avg_dst=        avg_dst)
                sample_num += 1

                loop_func(
                    func=       func,
                    device=     devices.pop(0),
                    spoint=     spoint,
                    est_score=  est_score,
                    s_time=     time.time(),
                    **func_defaults)

            # get from que at least one
            res = wrap_que.get()
            devices.append(res['device'])   # return device
            sr = SeRes(                     # add new point with results
                id=     cr_ID,
                point=  res['spoint'],
                score=  res['score'])

            search_RL, avg_dst = _update_and_save(
                name=       name,
                search_RL=  search_RL,
                paspa=      paspa,
                np_smooth=  np_smooth,
                new_SR=     sr,
                hpmser_FD=  hpmser_FD)

            # gots new MAX
            if search_RL[0].id != max_run_ID:
                prev_max_run_ID = max_run_ID
                max_run_ID = search_RL[0].id

            if verb > 0:
                # current sr report
                max_SR = search_RL[0]
                dif = sr.smooth_score - res['est_score']
                difs = '+' if dif>0 else '-'
                dif = abs(dif)
                difs += f'{dif:.4f}'
                srp = f'{sr.id} {sr.smooth_score:.4f} [{sr.score:.4f} {difs}] {max_SR.id}:{paspa.dist(max_SR.point, sr.point):.3f}'
                srp += f'  avg_dst:{avg_dst:.3f} conf: {np_smooth} {prob_opt:.1f} {n_opt} {prob_top:.1f} {n_top} {int(time.time() - res["s_time"])}s'
                print(srp)

                # new MAX report (last search is a new MAX)
                if max_run_ID == cr_ID:
                    msr = f'{paspa.point_2str(sr.point)}\n'

                    dp = 0
                    if prev_max_run_ID is not None:
                        for sa in search_RL:
                            if sa.id == prev_max_run_ID:
                                dp = paspa.dist(sa.point, sr.point)
                                break

                    msr += f' dst_prev:{dp:.3f}\n'
                    for nps in NP_SMOOTH:
                        ss_np, avd, all_sc = _smooth_score(search_RL[0].point, search_RL, paspa, nps)
                        msr += f'  NPS:{nps} {ss_np:.4f} [{max(all_sc):.4f}-{min(all_sc):.4f}] {avd:.3f}\n'
                    print(msr)

                if top_show_freq and len(search_RL) % top_show_freq == 0:
                    print(_nice_results_str(name, search_RL, paspa, n_top=5, all_nps=None))

            if len(search_RL) == n_loops:
                if verb>0: print(f'...n_loops ({n_loops}) done!')
                break

            cr_ID += 1

    except KeyboardInterrupt:
        if verb>0: print('...interrupted')

    search_RL, avg_dst = _update_and_save(
        name=       name,
        search_RL=  search_RL,
        paspa=      paspa,
        np_smooth=  np_smooth,
        hpmser_FD=  hpmser_FD)

    results = _nice_results_str(name, search_RL, paspa)
    if hpmser_FD:
        with open( f'{hpmser_FD}/{name}/{name}_results.txt', 'w') as file: file.write(results)

    if verb>0: print(results)

    return search_RL

# hpmser example
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
        func_const_kwargs= {'name': 'pio', 'a':3, 'wait': av_time * 2, 'verb': verb - 1},
        devices=    [None]*n_proc,
        #subprocess= False,
        hpmser_FD=  True,
        verb=       verb)


if __name__ == '__main__':

    example_hpmser()