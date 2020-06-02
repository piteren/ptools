"""

 2020 (c) piteren

    hpmser - hyperparameters searching function
        > searches hyperparameters space to MAXIMIZE the SCORE for func

        MAXIMIZE the SCORE == find cluster with:
         - high smooth_score of center
         - high lowest smooth_score
         - small dst
         - high num of points

         policy of sampling the space is crucial, it determines the speed, top result and convergence of the hpmser

    parameters:

        func - input function:
            > may have optional ‘device’ or ‘devices’ parameter - for NN, to run on given CUDA device
            > returns: a dict with ‘score’ key or a single value (score)
        psd - dictionary with parameters space to search in, check PaSpa @putils.neuralmess.params_dict
        def_kwargs - dict with other parameters of func
        devices - devices to use by hpmser, (syntax: check @ptools.neuralmess.dev_manager)

"""

#TODO: add TB to hpmser

from multiprocessing import cpu_count
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
        self.smooth_score = score # default, to be updated


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
        dst: float):

    tot_weight = 0
    s_val = 0
    for zix in range(len(search_RL)):
        d = paspa.dist(point, search_RL[zix].point)
        weight = (dst-d) / dst
        if weight < 0: weight = 0
        s_val += search_RL[zix].score * weight
        tot_weight += weight

    if tot_weight > 0: s_val /= tot_weight
    else: # no neighbour points >> average of all
        accs = [sr.score for sr in search_RL]
        if accs: s_val = sum(accs) / len(accs)
    return s_val

# smooths search_RL (updates .smooth_score)
def _smooth_RL(
        search_RL: List[SeRes],
        paspa :PaSpa,
        dst: float):

    for six in range(len(search_RL)):
        search_RL[six].smooth_score = _smooth_score(
            point=      search_RL[six].point,
            search_RL=  search_RL,
            paspa=      paspa,
            dst=        dst)

# smooths & sorts
def _smooth_and_sort(
        search_RL :List[SeRes],
        paspa :PaSpa,
        dst :float):
    _smooth_RL(
        search_RL=  search_RL,
        paspa=      paspa,
        dst=        dst)
    search_RL.sort(key=lambda x: x.smooth_score, reverse=True)

# updates list, smooths, sorts, writes
def _update_and_save(
        name,
        search_RL :List[SeRes],
        paspa :PaSpa,
        dst :float,
        new_SR :SeRes or List[SeRes]=   None,
        hpmser_FD :str=                 None):

    if new_SR:
        if type(new_SR) is not list: new_SR = [new_SR]
        search_RL += new_SR

    _smooth_and_sort(
        search_RL=  search_RL,
        paspa=      paspa,
        dst=        dst)

    if hpmser_FD:
        w_pickle((search_RL, paspa), f'{hpmser_FD}/{name}/{name}_results.srl')
        _write_graph(
            name=       name,
            search_RL=  search_RL,
            hpmser_FD=  hpmser_FD)
    return search_RL

# returns sample with policy and estimated score
def _get_opt_sample(
        paspa :PaSpa,
        search_RL :List[SeRes], # ...should be smoothed and sorted!
        clusters,               # list of clusters or None
        dst,                    # distance of close points
        prob_cl):               # probability of sampling from clusters

    # nice string from cl_weights
    def str_clw(clw):
        sclw = ''
        if len(clw)>10:
            for w in clw[:5]:
                sclw += f'{w:.4f} '
            sclw += '.. '
            for w in clw[-5:]:
                sclw += f'{w:.4f} '
        else:
            for w in clw:
                sclw += f'{w:.4f} '
        return f'[{sclw[:-1]}]'

    ref_point = None
    avg_score = [sr.score for sr in search_RL]
    if not avg_score: avg_score = 0
    else:             avg_score = sum(avg_score) / len(avg_score)

    from_clusters = random.random() < prob_cl

    report = ''
    cl_num = 0
    cl_weights = None
    n_again = 0
    while True:
        if from_clusters:
            if not clusters: clusters = _get_clusters(search_RL, paspa, dst)
            if len(clusters)>2:
                cl_f = dst if dst < 1 else 1 # factor of clusters to sample from
                cl_num = int(cl_f*len(clusters)) # number of top clusters
                if cl_num < 2: cl_num = 2 # at least 2
                if cl_num == len(clusters): cl_num -= 1 # at least last not
                top_clusters = clusters[:cl_num]

                cl_srIX = [cl[0] for cl in top_clusters]
                cl_weights = [cl[1] for cl in top_clusters]
                sub_weight = clusters[cl_num][1] # value of next
                cl_weights = [cw-sub_weight for cw in cl_weights] # subtract next

                cl_sel = random.choices(cl_srIX, weights=cl_weights, k=1)[0]
                ref_point = search_RL[cl_sel].point
        sample = paspa.sample_point(ref_point=ref_point, ax_dst=dst)
        est_score = _smooth_score(sample, search_RL, paspa, dst)
        if est_score >= avg_score: break
        else: n_again += 1

    if cl_num: report += f'   # sampling from {cl_num} clusters: {str_clw(cl_weights)}'
    if n_again: report += f'   @@@ sampling again ({n_again})'
    if report: print(report)

    return sample, est_score

# returns sorted clusters for given arguments [[srIX, smooth_score, [srIX,..]],..]
def _get_clusters(
        search_RL: List[SeRes], # should be sorted
        paspa: PaSpa,
        dst :float):            # distance to classify point as close to another
    clusters = {} # {srIX: (smooth_score, [srIX,..])}
    for srIX in range(len(search_RL)):
        sr = search_RL[srIX]
        in_cluster = False
        for ix in clusters:
            if paspa.dist(sr.point, search_RL[ix].point) <= dst:
                clusters[ix][1].append(srIX)
                in_cluster = True
        if not in_cluster: clusters[srIX] = (sr.smooth_score, [srIX])
    clusters = [[srIX, clusters[srIX][0], clusters[srIX][1]] for srIX in clusters]
    clusters.sort(key= lambda x: x[1], reverse=True)
    return clusters

# prepares nice string results (clusters & points)
def _nice_results_str(
        name,
        search_RL :List[SeRes], # should be sorted
        paspa :PaSpa,
        dst :float,             # distance to classify point as close to another
        n_clusters=     30):
    results = f'Search run {name} - {len(search_RL)} results (dst_smth: {dst})\n\n{paspa}\n'

    clusters = _get_clusters(search_RL, paspa, dst)
    if len(clusters) < n_clusters: n_clusters = len(clusters) // 2

    results += f'\nGot {len(clusters)} clusters, top {n_clusters} clusters:\n'
    results += '  smooth [   local]   id(nicl)   max   min   dif {params...}\n'
    top_srIX = [cl[0] for cl in clusters[:n_clusters]]

    for cIX in range(n_clusters):
        srIX = clusters[cIX][0]
        sr = search_RL[srIX]
        cl_points = clusters[cIX][2]
        scores = [search_RL[sIX].score for sIX in cl_points]
        maxs = max(scores)
        mins = min(scores)
        results += f'{sr.smooth_score:8.5f} [{sr.score:8.5f}] {sr.id:4d}({len(cl_points):4d}) {maxs:.3f} {mins:.3f} {maxs-mins:.3f} {paspa.point_2str(sr.point)}\n'

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
        results += f'{sr.smooth_score:8.5f} [{sr.score:8.5f}] {sr.id:4d}({n_close:4d}) {paspa.point_2str(sr.point)}\n'
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
        dst :float):

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
    paspa = PaSpa(paspa.psd)  # TODO: delete << (legacy ...for old paspa)
    print(f'\n{paspa}')
    _smooth_and_sort(search_RL, paspa, dst=dst)

    _write_graph(name, search_RL, hpmser_FD, silent=False)
    print(f'\n{_nice_results_str(name, search_RL, paspa, dst=dst)}')
    return name, search_RL, paspa

# reads/updates config file
def update_config(
        hpmser_FD,
        name,
        dst :float,
        pcl :float):

    cf_path = f'{hpmser_FD}/{name}/hpms.conf'
    write_file = True

    if os.path.isfile(cf_path):
        with open(cf_path) as cfile:
            vals = cfile.read()
            vals = vals.split()
            new_dst = float(vals[0])
            new_pcl = float(vals[1])

        if new_dst == dst and new_pcl == pcl: write_file = False
        else:
            dst = new_dst
            pcl = new_pcl

    if write_file:
        with open(cf_path, 'w') as cfile: cfile.write(f'{dst} {pcl}')

    return dst, pcl

# hpms searching function
@timing
def hpmser(
        func :Callable,                     # function which parameters need to be optimized
        psd :dict,                          # dictionary defining the space of parameters
        continue_last=              False,  # flag to continue last search from hpmser_FD
        name :str=                  None,   # for None stamp will be used
        add_stamp=                  True,   # adds short stamp to name, when name given
        dst=                        0.5,    # distance of close point (L1N) (sampling, smoothing)
        prob_cl=                    0.5,    # probability of sampling from clusters
        def_kwargs :dict=           None,   # func kwargs
        devices=                    None,   # devices to use for search
        use_all_cores=              True,   # True: when devices is None >> uses all cores, otherwise as set by devices
        subprocess=                 True,   # True: runs func in subprocesses, otherwise in this process
        n_loops=                    None,   # limit for number of search loops
        hpmser_FD : str or bool=    None,   # folder, where save search results and html, for None does not save, for True uses default
        use_config=                 True,   # uses config file to set/log settings
        verb=                       1):

    # manage hpmser_FD, create if needed
    if hpmser_FD is True: hpmser_FD = 'hpmser' # default for True
    if not os.path.isdir(hpmser_FD):
        os.mkdir(hpmser_FD)
        continue_last = False

    search_RL = []
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
            _smooth_and_sort(search_RL, paspa, dst)
    else:
        if not name: name = stamp()
        elif add_stamp: name = f'{stamp(letters=0)}_{name}'

        # create subfolder if needed
        subfolder = f'{hpmser_FD}/{name}'
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)
            continue_last = False

    if verb > 0:
        print(f'\n*** hpmser *** {name} started for: {func.__name__}, dst: {dst}, prob_cl: {prob_cl}')
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

    cpa, cpb = paspa.sample_corners()
    clusters = None

    runIX = len(search_RL)
    sIX = runIX
    max_runIX = None if not runIX else search_RL[0].id
    prev_max_runIX = None
    try:
        while True:

            if use_config: dst, prob_cl = update_config(hpmser_FD, name, dst, prob_cl) # update dst & prob from config

            # use all available devices
            while devices:
                if verb > 1: print(f' > got {len(devices)} devices at {runIX} loop start')
                spoint = None
                est_score = 0
                if sIX == 0: spoint = cpa # use corner point a
                if sIX == 1: spoint = cpb # use corner point b
                if sIX > 1:
                    spoint, est_score = _get_opt_sample(
                        paspa=          paspa,
                        search_RL=      search_RL,
                        clusters=       clusters,
                        dst=            dst,
                        prob_cl=        prob_cl)
                sIX += 1

                loop_func(
                    func=       func,
                    device=     devices.pop(0),
                    spoint=     spoint,
                    est_score=  est_score,
                    s_time=     time.time(),
                    **def_kwargs)

            # get from que at least one
            res = wrap_que.get()
            resIX = runIX
            runIX += 1

            # manage result
            devices.append(res['device'])   # return device
            sr = SeRes(                     # add new point with results
                id=     resIX,
                point=  res['spoint'],
                score=  res['score'])

            search_RL = _update_and_save(
                name=       name,
                search_RL=  search_RL,
                paspa=      paspa,
                dst=        dst,
                new_SR=     sr,
                hpmser_FD=  hpmser_FD)
            clusters = _get_clusters(search_RL, paspa, dst)

            # it is a new max
            if search_RL[0].id == resIX:
                prev_max_runIX = max_runIX
                max_runIX = resIX

            if verb > 0:
                max_SR = search_RL[0]
                dif = sr.smooth_score - res['est_score']
                difs = '+' if dif>0 else '-'
                dif = abs(dif)
                difs += f'{dif:8.5f}'
                srp = f'{sr.smooth_score:8.5f}({difs}) [{sr.score:8.5f}] {sr.id} {max_SR.id}:{paspa.dist(max_SR.point, sr.point):.2f}'
                srp += f' dst:{dst:.2f}, prob_cl:{prob_cl:.2f}'

                # clusters report
                n_cl_points = [len(cl[2]) for cl in clusters]
                avg_n_cl_points = sum(n_cl_points) / len(n_cl_points)
                np = [0 for _ in range(4)]
                nps = 'N123:'
                for n in range(1, 4):
                    for ncp in n_cl_points:
                        if ncp == n:
                            np[n] += 1
                    nps += f'{np[n]},'
                srp += f'  CL: n:{len(clusters)}, points: {min(n_cl_points)} {avg_n_cl_points:.1f} {max(n_cl_points)} {nps[:-1]}'
                clIX = [cl[0] for cl in clusters]
                cl_min_dst = []
                for cax in clIX:
                    clmdst = 1
                    for cbx in clIX:
                        cdst = paspa.dist(search_RL[cax].point, search_RL[cbx].point)
                        if 0 < cdst < clmdst: clmdst = cdst
                    cl_min_dst.append(clmdst)
                avg_cl_min_dst = sum(cl_min_dst) / len(cl_min_dst)
                srp += f' avg_cl_min_dst:{avg_cl_min_dst:.2f}'

                # points report
                min_dst = []
                n_in_dst = []
                for sa in search_RL:
                    mdst = 1
                    ndst = 0
                    for sb in search_RL:
                        cdst = paspa.dist(sa.point, sb.point)
                        if 0 < cdst < mdst: mdst = cdst
                        if cdst <= dst: ndst += 1
                    min_dst.append(mdst)
                    n_in_dst.append(ndst)
                avg_min_dst = sum(min_dst) / len(min_dst)
                avg_n_in_dst = sum(n_in_dst) / len(n_in_dst)
                srp += f'  SR: avg_min_dst:{avg_min_dst:.2f}, avg_n_in_dst:{avg_n_in_dst:.1f} ({int(time.time() - res["s_time"])}s)'
                print(srp)

                # new max
                if max_runIX == resIX:
                    msr = f'{paspa.point_2str(sr.point)}\n'

                    if prev_max_runIX is not None:
                        for sa in search_RL:
                            if sa.id == prev_max_runIX:
                                dp = paspa.dist(sa.point, sr.point)
                                msr += f'dst_prev:{dp:.2f}, '
                                break

                    msr += f'n_points:{len(clusters[0][2])}'
                    scs = [search_RL[sx].score for sx in clusters[0][2]]
                    msr +=f' scores:{max(scs):.4f}-{min(scs):.4f}'
                    print(msr)

            if n_loops and len(search_RL) == n_loops:
                if verb>0: print(f'...n_loops ({n_loops}) done!')
                break

    except KeyboardInterrupt:
        if verb>0: print('...interrupted')

    search_RL = _update_and_save(
        name=       name,
        search_RL=  search_RL,
        paspa=      paspa,
        dst=        dst,
        hpmser_FD=  hpmser_FD)

    results = _nice_results_str(name, search_RL, paspa, dst=dst)
    if hpmser_FD:
        with open( f'{hpmser_FD}/{name}/{name}_results.txt', 'w') as file: file.write(results)

    if verb>0: print(results)

    return search_RL


def example_hpmser_smpl(
        n_samples,
        hpmser_FD,
        dst,
        prob_cl):

    name, search_RL, paspa = show_hpmser_resuls(hpmser_FD, dst=dst)

    num_0 = 0
    sum_n = 0
    avg_dist = 0
    avg_sc = 0
    for ix in range(n_samples):
        spoint, est_score = _get_opt_sample(
            paspa=          paspa,
            search_RL=      search_RL,
            clusters=       None,
            dst=            dst,
            prob_cl=        prob_cl)

        n_close = _num_of_close(spoint, search_RL, paspa, dst)
        dist = paspa.dist(search_RL[0].point, spoint)

        if n_close == 0: num_0 += 1
        sum_n += n_close
        avg_dist += dist
        avg_sc += est_score

        print(f'{ix} point got {n_close} n_close, {dist:4.2f} {est_score:8.5f}')

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
        dst=            0.1,
        prob_cl=        0.5)