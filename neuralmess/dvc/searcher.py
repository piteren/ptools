"""

 2019 (c) piteren

 DVC searcher
 Searches (Random Grid Searcher) params values for maximum test accuracy

"""

import os
import pandas as pd
import plotly.express as px
import random
import shutil
import time

from ptools.lipytools.stats import msmx
from ptools.lipytools.little_methods import w_pickle, stamp
from ptools.neuralmess.dev_manager import get_available_cuda_id
from ptools.pms.paspa import PaSpa
from ptools.pms.paradict import ParaDict
from ptools.mpython.qmp import QueMultiProcessor

from putils.neuralmess.dvc.data import DVCData
from putils.neuralmess.dvc.starter import DVCStarter


# search result
class SeRes:

    def __init__(
            self,
            point: dict,
            id: int,
            ts_acc: list=           None,
            ts_type: list=          None,
            ts_acc_smooth: float=   None):

        self.point = point
        self.id = id
        self.ts_acc = ts_acc
        self.ts_acc_avg = None
        self.ts_acc_std = None
        if self.ts_acc is not None:
            sval = msmx(self.ts_acc)
            self.ts_acc_avg = sval['mean']
            self.ts_acc_std = sval['std']
        self.ts_type = ts_type
        self.ts_acc_smooth = ts_acc_smooth

# calculates ts_acc_smooth for given list of SRes
def smooth_RL(
        search_RL: list,        # list of SRes
        paspa :PaSpa,           # parameters space
        rad: float=     0.5):   # smoothing radius

    # returns ts_acc_smooth for given point in relation to search_RL
    def smooth_val(point :dict):

        tot_weight = 0
        s_val = 0
        for zix in range(len(search_RL)):
            dst = paspa.dist(point, search_RL[zix].point)
            weight = (rad-dst) / rad
            if weight < 0: weight = 0
            s_val += search_RL[zix].ts_acc_avg * weight
            tot_weight += weight

        if tot_weight > 0: s_val /= tot_weight
        # no neighbour points >> average of all
        else:
            accs = [sr.ts_acc_avg for sr in search_RL]
            s_val = sum(accs) / len(accs)
        return s_val

    for six in range(len(search_RL)):
        search_RL[six].ts_acc_smooth = smooth_val(search_RL[six].point)

# writes results and space to pickle file
def write_results(
        search_RL,
        paspa,
        name):
    w_pickle((search_RL, paspa), '_search/%s.results'%name)

# writes 3D graph to html with plotly
def write_graph(
        smth_RL,
        name,
        silent= True):

    p_keys = sorted(list(smth_RL[0].point.keys()))
    values = ['ts_acc','ts_acc_smooth']
    columns = p_keys + values
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

    valLL = [[res.point[key] for key in columns[:-2]] + [res.ts_acc_avg,res.ts_acc_smooth] for res in smth_RL]
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
        title=          '%s %d results'%(name,len(df[columns[0]])),
        x=              axes_data['x'],
        y=              axes_data['y'],
        z=              axes_data['z'],
        color=          axes_data['c'],
        range_color=    [cr_min,cr_max],
        opacity=        0.7,
        width=          700,
        height=         700)

    file = '_search/%s.results.html'%name
    fig.write_html(file, auto_open=False if os.path.isfile(file) else True)

# prints search report
def report_search(
        search_RL :list,
        name,
        limit :int=     None): # limit to N lines

    print('\nSearch run %s finished, results by ts_acc_smooth:' % name)
    ix = 0
    for sr in sorted(search_RL, key=lambda x: x.ts_acc_smooth, reverse=True):
        if not limit or limit and ix<limit:
            print(f'{sr.ts_acc_smooth:.5f}/{sr.ts_acc_avg:.5f} ({sr.id:3d}) {PaSpa.point_2str(sr.point)}')
        ix += 1

    ts_acc_avgs = []
    for sr in search_RL: ts_acc_avgs.append(sr.ts_acc_avg)
    print(' > avg ts_acc %.3f' % (sum(ts_acc_avgs) / len(ts_acc_avgs)))

# single run (process) function
def proc_func(task :dict):

    starter = DVCStarter(
        dvc_dict=       task['mdict'],
        dvc_data=       task['dvc_data'],
        dvc_dd=         task['dvc_dd'],
        name_timestamp= False,
        devices=        task['device'],
        seed=           task['seed'],
        rand_batcher=   task['rand_batcher'],
        save_TFD=       task['root_FD'],
        do_mlog=        False, # do not need logging per model
        do_TX=          False,
        do_TB=          False,
        verb=           0) # do not need verbosity of Starter @RGS

    train_results = starter.train(
        n_batches=      task['n_batches'],
        m_seed=         task['m_seed'],
        fq_VL=          task['fq_VL'],
        fq_AVG=         0,
        fqM_TB=         0,
        fqM_HTB=        0,
        save_max_VL=    True)

    return train_results, task['rgs_dict_sample'], task['sTime'], task['device']


class DVCSearcher:

    def __init__(
            self,
            dvc_dict: dict or ParaDict,
            dvc_data: DVCData,
            dvc_dd: dict=           None,
            devices=                -1,     # do not put masked devices here
            seed=                   12321,  # seed for NNModel
            rand_batcher=           True,
            root_FD=                '.',
            verb=                   1):

        self.verb = verb
        self.name = 'rgs.%s.%s'%(dvc_dict['name'],stamp(date=False))
        self.seed_counter = seed
        if self.verb > 0: print('\n*** DVCSearcher *** %s initializes, seed %d ...' % (self.name, self.seed_counter))

        self.devices = devices
        # searcher should not mask devices in main process but give unmasked ids to subprocesses >> need to resolve case of -1 or []
        if self.devices==-1 or (type(self.devices) is list and not self.devices):
            avd = get_available_cuda_id()
            if self.devices==-1: self.devices = avd[-1]
            else:                self.devices = avd
        if type(self.devices) is not list: self.devices = [self.devices] # to list

        n_proc = len(self.devices) # num of processes inferred from devices
        if self.verb > 1: print(' > devices to use(%d): %s' % (n_proc,self.devices))

        self.rand_batcher = rand_batcher

        self.mdict = dvc_dict
        if type(self.mdict) is not ParaDict:
            self.mdict = ParaDict(name='mdict', dct=dvc_presets['dvc_base'], verb=self.verb-1)
            dvc_dict['verb'] = 0  # override model verb
            self.mdict.refresh(dvc_dict)

        self.dvc_data = dvc_data
        self.dvc_dd = dvc_dd

        self.paspa = PaSpa(self.mdict['rgs'], seed=None, verb=self.verb - 1) # set seed as None for better parallel runs
        if self.verb > 0: print('PaSpa (parameters space):\n%s' % self.paspa.info_string())

        self.root_FD = root_FD

        self.queMP = QueMultiProcessor(
            proc_func=  proc_func,
            n_proc=     n_proc,
            reload=     None,
            user_tasks= True,
            verb=       self.verb) if n_proc > 1 else None

    # returns sample closer to local maximum
    def __get_opt_sample(
            self,
            smth_RL=            None,
            rad=                None,
            space_prob :float=  0.5): # exploration factor

        max_point = None
        if random.random() > space_prob and smth_RL:
            mix = 0
            mval = 0
            for ix in range(len(smth_RL)):
                if smth_RL[ix].ts_acc_smooth > mval:
                    mval = smth_RL[ix].ts_acc_smooth
                    mix = ix
            max_point = smth_RL[mix].point

            rad = random.random()*rad

        return self.paspa.sample_point(max_point, rad)

    # starts searching loop
    def search(
            self,
            n_batches=          1000,
            m_seed :int =       1,
            fq_VL=              50,
            n_loops :int=       False,  # limits searching loops to N
            rad :float=         0.5,    # radius for smoothing and sampling
            space_prob :float=  0.5):   # probability of sampling whole space (exploration)

        if self.verb > 0: print('DVCSearcher starts searching (mseed %d, %d batches, rad %.1f, space %.1f)'%(m_seed,n_batches,rad,space_prob))

        search_RL = [] # results list
        free_device = [] + self.devices
        try:
            run = 0
            result_ix = 0
            max_SR = None
            while True:

                sTime = time.time()
                if n_loops and run == n_loops: break
                run += 1

                rgs_dict_sample = self.__get_opt_sample(search_RL, rad, space_prob)
                self.mdict.refresh(rgs_dict_sample)
                self.mdict['name'] = '%s.%d'%(self.name,run)

                task = {
                    'm_seed':           m_seed,
                    'dvc_data':         self.dvc_data if not self.dvc_dd else None,
                    'dvc_dd':           self.dvc_dd,
                    'mdict':            self.mdict.get_dict_copy(),
                    'seed':             self.seed_counter,
                    'rand_batcher':     self.rand_batcher,
                    'root_FD':          self.root_FD,
                    'device':           free_device.pop(0),
                    'n_batches':        n_batches,
                    'fq_VL':            fq_VL,
                    'rgs_dict_sample':  rgs_dict_sample,
                    'sTime':            sTime}
                self.seed_counter += m_seed

                if self.queMP: self.queMP.putTask(task)

                tr_pack = None
                if not free_device:
                    if self.queMP:  tr_pack = self.queMP.getResult()
                    else:           tr_pack = proc_func(task)
                    free_device.append(tr_pack[3])

                if tr_pack:

                    train_results =     tr_pack[0]
                    rgs_dict_sample =   tr_pack[1]
                    sTime =             tr_pack[2]

                    new_SR = SeRes(
                        point=          rgs_dict_sample,
                        id=             result_ix,
                        ts_acc=         train_results['ts_acc'],
                        ts_type=        train_results['ts_type'])
                    search_RL.append(new_SR)
                    smooth_RL(search_RL, self.paspa, rad)  # update smooth
                    search_RL = sorted(search_RL, key=lambda x: x.ts_acc_smooth, reverse=True)  # sort

                    if self.verb > 0:
                        ts_type = new_SR.ts_type[0]
                        std_info = ''
                        if m_seed > 1:
                            ts_type = 'm_seed'
                            std_info = ' ts_acc_std %.1f' % new_SR.ts_acc_std
                        print('R:%d %.2f(%s) '%(new_SR.id, new_SR.ts_acc_avg, ts_type), end='')
                        if max_SR: print('[%.2f] %d (max: %6.3f/%6.3f) '%(self.paspa.dist(max_SR.point,new_SR.point), max_SR.id, max_SR.ts_acc_smooth, max_SR.ts_acc_avg), end='')
                        print('%s (%ds)%s' % (PaSpa.point_2str(new_SR.point), time.time() - sTime, std_info))

                    write_results(search_RL, self.paspa, self.name)
                    write_graph(search_RL, self.name)
                    result_ix += 1
                    max_SR = search_RL[0]

        except KeyboardInterrupt:
            if self.verb > 0: print('...interrupted')

        write_results(search_RL, self.paspa, self.name)

        # clear remaining folders
        m_folder = self.root_FD + '/_models'
        dirs = os.listdir(m_folder)
        rem_dirs = [dir for dir in dirs if self.name in dir]
        for dir in rem_dirs: shutil.rmtree(m_folder + '/' + dir)
        if self.verb > 1: print('removed %d remaining folders' % len(rem_dirs))

        if self.verb > 0: report_search(search_RL,self.name)

        return search_RL


# sample running code
if __name__ == '__main__':

    from putils.neuralmess.dvc.data import UDD
    from putils.neuralmess.dvc.presets import dvc_presets

    udd = UDD(
        TRsen=      (['This is sentence A.','This is sentence B']),
        TRlbl=      [0,1])
    dvc_data = DVCData(
        uDD=        udd,
        vl_split=   0,
        ts_split=   0,
        use=        True,
        verb=       1)

    searcher = DVCSearcher(
        dvc_data=       dvc_data,
        dvc_dict=       dvc_presets['dvc_base'])
    search_RL = searcher.search(
        n_batches=      100,
        m_seed=         3)