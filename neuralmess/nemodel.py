"""

 2019 (c) piteren

 NEModel class: wraps graph building function (fwd_func) with many features

    fwd_func:
        - should be build complete model forward graph (FWD) - from PH (placeholders) to loss
        - function should return dict with PH, tensors, variables lists, dict keys should meet naming policy
            - if there is a 'loss' tensor returned >> OPT part may be build
            - dict may contain variables_lists and single variable under keys with 'var' in name
                - list of special keys to consider while building fwd_func is under SPEC_KEYS
                - variables returned under 'train_vars' key are optimized (if 'train_vars' key is not present all trainable vars are optimized)
                - sub-lists of variables will serve for separate savers (saved in subfolders)

    Model building params and their arguments (p&a) may come from (in order of overriding):
        - NEModel __init__ p&a
        - mdict.dct p&a (when saved in model folder)
        - fwd_func p&a (defaults)
        - mdict (for fwd_func, given to NEModel init)

    - keeps updated model_building_params under self (dict) keys
    - tensors, placeholders, etc... returned by model function are also kept under self (dict) keys
    - model objects (graph, session, saver ...) are kept as self fields

 NEModel class implements:
    - one folder for all model data (subfolder of savePath named with model name)
    - logger (txt file saved into the model folder)
    - GPU management with multi-GPU training (placement control of model graph elements across devices)
    - builds optimization (OPT) graph part
        - calculates gradients for every tower >> averages them
        - AVT gradient clipping and scaled LR (warmUp, annealing)
    - MultiSaver (with option for saving sub-lists of variables into separate checkpoints)
    - sanity check of many graph elements and dependencies
    - inits session, TB writer, MultiSaver loads variables (or inits them)
    - returns (updated) dict with all model parameters, tensors, PH, session, savers ...
"""

import numpy as np
import os
import shutil
import tensorflow as tf

from ptools.lipytools.logger import set_logger
from ptools.lipytools.little_methods import get_defaults, short_scin, r_pickle, w_pickle, stamp
from ptools.pms.paradict import ParaDict
from ptools.neuralmess.base_elements import num_var_floats, lr_scaler, gc_loss_reductor, log_vars
from ptools.neuralmess.dev_manager import tf_devices
from ptools.neuralmess.multi_saver import MultiSaver

# restricted keys for fwd_func mdict (model params-arguments dict, if they appear in mdict, should be named exactly like below)
SPEC_KEYS = [
    'name',                                             # model name
    'seed',                                             # seed for TF nad numpy
    'iLR',                                              # initial learning rate
    'warm_up','ann_base','ann_step','n_wup_off',        # LR management (parameters of LR warmup and annealing)
    'avt_SVal','avt_window','avt_max_upd','do_clip',    # gradients clipping parameters
    'train_vars',                                       # list of variables to train (may be returned, otherwise all trainable are taken)
    'opt_class',                                        # optimizer class
    'batch_size',                                       # batch size
    'verb']                                             # fwd_func verbosity


class NEModel(dict):

    def __init__(
            self,
            fwd_func,                               # function building forward graph (from PH to loss)
            mdict :dict,                            # model parameters-arguments dictionary
            # TODO: I'm not sure whether session can be given here, since we create tf.Graph for model below (graph should be given to session...)
            # I will lock this feature...
            # session :tf.Session=        None,       # session to use
            devices=                    -1,         # check neuralmess.dev_manager.ft_devices for details
            do_opt :bool=               True,       # add optimization part to the graph (for training)
                # values below complement mdict
            name=                       'NEM',
            name_timestamp=             False,      # adds timestamp to name
            seed=                       12321,
            opt_class=                  None,       # opt_class None uses default (Adam), examples: tf.train.GradientDescentOptimizer, partial(tf.train.AdamOptimizer, beta1=0.7, beta2=0.7)
            iLR=                        1e-3,
            warm_up=                    None,
            ann_base=                   None,
            ann_step=                   1,
            n_wup_off :float=           1,
            avt_SVal=                   1,
            avt_window=                 100,
            avt_max_upd=                1.5,
            do_clip=                    False,
                # save
            save_TFD :str=              '_models',  # top folder of model save
            savers_names :tuple=        (None,),    # names of savers for MultiSaver
            load_saver : bool or str=   True,       # for None do not loads, for True loads default
                # GPU management
            sep_device=                 True,       # separate first device for variables, gradients_avg, optimizer (otherwise those ar placed on the first FWD calculations tower)
            collocate_GWO=              False,      # collocates gradient calculations with tf.OPs (gradients are calculated on every tower with its operations, but remember that vars are on one device...) (otherwise with first FWD calculations tower)
                # other
            do_log=                     True,       # enables saving log file in folder of NEModel
            verb :int=                  0):         # verb of NEModel (object/constructor), fwd_func has own verb in mdict

        super().__init__() # init self as dict
        self.verb = verb
        if self.verb > 0: print('\n*** NEModel *** initializes...')

        self_args_dict = { # dict with params from constructor
            'name':         name,
            'seed':         seed,
            'opt_class':    opt_class,
            'iLR':          iLR,
            'warm_up':      warm_up,
            'ann_base':     ann_base,
            'ann_step':     ann_step,
            'n_wup_off':    n_wup_off,
            'avt_SVal':     avt_SVal,
            'avt_window':   avt_window,
            'avt_max_upd':  avt_max_upd,
            'do_clip':      do_clip}

        fwdf_mdict = get_defaults(function=fwd_func) # defaults of fwd_func

        # resolve model name (extend with timestamp when needed)
        resolved_name =                                 self_args_dict['name']
        if 'name' in fwdf_mdict:    resolved_name =     fwdf_mdict['name']
        if 'name' in mdict:         resolved_name =     mdict['name']
        if name_timestamp:          resolved_name +=    '.' + stamp()
        if self.verb > 0: print(' > NEModel name: %s'%resolved_name)

        # model folder and logger
        self.model_FD = save_TFD + '/' + resolved_name
        if not os.path.isdir(self.model_FD): os.mkdir(self.model_FD)
        if do_log: set_logger(logFD=self.model_FD, custom_name=resolved_name, verb=self.verb)

        # read mdict file from folder (last saved model parameters)
        self.md_file = self.model_FD + '/mdict.dct'
        file_mdict = r_pickle(self.md_file)
        if not file_mdict: file_mdict = {}
        elif self.verb > 0: print(' > loaded model dict from file: %s' % self.md_file)

        self.mdict = ParaDict(fwdf_mdict)   # ParaDict with defaults of fwd_func
        self.mdict.update(file_mdict)       # update(override) with file dict
        self.mdict.add_new(self_args_dict)  # add new from self_args_dict (extends)
        self.mdict.update(mdict)            # update with mdict
        self.mdict['name'] = resolved_name  # finally override name
        if do_opt and not self.mdict['opt_class']: self.mdict['opt_class'] = tf.train.AdamOptimizer # default optimizer

        self.mdict.check_params_sim(SPEC_KEYS)  # safety check
        self.update(self.mdict)  # finally update self with all model building params

        # save dict files (in train mode)
        if do_opt:
            w_pickle(self.mdict, self.md_file)  # write dict of params
            md_file_txt = self.md_file[:-3]+'txt'
            if os.path.isfile(md_file_txt):
                shutil.copy(md_file_txt,md_file_txt+'_OLD')
            self.mdict.write_txtfile(md_file_txt)  # write dict to txt
            if self.verb > 0: print(' > params dictionary of %s NEModel saved to .dct and .txt files'%self.mdict['name'])

        devices = tf_devices(devices, verb=self.verb)

        # report devices
        if self.verb > 0:
            print()
            if len(devices)==1:
                if 'CPU' in devices[0]: print('NEModel builds CPU device setup')
                else:                   print('NEModel builds single-GPU setup')
            else:                       print('NEModel builds multi-dev setup for %d devices'%len(devices))

        if len(devices)<3: sep_device = False # SEP is available for 3 or more devices

        # build FWD graph(s) >> manage variables >> build OPT graph
        self.gFWD = [] # list of dicts of all FWD graphs (from all devices)
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self['seed']) # set graph seed
            np.random.seed(self['seed'])
            if self.verb > 0: print('\nNEModel set TF & NP seed to %d' % self['seed'])

            # builds graph @SEP, this graph wont be run, it is only needed to place variables, if not vars_sep >> variables will be placed with first tower
            if sep_device:
                if self.verb > 0: print('\nNEModel places %s VARs on %s...' % (self['name'], devices[0]))
                with tf.device(devices[0]):
                    fwd_func(**self)

            tower_devices = [] + devices
            if sep_device: tower_devices = tower_devices[1:] # trim SEP
            for dev in tower_devices:
                if self.verb > 0: print('\nNEModel builds FWD graph of %s model @device: %s' % (self['name'], dev))
                with tf.device(dev):
                    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                        self.gFWD.append(fwd_func(**self))

            self.update(self.gFWD[0]) # update self with dictionary returned by fwd_func

            # get FWD variables returned by fwd_func (4 saver)
            train_vars = [] # variables to train
            saver_vars = {} # dict of variables to save
            for key in self.keys():
                if 'var' in key.lower():
                    if key =='train_vars':
                        train_vars = self[key]
                        if type(train_vars) is not list: train_vars = [train_vars]
                    else:
                        if type(self[key]) is not list: saver_vars[key] = [self[key]]
                        else:                           saver_vars[key] = self[key]

            all_vars = tf.global_variables()

            # there are returned variables >> assert there are all variables returned in lists
            if saver_vars:
                all_vars_returned = []
                for key in saver_vars: all_vars_returned += saver_vars[key]
                there_are_all = True
                for var in all_vars:
                    if var not in all_vars_returned:
                        print(' *** variable %s not returned by fwd_func'%var.name)
                        there_are_all = False
                assert there_are_all, 'ERR: there are some variables not returned by fwd_func in lists!'

            else: saver_vars['fwd_vars'] = all_vars # put all

            if self.verb > 0:
                print('\nNEModel variables to save from fwd_func:')
                for key in sorted(list(saver_vars.keys())):
                    varList = saver_vars[key]
                    if varList: print(' ### vars @%s - num: %d, floats: %s (%s)' % (key, len(varList), short_scin(num_var_floats(varList)), varList[0].device))
                    else: print(' ### no vars')
                    if self.verb > 1: log_vars(varList)

            if 'loss' not in self:
                do_opt = False
                if self.verb > 0: print('\nthere is no loss in FWD graph, OPT graph wont be build')

            if not do_opt:
                if self.verb > 0: print('\nOPT graph wont be build')
            # build optimization graph
            else:
                if self.verb > 0: print('\nPreparing OPT part with %s' % self['opt_class'])
                # select trainable variables for OPT
                all_tvars = tf.trainable_variables()
                if train_vars:
                    # check if all train_vars are trainable:
                    for var in train_vars:
                        if var not in all_tvars:
                            if self.verb > 0: print('variable %s is not trainable but is in train_vars, please check the graph!' % var.name)
                else:
                    for key in saver_vars:
                        for var in saver_vars[key]:
                            if var in all_tvars:
                                train_vars.append(var)
                    assert train_vars, 'ERR: there are no trainable variables at the graph!'
                # log train_vars
                if self.verb > 0:
                    print('\nNEModel trainable variables:')
                    print(' ### train_vars: %d floats: %s' % (len(train_vars), short_scin(num_var_floats(train_vars))))
                    if self.verb > 1: log_vars(train_vars)

                # build gradients for towers
                for ix in range(len(self.gFWD)):
                    tower = self.gFWD[ix]
                    tower['gradients'] = tf.gradients(
                        ys=                             tower['loss'],
                        xs=                             train_vars,
                        colocate_gradients_with_ops=    not collocate_GWO) # TF default is False >> calculates gradients where OPS, for True >> where train_vars
                    # log gradients
                    if self.verb > 0:
                        nGrad = len(tower['gradients'])
                        print(' > gradients for %d tower got %d tensors (%s)' %(ix,nGrad,tower['gradients'][0].device))
                        if self.verb > 1:
                            print('NEModel variables and their gradients:')
                            for gix in range(len(tower['gradients'])):
                                grad = tower['gradients'][gix]
                                var = train_vars[gix]
                                print(var, var.device)
                                print(' > %s'%grad) # grad as a tensor displays device when printed (unless colocated with OP!)
                self['gradients'] = self.gFWD[0]['gradients']

                # None @gradients check
                none_grads = 0
                for grad in self['gradients']:
                    if grad is None: none_grads += 1
                if none_grads and self.verb > 0:
                    print('There are None gradients: %d/%d, some trainVars may be unrelated to loss, please check the graph!'%(none_grads,len(self['gradients'])))

                # average gradients
                if len(devices) > 1:

                    if self.verb > 0: print('\nNEModel builds gradients averaging graph with device %s for %d towers' % (devices[0], len(self.gFWD)))
                    with tf.device(devices[0]):

                        towerGrads = [tower['gradients'] for tower in self.gFWD]
                        avgGrads = []
                        for mGrads in zip(*towerGrads):
                            grads = []
                            for grad in mGrads:
                                if grad is not None: # None for variables not used while training now...
                                    expandedG = tf.expand_dims(input=grad, axis=-1)
                                    grads.append(expandedG)
                            if grads:
                                grad = tf.concat(values=grads, axis=-1)
                                grad = tf.reduce_mean(input_tensor=grad, axis=-1)
                                avgGrads.append(grad)
                            else: avgGrads.append(None)

                        self['gradients'] = avgGrads # update with averaged gradients
                        if self.verb > 0: print(' > NEModel averaged gradients (%s)' % self['gradients'][0].device)

                # build OPT graph
                with tf.variable_scope('OPT', reuse=tf.AUTO_REUSE):

                    if self.verb > 0: print('\nBuilding OPT graph for %s model @device: %s' % (self['name'], devices[0]))
                    with tf.device(devices[0]):

                        self['g_step'] = tf.get_variable(  # global step
                            name=           'g_step',
                            shape=          [],
                            trainable=      False,
                            initializer=    tf.constant_initializer(0),
                            dtype=          tf.int32)

                        self['scaled_LR'] = lr_scaler(
                            iLR=            self['iLR'],
                            g_step=         self['g_step'],
                            warm_up=        self['warm_up'],
                            ann_base=       self['ann_base'],
                            ann_step=       self['ann_step'],
                            n_wup_off=      self['n_wup_off'],
                            verb=           self.verb)['scaled_LR']

                        # updates with: optimizer, gg_norm, avt_gg_norm
                        self.update(gc_loss_reductor(
                            optimizer=      self['opt_class'](learning_rate=self['scaled_LR']),
                            vars=           train_vars,
                            g_step=         self['g_step'],
                            gradients=      self['gradients'],
                            avt_SVal=       self['avt_SVal'],
                            avt_window=     self['avt_window'],
                            avt_max_upd=    self['avt_max_upd'],
                            do_clip=        self['do_clip'],
                            verb=           self.verb))

                        # select OPT vars
                        saver_vars['opt_vars'] = tf.global_variables(scope=tf.get_variable_scope().name)
                        if self.verb > 0:
                            print(' ### opt_vars: %d floats: %s (%s)' % (len(saver_vars['opt_vars']), short_scin(num_var_floats(saver_vars['opt_vars'])), saver_vars['opt_vars'][0].device))
                            if self.verb > 1: log_vars(saver_vars['opt_vars'])

        # TODO: since disabled passing session to init... (check line 71)
        # self.session = session
        # if not self.session: # create
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(
            graph=  self.graph,
            config= config)

        # remove keys with no variables (corner case, for proper saver)
        sKeys = list(saver_vars.keys())
        for key in sKeys:
            if not saver_vars[key]: saver_vars.pop(key)
        # add saver and load
        self.saver = MultiSaver(
            model_name= self['name'],
            vars=       saver_vars,
            root_FD=    save_TFD,
            savers=     savers_names,
            session=    self.session,
            verb=       self.verb)
        if load_saver:
            if type(load_saver) is bool: load_saver=None
            self.saver.load(saver=load_saver)

        self.summ_writer = tf.summary.FileWriter(
            logdir=         self.model_FD,
            #graph=          self.graph, # you can call add_graph() later
            flush_secs=     10)

        if self.verb > 0: print('%s (NEModel) build finished!'%self['name'])
        if self.verb > 2: print(self)

    def __str__(self): return ParaDict.dict_2str(self)
