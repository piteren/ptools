"""

 2019 (c) piteren

    NN TF graph elements and tools

"""

import numpy as np
import os
import tensorflow as tf

from ptools.lipytools.little_methods import short_scin
from ptools.mpython.mpdecor import proc_wait


# default initializer for variables of graph
def my_initializer(seed=12321):
    #old initializers
    #tf.contrib.layers.xavier_initializer(uniform=False,seed=seed)
    #tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_IN',uniform=False,seed=seed) # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/variance_scaling_initializer
    #tf.random_normal_initializer(stddev=0.01, seed=seed)
    #tf.random_uniform_initializer(minval=-0.01, maxval=0.01, seed=seed)

    return tf.truncated_normal_initializer(stddev=0.02, seed=seed)  # recommended by TF, BERT (0.02?)

# GeLU (Gaussian Error Linear Unit) activation https://arxiv.org/abs/1606.08415
def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

# scales learning rate with warmUp and annealing (after warmUp)
def lr_scaler(
        iLR,                        # initial learning rate
        g_step :tf.Tensor=  None,   # global step tf.variable of tf.int type, for None creates one
        warm_up :int=       1000,   # warmup steps, None or 0 turns-off
        ann_base :float=    0.999,  # annealing base, None or 1 for turn-off
        ann_step :float=    1.0,    # annealing step, higher value speeds up annealing
        n_wup_off :float=   2.0,    # N warmUp offset of annealing
        verb=               0):

    if verb > 0: print('*** lr_scaler *** initial LR', iLR)

    # create global step variable if not given
    if g_step is None:
        g_step = tf.get_variable(
            name=           'g_step',
            shape=          [],
            trainable=      False,
            initializer=    tf.constant_initializer(0),
            dtype=          tf.int32)

    g_step_fl = tf.cast(g_step, dtype=tf.float32)
    if warm_up is None: warm_up = 0
    if warm_up:
        ratioWm = tf.reduce_min([g_step_fl, warm_up]) / warm_up # warmUp ratio
        lR = iLR * ratioWm # learning rate with warmup
        if verb > 0: print('applied warmUp (%d) to lR' % warm_up)
    else: lR = tf.constant(iLR)
    if ann_base is not None and ann_base != 1:
        gStep_offs = tf.reduce_max([0, g_step_fl - warm_up * n_wup_off]) # offset by warmUpSteps
        lR *= ann_base ** (gStep_offs * ann_step) # learning rate with annealing
        if verb>0: print('applied annealing to lR (%.5f,%.5f)' % (ann_base, ann_step))
    return {
        'scaled_LR':    lR,
        'g_step':       g_step}

# gradient clipping method (by clip_value or AVT algorithm)
def grad_clipper_AVT(
        gradients,                  # gradients to clip
        clip_value=     None,       # clipping value, for None clips with avt
        avt_SVal=       0.1,        # start value for AVT (smaller value makes warmup)
        avt_window=     100,        # width of averaging window (number of steps)
        avt_max_upd=    1.5,        # single step max factor of avt update
        do_clip=        True,       # disables clipping (just GN calculations)
        verb=           0):

    gg_norm = tf.global_norm(gradients) # gradients global norm
    avt_gg_norm = tf.get_variable( # time averaged gradients global norm variable
        name=           'avt_gg_norm',
        shape=          [],
        trainable=      False,
        initializer=    tf.constant_initializer(avt_SVal),
        dtype=          tf.float32)

    avt_update = tf.reduce_min([gg_norm, avt_max_upd * avt_gg_norm]) # single value to update AVTG with (current GNorm or clipped to max value)
    # assign new value
    avt_gg_norm = tf.assign(
        ref=    avt_gg_norm,
        value=  (avt_gg_norm * (avt_window-1) + avt_update) / avt_window)
    if verb > 0: print('grad_clipper_AVT: avt_SVal %.1f, avt_window %d, avt_max_upd %.1f' % (avt_SVal, avt_window, avt_max_upd))

    if do_clip:
        gradients, _ = tf.clip_by_global_norm(
            t_list=     gradients,
            clip_norm=  clip_value if clip_value else avt_gg_norm,
            use_norm=   gg_norm)
        if verb > 0: print(' >> is clipping gradients %s'%('with value' if clip_value else 'with AVT'))
    elif verb > 0: print(' >> not doing clipping')

    return {
        'gradients':    gradients,
        'gg_norm':      gg_norm,
        'avt_gg_norm':  avt_gg_norm}

# gradient clipping loss reductor (gradient clipping + optimizer) (wraps grad_clipper_AVT)
def gc_loss_reductor(
        optimizer :tf.compat.v1.train.Optimizer,
        vars :list=     None,
        g_step=         None,       # put here globalStep variable to update +1 with optimizer
        avg_loss=       None,       # put here loss if you do not have gradients yet
        gradients=      None,
        clip_value=     None,
        avt_SVal=       0.1,
        avt_window=     100,
        avt_max_upd=    1.5,
        do_clip=        True,
        verb=           0):

    if vars is None: vars = tf.trainable_variables()

    if gradients is None: gradients = tf.gradients(avg_loss, vars, colocate_gradients_with_ops=False)

    gc_out= grad_clipper_AVT(
        gradients=      gradients,
        clip_value=     clip_value,
        avt_SVal=       avt_SVal,
        avt_window=     avt_window,
        avt_max_upd=    avt_max_upd,
        do_clip=        do_clip,
        verb=           verb)
    clippedGradients =  gc_out['gradients']

    optimizer = optimizer.apply_gradients(
        grads_and_vars= zip(clippedGradients, vars),
        global_step=    g_step)

    return {
        'optimizer':    optimizer,
        'gg_norm':      gc_out['gg_norm'],
        'avt_gg_norm':  gc_out['avt_gg_norm']}

# replaces nan values @tensor with zero
def replace_nan_with_zero(tensor): return tf.where(tf.is_nan(tensor), tf.zeros_like(tensor), tensor)

# flattens list of tensors of any dimensions and returns one (flattened) tensor
def flatten_LOTens(tList):

    resh_vars = [tf.reshape(var, [-1]) for var in tList]
    return tf.concat(resh_vars, axis=-1)

# returns list of indexes of n layers, len(list)==n_select
def list_of_layers(
        n_layers :int,          # num of all layers
        n_select :int=  None):  # num of layers to select

    if n_select is None: n_select = n_layers
    if n_select > n_layers: n_select = n_layers

    layers = [] # 0 case
    if n_select == 1: layers = [0]
    if n_select > 1:

        layers = [0, n_layers - 1]  # first and last

        for i in range(n_select - 2):
            layers.append(int((n_layers - 1) / (n_select - 1) * (i + 1)))

    return sorted(list(set(layers)))

# calculates and returns num of variables(floats) @graph from given variables list or scope
def num_var_floats(
        variables :list=    None,   # list of variables
        graph=              None,
        scope :str=         None):  # scope name

    assert variables is not None or (scope & graph), 'ERR: variables list or (scope & graph) must be given'
    if variables is None: variables = graph.get_collection(name='variables', scope=scope)

    #with graph.as_default():
    numVars = sum(v.get_shape().num_elements() for v in variables)
    return numVars

# calculates size (num of values) of tensor given its shape
def sh_size(shape : list or tuple):
    size = 1
    for e in shape: size *= e
    return size

# prints variables log
def log_vars(
        variables: list,
        simple=     False,
        sort=       True): # use order from list or sorted by name

    print(f'Total num of variables: {len(variables)}')
    print(f' > num of floats: {short_scin(num_var_floats(variables), precise=True)}')
    if not simple:
        vns = [(v.name, v.shape) for v in variables]
        if sort:
            dVar = {v.name: v.shape for v in variables}
            vns = [(key, dVar[key]) for key in sorted(list(dVar.keys()))]
        for v in vns: print(f' >> v: {v[0]} {v[1]}')

# logs variables from checkpoint
def log_checkpoint(ckpt_FD):
    ckpt_vars = tf.contrib.framework.list_variables(ckpt_FD)
    tot_siz = 0
    for _, shape in ckpt_vars:
        tot_siz += sh_size(shape)
    print(f'\nGot {len(ckpt_vars)} variables in original checkpoint (total size {tot_siz} nums)')

    max_nm_len = 0
    max_sh_len = 0
    for var_name, shape in ckpt_vars:
        if len(var_name) > max_nm_len:   max_nm_len = len(var_name)
        if len(str(shape)) > max_sh_len: max_sh_len = len(str(shape))
    if max_nm_len > 90: max_nm_len = 90
    if max_sh_len > 20: max_sh_len = 20

    for var_name, shape in ckpt_vars:
        var = tf.contrib.framework.load_variable(ckpt_FD, var_name)
        print(f' > ({100*sh_size(shape)/tot_siz:4.1f}%) {var_name:{max_nm_len}s} {str(shape):{max_sh_len}s} {var.dtype}')

# weighted merge of two checkpoints
@proc_wait
def mrg_ckpts(
        ckptA :str,                 # checkpoint A (folder name)
        ckptA_FD :str,              # root folder of cpktA (absolute or relative)
        ckptB :str or None,         # checkpoint B (folder name), for None takes 100% ckptA
        ckptB_FD :str or None,      # root folder of cpktB (absolute or relative)
        ckptM :str,                 # checkpoint merged (folder name)
        ckptM_FD :str,              # root folder of cpktM (absolute or relative)
        mrgF :float=        0.5,    # merge factor
        noiseF :float=      0.0,    # noise factor, amount of noise added to nev value (0.0-1.0...)
        replace_scope :str= None,   # replaces outer scope with given string
        verb=               0):

    if ckptA_FD[-1] != '/': ckptA_FD += '/'
    if ckptB_FD and ckptB_FD[-1] != '/': ckptB_FD += '/'
    if ckptM_FD[-1] != '/': ckptM_FD += '/'

    var_namesA = sorted([v[0] for v in tf.contrib.framework.list_variables(ckptA_FD+ckptA)])
    if verb>0: print(f'variables from ckptA ({len(var_namesA):4d}): {var_namesA}')
    var_namesB = sorted([v[0] for v in tf.contrib.framework.list_variables(ckptB_FD+ckptB)]) if ckptB else []
    if verb>0: print(f'variables from ckptB ({len(var_namesB):4d}): {var_namesB}')

    oscope_len = 0
    if replace_scope:
        for c in var_namesA[0]:
            if c == '/': break
            oscope_len += 1
    if verb>0:
        print(f'oscope_len {oscope_len}')
        if oscope_len: print(f' > will replace {var_namesA[0][:oscope_len]} with {replace_scope}')

    avL = []
    with tf.variable_scope('av'):
        for var_name in var_namesA:
            var = tf.contrib.framework.load_variable(f'{ckptA_FD}{ckptA}', var_name)
            avL.append(tf.Variable(var, name=var_name))

    bvL = []
    if ckptB:
        with tf.variable_scope('bv'):
            for var_name in var_namesB:
                var = tf.contrib.framework.load_variable(f'{ckptB_FD}{ckptB}', var_name)
                bvL.append(tf.Variable(var, name=var_name))

    fin_vars = []
    for ix in range(len(var_namesA)):
        var_name = var_namesA[ix]
        if verb>0: print(f'old var_name: {var_name}')
        if replace_scope: var_name = replace_scope + var_name[oscope_len:]

        varA = avL[ix]
        if bvL and varA.dtype == 'float32':
            varB = bvL[ix]
            noise = tf.random.truncated_normal(
                shape=  varA.shape,
                stddev= tf.math.reduce_std(varA))
            var = tf.Variable(mrgF*varA + (1-mrgF)*varB + noiseF*noise, name=var_name)
        else:
            var = tf.Variable(varA, name=var_name)
        fin_vars.append(var)

    # save
    if verb>0: print('\nWriting checkpoint... ', end='')
    fin_saver = tf.train.Saver(fin_vars)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        fin_saver.save(sess, f'{ckptM_FD}{ckptM}/{ckptM}', write_meta_graph=False)
    tf.reset_default_graph()
    if verb>0: print('done!')

# processes zeroes array returned by model in following intervals
class ZeroesProcessor:

    def __init__(
            self,
            intervals :tuple=   (50,500,5000),
            tag_pfx=            'nane', # prefix of tag in TB
            summ_writer :tf.compat.v1.summary.FileWriter=   None):  # if given will put summaries to TB with intervals frequencies

        self.intervals = intervals
        self.zsL = {k: [] for k in self.intervals}
        self.single = []
        self.summ_writer = summ_writer
        self.tag_pfx = tag_pfx

    # takes next zeroes array and processes
    def process(
            self,
            zs :np.array,
            step :int=  None):

        self.single.append(np.mean(zs))
        rd = {}
        if len(self.single) == self.intervals[0]:
            rd[1] = np.mean(self.single)
            self.single = []
        for k in self.zsL:
            self.zsL[k].append(zs)
            if len(self.zsL[k]) == k:
                rd[k] = np.mean(np.where(np.mean(np.stack(self.zsL[k], axis=0), axis=0) == 1, 1, 0))
                self.zsL[k] = []

        if self.summ_writer and step:
            for k in rd:
                nane_summ = tf.Summary(value=[tf.Summary.Value(tag=f'{self.tag_pfx}/nane_{k}', simple_value=rd[k])])
                self.summ_writer.add_summary(nane_summ, step)

        return rd

# TensorBoard writer
class TBwr:

    def __init__(
            self,
            logdir: str,
            flush_secs= 10):

        if not os.path.isdir(logdir): os.mkdir(logdir)
        self.sw = tf.summary.FileWriter(logdir=logdir, flush_secs=flush_secs)

    def add(self,
            val,
            name,
            step):

        sv = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=val)])
        self.sw.add_summary(sv, step)
