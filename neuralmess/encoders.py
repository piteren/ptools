"""

 2019 (c) piteren

"""

import tensorflow as tf
from tensorflow.contrib import rnn

from ptools.neuralmess.base_elements import my_initializer, list_of_layers
from ptools.neuralmess.layers import lay_res, zeroes, lay_dense, lay_conv1D, attn


# single DRT layer, base of enc_DRT (based on some Transformer concepts)
def lay_DRT(
        input,
        name=           'lay_DRT',  # scope name, be careful when stacked since auto_reuse
        hist_name=      None,       # family name of histogram
        dns_scale=      4,
        activation=     tf.nn.relu, # gelu is really worth a try
        dropout=        0.0,        # dropout (after two denses)
        training_flag=  None,       # training flag tensor (for dropout)
        initializer=    None,
        seed=           12321):

    if not hist_name: hist_name = name
    lay_width = input.shape[-1]
    if initializer is None: initializer = my_initializer(seed)
    hist_summ = []

    with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):

        hist_summ.append(tf.summary.histogram('a_denseSin', input, family=hist_name))

        # dense (scale up)
        output = lay_dense(
            input=          input,
            units=          int(lay_width * dns_scale),
            activation=     None,
            use_bias=       True,
            initializer=    initializer,
            seed=           seed,
            name=           'denseS')
        hist_summ.append(tf.summary.histogram('b_denseSout', output, family=hist_name))

        # activation
        output = activation(output)
        hist_summ.append(tf.summary.histogram('c_activation', output, family=hist_name))
        zsL = [zeroes(output)] # zeroes list

        # dense (scale down) no activ
        output = lay_dense(
            input=          output,
            units=          lay_width,
            name=           'DRTdenseNA',
            use_bias=       True,
            initializer=    initializer,
            seed=           seed)
        hist_summ.append(tf.summary.histogram('d_denseNAout', output, family=hist_name))

        # layer dropout
        if dropout:
            output = tf.layers.dropout(
                inputs=     output,
                rate=       dropout,
                training=   training_flag,
                seed=       seed)

        # residual
        output = lay_res(input, output)
        hist_summ.append(tf.summary.histogram('e_residual', output, family=hist_name))

        # layer_norm
        output = tf.contrib.layers.layer_norm(
            inputs=             output,
            begin_norm_axis=    -1,
            begin_params_axis=  -1)

        hist_summ.append(tf.summary.histogram('f_LAYout', output, family=hist_name))

    return{
        'output':       output,
        'hist_summ':    hist_summ,
        'zeroes':       zsL}

# Deep Residual encoder
def enc_DRT(
        input,
        name=           'enc_DRT',
        n_layers=       12,
        lay_width :int= None,       # for None matches input width
        dns_scale=      6,          # scale(*) of first dense
        activation=     tf.nn.relu,     # gelu is really worth a try
        dropout=        0.0,        # dropout after two denses
        training_flag=  None,       # training flag tensor (for dropout)
        initializer=    None,
        seed=           12321,
        n_hist=         4,          # number of histogram layers (for TB)
        verb=           0):

    lay_width_matched = ''
    if lay_width is None:
        lay_width = input.shape.as_list()[-1]
        lay_width_matched = '(lay_width taken form input width)'
    if verb > 0: print('\nBuilding DRTencoder (%dx%d drop:%.2f) %s...' % (n_layers, lay_width, dropout, lay_width_matched))

    if initializer is None: initializer = my_initializer(seed)

    hist_summ = []
    hist_layers = list_of_layers(n_layers, n_select=n_hist)
    if verb > 1: print(' > histogram layers of DRencoder:', hist_layers)

    zsL = [] # zeroes list
    with tf.variable_scope(name):

        # input projection
        iW = input.shape[-1]
        if iW != lay_width:
            input = lay_dense(
                input=          input,
                units=          lay_width,
                use_bias=       False,
                initializer=    my_initializer(),
                seed=           seed)
            if verb > 0: print('projected input to layWidth(%d) since it differs(%d)' % (lay_width, iW))

        # input layer_norm
        input = tf.contrib.layers.layer_norm(
            inputs=             input,
            begin_norm_axis=    -1,
            begin_params_axis=  -1)

        output = input # for 0 layers case
        for nL in range(n_layers):

            lay_out = lay_DRT(
                input=          output,
                name=           f'DRLay_{nL}',
                hist_name=      name,
                dns_scale=      dns_scale,
                activation=     activation,
                dropout=        dropout,
                training_flag=  training_flag,
                initializer=    initializer,
                seed=           seed)

            output = lay_out['output']
            if nL in hist_layers: hist_summ.append(lay_out['hist_summ'])
            zsL += lay_out['zeroes']

    return {
        'output':       output,
        'hist_summ':    hist_summ,
        'zeroes':       zsL}

# CNN 1D encoder (for sequences, LN > cnn > act > drop > RES), number of parameters: n_layers*kernel*input.width*n_filters
def enc_CNN(
        input :tf.Tensor,
        history :tf.Tensor=                 None,       # optional history(state) tensor with shape [bsz, n_layers ,kernel-1, n_filters], >> masked cnn
        name=                               'enc_CNN',
        shared_lays :bool=                  False,      # shared variables in layers
        n_layers :int=                      12,         # num of layers
        kernel :int=                        3,          # layer kernel
        n_filters :int=                     128,        # num of filters
        dns_scale :int or None=             0,          # scale(*) of first dense for lay_DRT after CNN, for None or 0 DRT won't be build
        activation=                         tf.nn.relu, # gelu is really worth a try
        dropout :float or None=             0.0,
        dropout_drt :float or None=         0.0,
        training_flag :tf.Tensor or bool=   None,       # dropout training flag tensor
        initializer=                        None,
        seed :int=                          12321,
        n_hist :int=                        4,          # number of histogram layers
        verb=                               0):

    if verb>0: print(f'\n *** enc_CNN *** Building {name} ({n_layers}x{n_filters})...')

    if initializer is None: initializer = my_initializer(seed)

    # manage history
    history_lays = None
    if history is not None:
        history_lays = tf.unstack(history, axis=-3)
        if verb>1: print(f' > state_lays len {len(history_lays)} of: {history_lays[0]}')

    hist_summ = []
    hist_layers = list_of_layers(n_layers, n_select=n_hist)
    if verb>1: print(f' > histogram layers of cnn encoder: {hist_layers}')

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        input_lays = [] # here we will store inputs of the following layers to extract the state (history)
        zsL = [] # zeroes

        # input projection - to match n_filters and input width
        if verb > 1: print(f' > encoder input: {input}')
        if input.shape[-1] != n_filters:
            input = lay_dense(
                input=          input,
                units=          n_filters,
                name=           'enc_input_projection',
                initializer=    initializer)
            if verb > 1: print(f' > encoder projected input: {input}')

        output = input # for 0 layers case
        sub_output = input # first input
        for depth in range(n_layers):

            lay_name = f'enc_CNN_lay_{depth}' if not shared_lays else 'enc_CNN_lay_shared'
            if verb>1: print(f'<< layer {lay_name}:')

            lay_input = tf.concat([history_lays[depth], sub_output], axis=-2) if history_lays else sub_output
            if verb>1:
                print(f' > sub_output (previous): {sub_output}')
                print(f' > lay_input (eventually padded): {lay_input}')
            input_lays.append(lay_input)

            hist_lay = depth in hist_layers

            with tf.variable_scope(lay_name):

                if hist_lay: hist_summ.append(tf.summary.histogram('a_lay_in', lay_input, family=name))

                # LN
                lay_input = tf.contrib.layers.layer_norm(
                    inputs=             lay_input,
                    begin_norm_axis=    -1,
                    begin_params_axis=  -1)
                if hist_lay: hist_summ.append(tf.summary.histogram('b_LN', lay_input, family=name))

                # conv no activation
                output = lay_conv1D(
                    input=          lay_input,
                    name=           'conv1D',
                    kernels=        kernel,
                    filters=        n_filters,
                    activation=     None,
                    initializer=    initializer,
                    padding=        'same' if history is None else 'valid',
                    seed=           seed,
                    verb=           0)
                if hist_lay: hist_summ.append(tf.summary.histogram('c_cnn', output, family=name))

                # activation
                if activation:
                    output = activation(output)
                    if hist_lay: hist_summ.append(tf.summary.histogram('d_activation', output, family=name))
                    zsL += zeroes(output) # catch zeroes

                # dropout
                if dropout:
                    output = tf.layers.dropout(
                        inputs=     output,
                        rate=       dropout,
                        training=   training_flag,
                        seed=       seed)
                    if hist_lay: hist_summ.append(tf.summary.histogram('e_drop', output, family=name))

                # RES, here we take sub_output, since lay_input may be padded by history
                output += sub_output
                if hist_lay: hist_summ.append(tf.summary.histogram('f_residual', output, family=name))

                if verb > 1: print(f' > output (layer): {output}')

                if dns_scale:
                    lay_out = lay_DRT(
                        input=          output,
                        name=           lay_name + '_lay_DRT',
                        hist_name=      name,
                        dns_scale=      dns_scale,
                        activation=     activation,
                        dropout=        dropout_drt,
                        training_flag=  training_flag,
                        initializer=    initializer,
                        seed=           seed)
                    output = lay_out['output']
                    if hist_lay: hist_summ.append(lay_out['hist_summ'])
                    zsL += lay_out['zeroes']

                sub_output = output

    # final LN
    output = tf.contrib.layers.layer_norm(
        inputs=             output,
        begin_norm_axis=    -1,
        begin_params_axis=  -1)

    # prepare fin_state
    fin_state = None
    if history is not None:
        state = tf.stack(input_lays, axis=-3)
        if verb > 1: print(f' > state (stacked): {state}')
        fin_state = tf.split(state, num_or_size_splits=[-1, kernel-1], axis=-2)[1]
        if verb > 1: print(f' > fin_state (split): {fin_state}')

    if verb > 1: print(f' > {name} output: {output}')
    return {
        'output':       output,
        'state':        fin_state, # history for next
        'hist_summ':    hist_summ,
        'zeroes':       zsL}

# rnn encoder(for sequence encoding)
def enc_RNN(
        input,
        name=           'enc_RNN',
        cellFN=         rnn.NASCell,    # cell function (GRUCell NASCell LSTMCell)
        biDir=          True,
        cellWidth=      64,
        numLays=        1,
        dropout=        0,
        dropFlagT=      None,       # dropout training flag tensor
        seed=           12321,
        n_hist=         4,              # number of histogram layers
        verb=           0):
    """
    bidirectional rnn encoder (wRES)
    number of parameters for LSTMcell:  8c(e+c+1)
    number of parameters for NAScell:   16c(e+c)
    """

    histSumm = []
    histLayers = list_of_layers(numLays, n_select=n_hist)
    if verb > 0: print('histogram layers of rnnEncoder:', histLayers)

    with tf.variable_scope(name):
        if verb > 0: print('building RNNencoder...')

        fCells = [cellFN(cellWidth) for _ in range(numLays)]
        bCells = [cellFN(cellWidth) for _ in range(numLays)] if biDir else None

        layIN = input
        for lay in range(len(fCells)):

            histLay = lay in histLayers
            layName = 'rnnEncLay_%s' % lay
            with tf.variable_scope(layName):

                if biDir:
                    # bidirectional rnn layer
                    layOUT, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=    fCells[lay],
                        cell_bw=    bCells[lay],
                        inputs=     layIN,
                        dtype=      tf.float32)
                    layOUT = tf.concat(layOUT, -1)

                else:
                    # f direction rnn
                    layOUT, _ = tf.nn.dynamic_rnn(
                        cell=       fCells[lay],
                        inputs=     layIN,
                        dtype=      tf.float32)

                if histLay: histSumm.append(tf.summary.histogram('a_rnnOut', layOUT, family='tensor'))

                # residual connection
                layOUT = lay_res(layIN, layOUT)
                if histLay: histSumm.append(tf.summary.histogram('b_residual', layOUT, family='tensor'))

                # dropout
                if dropout:
                    layOUT = tf.layers.dropout(
                        inputs=     layOUT,
                        rate=       dropout,
                        training=   dropFlagT,
                        seed=       seed)
                    if histLay: histSumm.append(tf.summary.histogram('c_dropOut', layOUT, family=name))

                layIN = layOUT

    return {
        'output':   layOUT,
        'histSumm': histSumm}

# transformer encoder
def enc_TNS(
        in_seq,                                 # input sequence embeddings [batch, seq, emb], for TAT in_seq should be LNormalized
        name=                       'enc_TNS',
        seq_out :bool=              True,       # transformer seq2seq, if False seq2one (Task Attention)
        add_PE :bool=               True,       # add positional embeddings
        do_LN :bool=                True,       # do layer norm
        n_blocks=                   12,
        n_heads=                    8,
        dense_mul :int or float=    4,          # dense (after att) scale
        activation=                 tf.nn.relu,
        max_seq_len=                100,        # used only to set shape (axis 0) of positional embeddings
        dropout=                    0.0,        # dropout of FC after attention
        dropout_att=                0.0,        # dropout of attention probabilities
        drop_flag=                  None,       # dropout training flag (bool or tensor)
        initializer=                None,
        seed=                       12321,
        n_hist=                     4,          # number of histogram layers
        verb=                       0):

    if initializer is None: initializer = tf.truncated_normal_initializer(stddev=0.01, seed=seed)

    # split feats(-1) for heads
    def split_heads(x):
        x = tf.split(x, n_heads, axis=-1) # list of tensors
        return tf.stack(x, axis=-3)  # [batch, head, seq, feats]

    # merge heads over feats(-1)
    def merge_heads(x):
        x = tf.unstack(x, axis=-3)
        return tf.concat(x, axis=-1)

    # multi_head_attention for input
    def mh_attn(
            in_seq,                 # input sequence [batch, seq, feats]
            query=          None,   # None for self attention, otherwise TAT [batch, n_queries, feats]
            activation=     None,   # activation of KQV dense
            dropout_att=    0.0,
            drop_flag=      None,
            seed=           seed):

        # input projection of in_seq for KQV or KV(if query)
        width = in_seq.shape[-1].value
        proj_size = 3 if query is None else 2
        c = lay_dense(
            input=          in_seq,  # [batch, seq, feats]
            units=          width*proj_size,
            name=           'mhProj',
            activation=     activation,
            initializer=    initializer,
            seed=           seed)
        ins_split = tf.split(c, proj_size, axis=-1) # split projected

        if query is not None:
            q = query # projection for Q is not needed (at least with 1 head)
            k,v = ins_split
        else:
            q,k,v = ins_split
        q,k,v = map(split_heads, [q,k,v])

        # attention
        att_out = attn(q, k, v, dropout_att, drop_flag, seed)
        a = att_out['attention']
        a = merge_heads(a)
        return {
            'attention':    a,
            'att_vals':     att_out['att_weights']}

    # transformer block
    def tblock(
            in_seq,
            seed,
            task_query=  None):

        hist_summ = []

        output = in_seq
        taskQueryNorm = None
        if task_query is None:
            hist_summ.append(tf.summary.histogram('a_inputSeq', output, family=name))
            # layer norm 1 on seq
            if do_LN:
                output = tf.contrib.layers.layer_norm(
                    inputs=             output,
                    begin_norm_axis=    -1,
                    begin_params_axis=  -1)
                hist_summ.append(tf.summary.histogram('b_inputSeqLN', output, family=name))
        else:
            hist_summ.append(tf.summary.histogram('a_inTaskQuery', task_query, family=name))
            taskQueryNorm = task_query
            # layer norm 1 on taskQuery
            if do_LN:
                taskQueryNorm = tf.contrib.layers.layer_norm(
                    inputs=             task_query,
                    begin_norm_axis=    -1,
                    begin_params_axis=  -1)
                hist_summ.append(tf.summary.histogram('b_taskQueryLN', task_query, family=name))

        # multi head self attention
        mha_out = mh_attn(
            in_seq=         output,
            query=          taskQueryNorm,
            dropout_att=    dropout_att,
            drop_flag=      drop_flag,
            seed=           seed)
        output =    mha_out['attention']
        att_vals =  mha_out['att_vals']
        hist_summ.append(tf.summary.histogram('c_mhAttn', output, family=name))

        # dense without activation
        output = lay_dense(
            input=          output,
            units=          output.shape[-1].value,
            name=           'afterAttProj',
            initializer=    initializer,
            seed=           seed)
        hist_summ.append(tf.summary.histogram('d_denseAftAtt', output, family=name))

        if dropout:
            output = tf.layers.dropout(
                inputs=     output,
                rate=       dropout,
                training=   drop_flag,
                seed=       seed)

        # residual 1
        if task_query is None:
            res1_out = in_seq + output
            hist_summ.append(tf.summary.histogram('e_res_onInputSeq', res1_out, family=name))
        else:
            res1_out = task_query + output
            hist_summ.append(tf.summary.histogram('e_res_onTaskQuery', res1_out, family=name))

        output = res1_out
        # layer norm 2
        if do_LN:
            output = tf.contrib.layers.layer_norm(
                inputs=             output,
                begin_norm_axis=    -1,
                begin_params_axis=  -1)
            hist_summ.append(tf.summary.histogram('f_layNorm', output, family=name))

        # 2x dense
        base_width = output.shape[-1].value
        output = lay_dense(
            input=          output,
            units=          int(base_width*dense_mul),
            name=           'dense1afterAtt',
            activation=     activation,
            initializer=    initializer,
            seed=           seed)
        zsL =  zeroes(output)
        hist_summ.append(tf.summary.histogram('g_1denseOut', output, family=name))
        output = lay_dense(
            input=          output,
            units=          base_width,
            name=           'dense2afterAtt',
            initializer=    initializer,
            seed=           seed)
        hist_summ.append(tf.summary.histogram('h_2denseOut', output, family=name))

        if dropout:
            output = tf.layers.dropout(
                inputs=     output,
                rate=       dropout,
                training=   drop_flag,
                seed=       seed)

        # residual2
        output += res1_out
        hist_summ.append(tf.summary.histogram('i_res', output, family=name))

        return {
            'output':       output,
            'hist_summ':    hist_summ,
            'att_vals':     att_vals,
            'zeroes':       zsL}

    width = in_seq.shape[-1] # sequence width (feats)
    seq_len = tf.shape(in_seq)[-2] # sequence length (time)

    if verb > 0:
        print('\nBuilding %s (transformer encoder) (%dx%d, denseMul %.1f), ' % (name, n_blocks, width, dense_mul))
        print(' > dropout: %.2f %.2f(att)' % (dropout, dropout_att))
        print(' > seq2seq mode...') if seq_out else print(' > task attention mode...')

    hist_layers = list_of_layers(n_blocks, n_select=n_hist)
    if verb > 1: print(' > histogram layers of transformer encoder:', hist_layers)

    with tf.variable_scope(name):

        hist_summ = []  # list of histogram summaries

        if verb > 1: print(' > transformer input', in_seq)
        hist_summ.append(tf.summary.histogram('a_transformerInput', in_seq, family=name))

        # init task_query (for first block - input averaged over time (seq))
        task_query = None
        if not seq_out:
            task_query = tf.reduce_mean(in_seq, axis=-2, keep_dims=True)# [batch,1,feats]
            if verb>1: print(' > first task_query (reduced input) for TAT', task_query)

        # positional embedding
        if add_PE:
            pos_emb_var = tf.get_variable(
                name=           'tnsPosEmb',
                shape=          [max_seq_len, width],
                initializer=    initializer)
            in_seq += tf.nn.embedding_lookup(params=pos_emb_var, ids=tf.range(seq_len))
            if verb>1: print(' > added positional embedding to the input...')
            hist_summ.append(tf.summary.histogram('b_transformerPosEmbInput', in_seq, family=name))

        if verb > 1: print(' > building %d blocks of transformer...' % n_blocks)
        att_vals = [] # list of block attention values
        zsL = []
        block_output = None
        for nB in range(n_blocks):
            hist_lay = nB in hist_layers
            with tf.variable_scope('block_%d' % nB):
                bo_dict = tblock(
                    in_seq=     in_seq,
                    seed=       seed,
                    task_query= task_query)
                block_output = bo_dict['output']
                if task_query is None: in_seq = block_output
                else: task_query = block_output

                zsL +=                      bo_dict['zeroes']
                if hist_lay: hist_summ +=   bo_dict['hist_summ']
                att_block_vals =            bo_dict['att_vals'] #[batch,head,query_n or seq,seq]
                att_vals.append(att_block_vals)

        if task_query is None:  output = block_output
        else:                   output = tf.squeeze(task_query, axis=-2)

        if do_LN:
            output = tf.contrib.layers.layer_norm(
                inputs=             output,
                begin_norm_axis=    -1,
                begin_params_axis=  -1)
        hist_summ.append(tf.summary.histogram('c_transformer_out', output, family=name))

    if verb > 1: print(' > %s output' % name, output)
    return {
        'output':       output,
        'hist_summ':    hist_summ,
        'att_vals':     att_vals,
        'zeroes':       zsL}


if __name__ == "__main__":

    n_lays = 3
    width = 50
    kernel = 5

    input_PH = tf.placeholder(
        name=   'input_PH',
        dtype=  tf.float32,
        shape=  [None, 20, width])  # [bsz,seq,width]

    # uncomment for masked CNN
    #"""
    state_PH = tf.placeholder(
        name=   'state_PH',
        dtype=  tf.float32,
        shape=  [None, n_lays, kernel-1, width])  # [bsz,n_lay,2,width]
    #"""
    #state_PH = None

    encoder = enc_CNN(
        input=      input_PH,
        history=    state_PH,
        n_layers=   n_lays,
        kernel=     kernel,
        n_filters=  width,
        verb=       2)

