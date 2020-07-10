"""

 2018 (c) piteren

    Deep Vector Classifier - Model

    TODO:
       - seq2seq model (enc_tower is ready >> need to add seq2seq flag for model >> build seq2seq classifiers)
       - whether seq2seq would advance from reduced vec concatenated to every token
        - implement regression (classes [1])

"""

import tensorflow as tf

from putils.neuralmess.base_elements import gelu
from putils.neuralmess.layers import lay_dense, tf_drop
from putils.neuralmess.encoders import enc_CNN, enc_TNS, enc_DRT, enc_RNN

# builds single encoding tower graph, from model input to single vector (sub-graph of dvc_model)
def enc_tower(
        actv_func,                      # activation function
        vec_PH :tf.placeholder,         # vector placeholder (vec input)
        tks_PH :tf.placeholder,         # tokens seq placeholder (seq input - IDs)
        seq_PH :tf.placeholder,         # vector seq placeholder (seq input)
        train_flag_PH :tf.placeholder,  # train flag placeholder
        tok_emb,
        train_TE :bool,
        tok_emb_add,                    # np.arr/LL with values of additional embeddings (always trainable)
        max_seq_len,
        inV_drop :float,
        inV_proj :int or None,          # value equal to last dimension width turns-off projection
        inS_drop :float,
        intime_drop :float,
        infeat_drop :float,
        inS_proj :int or None,          # value equal to last dimension width turns-off projection
        inS_actv :bool,                 # inS_proj activation
        drt_scale,
        tns_scale,
        cnn_nLay,
        lstm_nLay,
        tns_nBlocks,
        enc_drop,
        tnsAT_drop,
        tat_nBlocks,
        tatAT_drop,
        tat_drop,
        drt_nLay,
        drt_drop,
        seed,
        verb,
        **kwargs):

    if verb > 0: print('\nenc_tower inits...')
    zsL = []
    hist_summ = []

    with tf.variable_scope('encTower', reuse=tf.AUTO_REUSE):
        vectorL = []  # list of vectors to concatenate (vec form vec_PH + reduced sequence (tok_PH & seq_PH))

        # ********************************* vector processing
        if vec_PH is not None:
            vector = vec_PH
            if verb > 1: print(' > vector input:', vector)
            hist_summ.append(tf.summary.histogram('1vecIn', vector, family='A.vec'))

            # layerNorm (on input, always)
            vector = tf.contrib.layers.layer_norm(
                inputs=             vector,
                begin_norm_axis=    -1,
                begin_params_axis=  -1)
            hist_summ.append(tf.summary.histogram('2inLNorm', vector, family='A.vec'))

            # dropout (on input, before projection)
            if inV_drop:
                vector = tf.layers.dropout(
                    inputs=     vector,
                    rate=       inV_drop,
                    training=   train_flag_PH,
                    seed=       seed)
                if verb > 1: print(' > dropout %.2f applied to vec:' % inV_drop, vector)

            # projection (rescales input, without activation)
            if inV_proj and inV_proj!=vector.shape.as_list()[-1]:
                vector = lay_dense(
                    input=      vector,
                    units=      inV_proj,
                    activation= None,
                    use_bias=    True,
                    seed=       seed,
                    name=       'inVProjection')
                if verb > 1: print(' > projected vector input:', vector)
                hist_summ.append(tf.summary.histogram('3inProj', vector, family='A.vec'))

                # layerNorm (after projection)
                vector = tf.contrib.layers.layer_norm(
                    inputs=             vector,
                    begin_norm_axis=    -1,
                    begin_params_axis=  -1)
                hist_summ.append(tf.summary.histogram('4projLNorm', vector, family='A.vec'))

            # DRT encoder for vector @tower
            if drt_nLay:
                eDRTout = enc_DRT(
                    input=          vector,
                    n_layers=       drt_nLay,
                    dns_scale=      drt_scale,
                    activation=     actv_func,
                    dropout=        drt_drop,
                    training_flag=  train_flag_PH,
                    seed=           seed,
                    n_hist=            2,
                    verb=           verb)
                vector =        eDRTout['output']
                zsL +=          eDRTout['zeroes']
                hist_summ +=    eDRTout['hist_summ']
                if verb > 1: print(' > drtLay output', vector)
                hist_summ.append(tf.summary.histogram('5drtLayOut', vector, family='A.vec'))

            vectorL.append(vector)

        # ********************************* tokens embedding for sequence
        seq_to_concat = []
        if tks_PH is not None:
            if type(tok_emb) is tuple:
                allEmb = tf.get_variable( # embeddings initialized from scratch
                    name=           'tokEmbV',
                    shape=          tok_emb,
                    initializer=    tf.truncated_normal_initializer(stddev=0.01, seed=seed),
                    dtype=          tf.float32,
                    trainable=      True)
            else:
                allEmb = tf.get_variable( # embeddings initialized with given variable
                    name=           'tokEmbV',
                    initializer=    tok_emb,
                    dtype=          tf.float32,
                    trainable=      train_TE)
            if tok_emb_add is not None:
                tokEmbAddV = tf.get_variable( # add embeddings initialized with given variable
                    name=           'tokEmbAdd',
                    initializer=    tok_emb_add,
                    dtype=          tf.float32,
                    trainable=      True)
                allEmb = tf.concat([allEmb, tokEmbAddV], axis=0)

            sequence = tf.nn.embedding_lookup(params=allEmb, ids=tks_PH)
            if verb > 1: print('\n > sequence (tokens lookup):', sequence)
            hist_summ.append(tf.summary.histogram('1seqT', sequence, family='B.seq'))
            seq_to_concat.append(sequence)

        sequence = None
        if seq_PH is not None:
            if verb > 1: print(' > sequence of vectors:', seq_PH)
            hist_summ.append(tf.summary.histogram('1seqV', seq_PH, family='B.seq'))
            seq_to_concat.append(seq_PH)
        if len(seq_to_concat) > 2:
            """
            sequence = tf.concat(seq_to_concat, axis=1)
            if verb > 1: print(' > concatenated sequence (vec+tok):', sequence)
            """
            assert False, 'Need to match shapes (to.do)...'  # do it if needed
        if len(seq_to_concat) == 1: sequence = seq_to_concat[0]

        # ********************************* sequence processing
        if sequence is not None:

            # dropout (applied to seq of tokEmb works much better than applied after projection)
            if inS_drop:
                sequence = tf.layers.dropout(
                    inputs=     sequence,
                    rate=       inS_drop,
                    training=   train_flag_PH,
                    seed=       seed)
                if verb > 1: print(' > dropout %.2f applied to seq:' % inS_drop, sequence)

            # time & feats drop
            sequence = tf_drop(
                input=      sequence,
                time_drop=  intime_drop,
                feat_drop=  infeat_drop,
                train_flag= train_flag_PH,
                seed=       seed)

            # sequence layer_norm (on (dropped)input, always)
            sequence = tf.contrib.layers.layer_norm(
                inputs=             sequence,
                begin_norm_axis=    -2,
                begin_params_axis=  -2)
            if verb > 1: print(' > normalized seq:', sequence)
            hist_summ.append(tf.summary.histogram('2inLNorm', sequence, family='B.seq'))

            # in_projection (rescales input) without activation
            if inS_proj and inS_proj!=sequence.shape.as_list()[-1]:
                sequence = lay_dense(
                    input=      sequence,
                    units=      inS_proj,
                    activation= actv_func if inS_actv else None,
                    use_bias=    True,
                    seed=       seed,
                    name=       'inSProjection')
                if verb > 1: print(' > inProjection (%d) for seq:' % inS_proj, sequence)
                hist_summ.append(tf.summary.histogram('3inProj', sequence, family='B.seq'))

                # layerNorm (after projection)
                sequence = tf.contrib.layers.layer_norm(
                    inputs=             sequence,
                    begin_norm_axis=    -2,
                    begin_params_axis=  -2)
                if verb > 1: print(' > normalized seq:', sequence)
                hist_summ.append(tf.summary.histogram('4projLNorm', sequence, family='B.seq'))

            # ********* below are 3 types of seq2seq encoders stacked each on another
            enc_width = sequence.shape.as_list()[-1]
            if cnn_nLay:
                eCOut = enc_CNN(
                    input=          sequence,
                    n_layers=        cnn_nLay,
                    activation=     actv_func,
                    dropout=        enc_drop,
                    training_flag=      train_flag_PH,
                    n_filters=       enc_width,
                    n_hist=            2,
                    seed=           seed,
                    verb=        verb)
                sequence = eCOut['output']
                hist_summ += eCOut['hist_summ']

            if lstm_nLay:
                from tensorflow.contrib import rnn
                eLOut = enc_RNN(
                    input=          sequence,
                    cellFN=         rnn.LSTMCell,
                    biDir=          False,
                    cellWidth=      enc_width,
                    numLays=        lstm_nLay,
                    dropout=        enc_drop,
                    dropFlagT=      train_flag_PH,
                    seed=           seed)
                sequence = eLOut['output']

            if tns_nBlocks:
                tns_out = enc_TNS(
                    in_seq=         sequence,
                    name=           'encTRNS',
                    n_blocks=       tns_nBlocks,
                    n_heads=        1,
                    dense_mul=      tns_scale,
                    activation=     actv_func,
                    max_seq_len=    max_seq_len,
                    dropout_att=    tnsAT_drop,
                    dropout=        enc_drop,
                    drop_flag=      train_flag_PH,
                    seed=           seed,
                    n_hist=        2,
                    verb=           verb)
                sequence =      tns_out['output']
                hist_summ +=    tns_out['hist_summ']
                zsL +=          tns_out['zeroes']

            # ********** below sequence is reduced to vector, with TAT or pooling
            # TAT reduction
            if tat_nBlocks:
                tat_out = enc_TNS(
                    in_seq=         sequence,
                    seq_out=        False,
                    name=          'tatTRNS',
                    n_blocks=       tat_nBlocks,
                    n_heads=        1,
                    dense_mul=      tns_scale,
                    activation=     actv_func,
                    max_seq_len=    max_seq_len,
                    dropout_att=    tatAT_drop,
                    dropout=        tat_drop,
                    drop_flag=      train_flag_PH,
                    seed=           seed,
                    n_hist=        2,
                    verb=           verb)
                sequence_reduced =  tat_out['output']
                hist_summ +=        tat_out['hist_summ']
                attVals =           tat_out['att_vals']
                zsL +=              tat_out['zeroes']

            # reduce sequence with concat of avg & max
            else:
                sequence_reduced = tf.concat([tf.reduce_mean(sequence, axis=-2),tf.reduce_max(sequence, axis=-2)], axis=-1)
                if verb > 1: print(' > reduced sequence to one vec with mean (+) max:', sequence_reduced)

            vectorL.append(sequence_reduced)

        # ********************************* concatenate and finish
        vector = tf.concat(vectorL, axis=-1) if len(vectorL) > 1 else vectorL[0]
        if verb > 1: print(' > vector (tower output):', vector)

        tower_vars = tf.global_variables(scope=tf.get_variable_scope().name) # eTower variables

    return {
        'vector':       vector,
        'sequence':     sequence,
        'tower_vars':   tower_vars,
        'hist_summ':    hist_summ,
        'zeroes':        zsL}


# builds DVC model graph
def dvc_model(
        seed :int,              # seed for TF OPs
        multi_sen :int,
        train_tower :bool,
        vec_width :int,
        tok_emb,                # tuple with embeddings shape or np.arr/LL with values of embeddings
        seq_width :int,
        max_seq_len: int,
        drt_scale :float or int,
        classes :int or list,   # (Multi-Classif)
        vtc_drop :float,
        vtc_proj :int,
        drtC_nLay :int,
        drtC_drop :float,
        out_drop :float,
        l2lc :float,
        verb,
        **kwargs):

    #actv_func = tf.nn.relu
    actv_func = gelu

    hist_summ = []
    zsL = []
    isVec = vec_width is not None
    isTks = tok_emb is not None
    isSeq = seq_width is not None

    if verb > 0:
        print('\n*** DVCmodel *** builds graph for', end='')
        if isVec: print(' vec(%d)' % vec_width, end='')
        if isTks: print(' tks (tokens sequence)', end='')
        if isSeq: print(' seq (vectors sequence)', end='')
        print()

    if type(classes) is not list:
        classes = [classes] if classes is not None else []  # nClasses may be None >> no classifiers

    with tf.variable_scope(name_or_scope='FWD'):

        # ********************************* input placeholders
        vec_PHL = [tf.placeholder(
            name=       'vec%dPH'%nS,
            dtype=      tf.float32,
            shape=      [None, vec_width]) for nS in range(multi_sen)] if isVec else None

        tks_PHL = [tf.placeholder(
            name=       'tks%dPH'%nS,
            dtype=      tf.int32,
            shape=      [None,max_seq_len]) for nS in range(multi_sen)] if isTks else None # batch, seqLen

        seq_PHL = [tf.placeholder(
            name=       'seq%dPH'%nS,
            dtype=      tf.float32,
            shape=      [None,max_seq_len,seq_width]) for nS in range(multi_sen)] if isSeq else None # batch, seqLen, vec

        lab_PHL = [tf.placeholder(
            name=       'labC%dID'%nC,
            dtype=      tf.int32,
            shape=      [None]) for nC in range(len(classes))]

        train_flag_PH = tf.placeholder(name='trainFlag', dtype=tf.bool, shape=[]) # placeholder marking training process

        # ********************************* eTowers
        if verb > 0: print('...building %d DVC encTowers' % multi_sen)
        enc_outs = []
        for nS in range(multi_sen):
            encT_out = enc_tower(
                actv_func=      actv_func,
                vec_PH=         vec_PHL[nS] if vec_PHL is not None else None,
                tks_PH=         tks_PHL[nS] if tks_PHL is not None else None,
                seq_PH=         seq_PHL[nS] if seq_PHL is not None else None,
                train_flag_PH=  train_flag_PH,
                tok_emb=        tok_emb,
                max_seq_len=    max_seq_len,
                drt_scale=      drt_scale,
                seed=           seed,
                verb=           verb,
                **kwargs)
            enc_outs.append(encT_out)

        vec_output = tf.concat([eo['vector'] for eo in enc_outs], axis=-1)
        if len(enc_outs) > 1 and verb > 0:
            print('\n > outputs (concatenated) of %d towers:'%len(enc_outs), vec_output)

        tower_vars = enc_outs[0]['tower_vars']
        hist_summ += enc_outs[0]['hist_summ']
        for encT_out in enc_outs: zsL += encT_out['zeroes']

        hist_summ.append(tf.summary.histogram('5towersOut_concatALL', vec_output, family='C.cls'))

    # ********************************* Multi-Classifier
    with tf.variable_scope('vClassif'):

        if classes:

            # dropout on vector to classifier
            if vtc_drop:
                vec_output = tf.layers.dropout(
                    inputs=     vec_output,
                    rate=       vtc_drop,
                    training=   train_flag_PH,
                    seed=       seed)
                if verb > 1: print(' > dropout %.2f applied to vec_output of tower(s):' % vtc_drop, vec_output)

            # projection on vector to classifier
            if vtc_proj and vtc_proj!=vec_output.shape.as_list()[-1]:

                vec_output = lay_dense(
                    input=      vec_output,
                    units=      vtc_proj,
                    activation= None,
                    use_bias=    True,
                    seed=       seed,
                    name=       'inVProjection')
                if verb > 1: print(' > projected vector input:', vec_output)
                hist_summ.append(tf.summary.histogram('7vecTCProj', vec_output, family='C.cls'))

                # layerNorm (after projection)
                vec_output = tf.contrib.layers.layer_norm(
                    inputs=             vec_output,
                    begin_norm_axis=    -1,
                    begin_params_axis=  -1)
                hist_summ.append(tf.summary.histogram('8projLNorm', vec_output, family='C.cls'))

            mc_losses = []
            if verb > 1: print('\nBuilding multi-classifier graphs...')
            for cix in range(len(classes)):
                if verb > 1: print(' > multi-classifier (%d/%d):' % (cix + 1, len(classes)))

                # DRT encoder @classifier
                if drtC_nLay:
                    eDRTout = enc_DRT(
                        input=          vec_output,
                        n_layers=       drtC_nLay,
                        dns_scale=      drt_scale,
                        activation=     actv_func,
                        dropout=        drtC_drop,
                        training_flag=  train_flag_PH,
                        seed=           seed,
                        n_hist=            2,
                        verb=           verb)
                    vec_output =    eDRTout['output']
                    zsL +=          eDRTout['zeroes']

                if out_drop:
                    vec_output = tf.layers.dropout(
                        inputs=     vec_output,
                        rate=       out_drop,
                        training=   train_flag_PH,
                        seed=       seed)

                logits = lay_dense(
                    input=      vec_output,
                    units=      classes[cix],
                    activation= None,
                    use_bias=    True,
                    seed=       seed,
                    name=       'outProjection_cix%d'%cix)
                if verb > 1: print(' >> logits (projected)', logits)
                hist_summ.append(tf.summary.histogram('9logits', logits, family='C.cls'))

                probs = tf.nn.softmax(logits)
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                if verb > 1: print(' >> predictions:', predictions)
                correct = tf.equal(predictions, lab_PHL[cix])
                if verb > 1: print(' >> correct prediction:', correct)
                accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

                # softmax loss
                cLoss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lab_PHL[cix])
                if verb > 1: print(' > cLoss (softmax)', cLoss) # shape [batch] (per sample)

                """ TODO (experimental): scaled cLoss
                # scale cLoss
                scale = tf.where(
                    condition=  correct,
                    x=          tf.ones_like(correct, dtype=tf.float32)*tf.constant(0.8), # positive scale
                    y=          tf.ones_like(correct, dtype=tf.float32)*tf.constant(1.7)) # negative scale
                cLoss *= scale
                """

                mc_losses.append(cLoss)

        # average all losses (multi-classifiers losses)
        loss = tf.reduce_mean(tf.stack(mc_losses)) # shape [1]
        if verb > 1: print(' > loss (averaged all multi-classif)', loss)

        class_vars = tf.global_variables(scope=tf.get_variable_scope().name)  # vClass variables

        train_vars = []
        if train_tower: train_vars += tower_vars
        train_vars += class_vars

        # L2 cLoss
        if l2lc:
            restrictedNames = [
                'bias',         # dense bias
                'beta',         # LN offset
                'gamma',        # LN scale
                'tns_pos_emb',  # position embeddings
                'tok_emb']      # token embeddings
            if verb > 1: print(' > applying L2 cLoss to variables (not including %s)' % restrictedNames)
            l2Vars = []
            for v in train_vars:
                vIsOk = True
                for nmp in restrictedNames:
                    if nmp in v.name: vIsOk = False
                if vIsOk: l2Vars.append(v)
            if verb > 1:
                print(' > L2 / all(--) variables of model:')
                for var in train_vars:
                    if var in l2Vars:   print(' >> L2', var)
                    else:               print(' >> --', var)
            l2loss = tf.add_n([tf.nn.l2_loss(v) for v in l2Vars]) * l2lc # shape [1]
            if verb > 1: print(' > L2 cLoss', l2loss)
            loss += l2loss

    return {
        # placeholders
        'vec_PHL':          vec_PHL,
        'tks_PHL':          tks_PHL,
        'seq_PHL':          seq_PHL,
        'lab_PHL':          lab_PHL,
        'train_flag_PH':    train_flag_PH,
        # variables
        'train_vars':       train_vars,     # to train
        'tower_vars':       tower_vars,     # to save
        'class_vars':       class_vars,     # to save
        # tensors
        'probs':            probs,          # ...of last multi-classifier
        'predictions':      predictions,    # ...of last multi-classifier
        'accuracy':         accuracy,       # ...of last multi-classifier
        'loss':             loss,           # avg of all multi-classifiers

        'hist_summ':        tf.summary.merge(hist_summ),
        'zeroes':           zsL}


if __name__ == '__main__':

    from putils.neuralmess.dvc.presets import dvc_presets

    mdict = dvc_presets['dvc_base']
    mdict['verb'] = 2

    modelD = dvc_model(**mdict)