"""

 2019 (c) piteren

"""

import tensorflow as tf
from tensorflow.contrib import rnn

from ptools.neuralmess.base_elements import my_initializer
from ptools.neuralmess.layers import lay_dense


# N prediction decoder
def decN(
        input,
        dictW,
        predN=          1,          # N samples for every feature
        name=           'decN',
        hLays=          None,       # tuple or list of ints
        hActiv=         tf.nn.relu,
        initializer=    None,
        seed=           12321,
        verbLev=        0):

    if verbLev > 0: print('\nBuilding decoderN ...')
    if verbLev > 1: print('decoder input:', input)

    if initializer is None: initializer = my_initializer(seed)

    with tf.variable_scope(name):

        # hidden layers
        if hLays:
            for nLay in range(len(hLays)):
                laySize = hLays[nLay]
                input = lay_dense(
                    input=          input,
                    units=          laySize,
                    activation=     hActiv,
                    use_bias=       True,
                    initializer=    initializer,
                    seed=           seed,
                    name=           'decoderN_Hlay_%s' % nLay)

        # projection to predN x dictW
        logits = lay_dense(
            input=          input,
            units=          predN * dictW,
            activation=     None,
            use_bias=       True,
            initializer=    initializer,
            seed=           seed,
            name=           'decoderNProjection')
        if verbLev > 1: print(' > projection to logits (%dx dictW):' % predN, logits)

        if predN > 1:
            logits = tf.reshape(logits, [tf.shape(logits)[0], -1, dictW])
            if verbLev > 1: print(' > reshaped logits (B,%dxS,dictW):' % predN, logits)

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        if verbLev > 1: print(' > predictions:', predictions)

    return logits, predictions

# attention rnn decoder
def decARNN(
        input,
        target,
        embWeights,
        name=           'decARNN',
        cellFN=         rnn.NASCell,
        cellWidth=      64,
        attentionType=  'Luong',
        attMechSize=    72,             # att mechanism layer size (for Luong == cellWidth)
        attLaySize=     52,             # (opt) IN: concat(rnn+context), no bias => projectionL & rnnIN
        verbLev=        0):

    with tf.variable_scope(name):
        suppAttTypes = ['Luong', 'Bahdanau']
        assert attentionType in suppAttTypes, '>>> Err: unsupported attention type'
        attType = tf.contrib.seq2seq.BahdanauAttention
        if attentionType == 'Luong':
            attType = tf.contrib.seq2seq.LuongAttention
            attMechSize = cellWidth

        if verbLev > 1: print('decoder inputs:', input)
        if verbLev > 1: print('decoder targets:', target)
        targetsShape = target.shape

        # prepare targets with <go> token at the beginning
        goLabel = embWeights.shape[-1] - 1
        goLabelTensor = tf.fill([targetsShape[0], 1], goLabel)
        goTargets = tf.concat([goLabelTensor, target[:, :targetsShape[1] - 1]], -1)
        if verbLev > 1: print('decoder targets with go:', goTargets)

        # targets embedding for helper (decoder cell input)
        embTargets = tf.nn.embedding_lookup(params=embWeights, ids=goTargets)

        decoderCell = cellFN(cellWidth)
        attMechanism = attType(num_units=attMechSize,
                               memory=input)
        decoderCell = tf.contrib.seq2seq.AttentionWrapper(
            cell=                   decoderCell,
            attention_mechanism=    attMechanism,
            attention_layer_size=   attLaySize)

        tSEQlen = tf.fill(dims=[targetsShape[0]], value=targetsShape[1])
        if verbLev > 1: print('decoder tSEQlen:', tSEQlen)
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=         embTargets,
            sequence_length=tSEQlen)
        initialState = decoderCell.zero_state(targetsShape[0], tf.float32)
        projection_layer = tf.layers.Dense(
            units=          embWeights.shape[0],
            use_bias=        False,
            name=           'decoderProjection')
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=           decoderCell,
            helper=         helper,
            initial_state=  initialState,
            output_layer=   projection_layer)

        output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
        if verbLev > 1: print('decoder output:', output)

        logits = output.rnn_output
        labels = output.sample_id
    return logits, labels