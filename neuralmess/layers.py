"""

 2019 (c) piteren

"""

import numpy as np
import tensorflow as tf
from typing import List

from ptools.neuralmess.base_elements import my_initializer, flatten_LOTens

# residual (advanced) connection for (any) layer
def lay_res(
        lay_in,                     # layer input
        lay_out,                    # layer output
        name=           'residual',
        use_RCW=        False,      # use residual connection weights
        use_PWRCW=      False,      # pointwise weights
        match_dims=     True):      # concatenates zeros to input when thinner

    # TODO: not working for higher dimm tensors
    with tf.variable_scope(name):

        output = lay_out
        iW = int(lay_in.shape[-1])
        oW = int(output.shape[-1])
        matchedDims = iW == oW

        # pad input with zeros to match dimension of output
        if iW < oW and match_dims:
            lay_in = tf.pad(
                tensor=     lay_in,
                paddings=   tf.constant([[0,0],[0,oW-iW]]))
            matchedDims = True

        if matchedDims:
            if use_RCW:
                if use_PWRCW:   shape = [oW]
                else:           shape = []

                convRCW = tf.get_variable(
                    name=           'rcw',
                    shape=          shape,
                    initializer=    tf.constant_initializer(0))

                output = lay_in * (1 - tf.sigmoid(convRCW)) + output * tf.sigmoid(convRCW)
            else:
                output = lay_in + output

    return output

# returns [0,1] tensor: 1 where input not activated over whole batch, seq... (nane - Not Activated NEurons)
# we assume that value =< 0 means: not activated
# we check over all axes but last (feats)
def zeroes(input :tf.Tensor) -> List[tf.Tensor]:
    axes = [ix for ix in range(len(input.shape))][:-1]      # axes all but last(feats) axes list
    activated = tf.where(                                   # 1 for value greater than zero, other 0
        condition=  tf.math.greater(input, 0),
        x=          tf.ones_like(input), # true
        y=          tf.zeros_like(input)) # false
    activated_reduced = tf.reduce_sum(activated, axis=axes) #1 or more for activated, 0 for not activated
    not_activated = tf.equal(activated_reduced, 0)          # true where summed gives zero (~invert)
    nn_zeros = tf.cast(not_activated, dtype=tf.int8)        # cast to 1 where summed gives zero
    return nn_zeros

# dense layer
def lay_dense(
        input,
        units :int,                 # layer width
        name=           'dense',
        reuse=          False,
        activation=     None,
        use_bias=       True,
        initializer=    None,
        seed=           12321):

    if initializer is None: initializer = my_initializer(seed)
    denseLay = tf.layers.Dense(
        units=              units,
        activation=         activation,
        use_bias=           use_bias,
        kernel_initializer= initializer,
        name=               name,
        _reuse=             reuse)
    output = denseLay(input)
    return output

# 1d convolution layer, with Gated Linear Unit option
def lay_conv1D(
        input,
        name=           'conv1D',
        kernels=        (3,5,7),        # layer kernels
        filters=        (36,12,6),      # int divisible by len(kernels) or tuple of len(kernels)
        dilation=       1,
        activation=     None,
        use_bias=       True,
        gated_LU=       False,          # Gated Linear Unit architecture
        initializer=    None,
        padding=        'valid',        # 'same' adds padding, 'valid' does not
        seed=           12321,
        verb=           0):

    if initializer is None: initializer = my_initializer(seed)
    with tf.variable_scope(name):
        sub_out_list = []
        if type(kernels) is not tuple: kernels = (kernels,)
        if verb > 1:
            print(' > %s: kernels %s, filters %s, dilation %s' % (name, kernels, filters, dilation))
        for k in range(len(kernels)):
            with tf.variable_scope('kernel_%d' % k):
                sub_kernel = kernels[k]
                if type(filters) is not tuple:  sub_filters = filters // len(kernels)
                else:                           sub_filters = filters[k]
                if gated_LU: sub_filters *= 2

                conv_lay = tf.layers.Conv1D(
                    filters=            sub_filters,
                    kernel_size=        sub_kernel,
                    dilation_rate=      dilation,
                    activation=         None,
                    use_bias=           use_bias,
                    kernel_initializer= initializer,
                    padding=            padding,
                    data_format=        'channels_last')
                sub_output = conv_lay(input)

                if verb > 1: print(' >> sub_conv: filters %s, kernel %s' % (sub_filters, sub_kernel))
                sub_out_list.append(sub_output)

        output = tf.concat(sub_out_list, axis=-1)
        if gated_LU:
            s1, s2 = tf.split(output, num_or_size_splits=2, axis=-1)
            output = s1 * tf.sigmoid(s2)
        elif activation: output = activation(output)

    return output

# 2d convolution layer
def lay_conv2D(
        input,
        name=           'conv2d',
        kernels=        (3, 5, 7),      # layer kernels
        filters=        (36, 12, 6),    # int divisible by len(kernels) or tuple of len(kernels)
        dilation=       1,
        activation=     None,
        useBias=        True,
        gatedLU=        False,          # Gated Linear Unit architecture
        initializer=    None,
        seed=           12321,
        verbLev=        0):

    if initializer is None: initializer = my_initializer(seed)
    with tf.variable_scope(name):
        variables = []
        subOutList = []
        if type(kernels) is not tuple: kernels = (kernels,)
        if verbLev > 0:
            print(' > %s: kernels %s, filetrs %s, dilation %s' % (name, kernels, filters, dilation))
        for k in range(len(kernels)):
            with tf.variable_scope('kernel_%d' % k):
                subKernel = kernels[k]
                if type(filters) is not tuple:
                    subFilters = filters / len(kernels)
                else:
                    subFilters = filters[k]
                if gatedLU: subFilters *= 2

                convLay = tf.layers.Conv2D(
                    filters=            subFilters,
                    kernel_size=        subKernel,
                    dilation_rate=      dilation,
                    activation=         None,
                    use_bias=           useBias,
                    kernel_initializer= initializer,
                    padding=            'valid',
                    data_format=        'channels_last')
                subOutput = convLay(input)
                for var in convLay.variables: variables.append(var)

                if verbLev > 1: print(' >> subConv: filters %s, kernel %s' % (subFilters, subKernel))
                subOutList.append(subOutput)

        output = tf.concat(subOutList, axis=-1)
        if gatedLU:
            s1, s2 = tf.split(output, num_or_size_splits=2, axis=-1)
            output = s1 * tf.sigmoid(s2)
        else:
            if activation: output = activation(output)

        variables = flatten_LOTens(variables)

    return output, variables

# attention for Query, Key and Value
def attn(
        q,                  # decides about output shape
        k,
        v,
        dropout=    0.0,
        drop_flag=  None,
        seed=       12321):

    w = tf.matmul(q, k, transpose_b=True)                   # q*kT
    w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))   # scale by 1/sqrt(v.dim[-1])
    w = tf.nn.softmax(w)
    if dropout:
        w = tf.layers.dropout(
            inputs=     w,
            rate=       dropout,
            training=   drop_flag,
            seed=       seed)
    att = tf.matmul(w, v)

    return {
        'attention':    att,
        'att_weights':  w}

# time & feats dropout (for sequences)
def tf_drop(
        input,              # tensor [batch,seq,feats]
        time_drop :float,
        feat_drop :float,
        train_flag,         # tensor
        seed=       12321):

    output = input
    in_shape = tf.shape(input)

    # time (per vector) dropout
    if time_drop:
        t_drop = tf.ones(shape=[in_shape[-3], in_shape[-2]])
        t_drop = tf.layers.dropout(
            inputs=     t_drop,
            rate=       time_drop,
            training=   train_flag,
            seed=       seed)
        output *= tf.expand_dims(t_drop, axis=-1)

    # feature (constant in time) dropout
    if feat_drop:
        f_drop = tf.ones(shape=[in_shape[-3], in_shape[-1]])
        f_drop = tf.layers.dropout(
            inputs=     f_drop,
            rate=       feat_drop,
            training=   train_flag,
            seed=       seed)
        output *= tf.expand_dims(f_drop, axis=-2)

    return output

# positional encoding layer
def positional_encoding(
        positions :int,         # max number of positions to encode
        width :int,             # width of positions vector
        min_pi_range=       1,
        max_pi_range=       10,
        as_numpy=           True,
        verb=               0):

    angle_rates = np.linspace(min_pi_range/max_pi_range, 1, num=width)
    if verb > 0: print(f'\ni.linspace\n{angle_rates}')
    angle_rates = angle_rates[np.newaxis, :]
    if verb > 0: print(f'\nangle_rates.new_axis\n{angle_rates}')

    pos = np.arange(positions)[:, np.newaxis]
    if verb > 0: print(f'\npos.arange.newaxis\n{pos}')
    pos = pos / positions * max_pi_range
    if verb > 0: print(f'\npos.scaled to range\n{pos}')
    angle_rads = pos * angle_rates
    if verb > 0: print(f'\nangle_rads {angle_rads.shape}\n{angle_rads}')

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2] * np.pi)
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2] * np.pi)

    pos_encoding = angle_rads[np.newaxis, ...]

    pos_encoding = pos_encoding - pos_encoding.mean()

    if as_numpy: pos_encoding = pos_encoding.astype(dtype=np.float32)
    else: pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
    return pos_encoding