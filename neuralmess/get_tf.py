def get_tf():

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow
    tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
    if str(tensorflow.__version__)[0]=='2':
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
    else: import tensorflow as tf
    return tf

tf = get_tf()