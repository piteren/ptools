import numpy as np
import tensorflow as tf


# runs tflite model with list of data samples
def run_tflite(
        tflite_path,
        in_data :list=  None,
        verb=           1):

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    num_inputs = len(input_details)
    if verb>0:
        print('Tflite model loaded')
        print(f' > gots {num_inputs} inputs')
        for inp in input_details: print(f' >> {inp}')
        print(f' > gots {len(output_details)} outputs')
        for out in output_details: print(f' >> {out}')

    # Test model on random input data.
    if not in_data:
        sample = []
        for i in range(num_inputs):
            input_shape = input_details[i]['shape']
            print(input_shape)
            input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
            sample.append(input_data)
        in_data = [sample]

    outs = []
    assert len(in_data[0])==num_inputs, 'ERR: number of inputs do not match'
    for inp in in_data:
        for i in range(num_inputs):
            interpreter.set_tensor(input_details[i]['index'], inp[i])

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = output_data[0]
        output_data = [f for f in output_data]
        outs.append(output_data)
    return outs