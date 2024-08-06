# Description: This file contains utility functions that are used in the main script. The check_gpu function checks if TensorFlow is built with CUDA support and if a GPU is available. This is useful for determining whether the GPU can be used for training deep learning models. The check_gpu function is called in the main script to provide information about the GPU availability.
import tensorflow as tf

def check_gpu():
    if tf.test.is_built_with_cuda():
        print("TensorFlow is built with CUDA support.")
        if tf.config.list_physical_devices('GPU'):
            print("GPU is available")
        else:
            print("GPU is not available, although TensorFlow is built with CUDA support")
    else:
        print("TensorFlow is not built with CUDA support")

check_gpu()