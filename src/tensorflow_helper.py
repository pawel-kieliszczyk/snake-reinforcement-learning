import tensorflow as tf


def setup_gpu_memory_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if not gpus:
        return

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

