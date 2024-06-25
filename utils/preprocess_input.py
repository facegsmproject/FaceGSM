import tensorflow as tf
from utils.error_handling import show_error


def preprocess_input_facenet(face):
    try:
        face = tf.cast(face, tf.float32)  # Use tf.cast to change data type
        mean, std = tf.math.reduce_mean(face), tf.math.reduce_std(
            face
        )  # Use tf.math functions for mean and standard deviation
        face = (face - mean) / std
        if len(face.shape) == 3:
            face = tf.expand_dims(face, axis=0)
        return face
    except:
        show_error("ERROR_PREPROCESSING")