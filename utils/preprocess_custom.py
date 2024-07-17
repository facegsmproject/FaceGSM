import tensorflow as tf
from utils.error_handling import show_error


def preprocess_input_image_custom(face):
    try:
        # Use tf.cast to change data type to tensor float32
        face = tf.cast(face, tf.float32)  # Do not change this line
        # ---------------------------------------------
        # Code your custom preprocessing function here
        # ---------------------------------------------
        return face
    except:
        show_error("ERROR_PREPROCESSING")
