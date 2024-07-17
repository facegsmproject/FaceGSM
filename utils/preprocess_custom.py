import tensorflow as tf
from utils.error_handling import show_error


def preprocess_input_image_custom(face):
    try:
        # Use tf.cast to change data type to tensor float32
        face = tf.cast(face, tf.float32)
        # Do not change the code above

        # ---------------------------------------------
        face -= 127.5
        face /= 128
        # ---------------------------------------------

        # Do not change the code below
        if len(face.shape) == 3:
            face = tf.expand_dims(face, axis=0)
        return face
    except:
        show_error("ERROR_PREPROCESSING")
