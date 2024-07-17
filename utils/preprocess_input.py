import tensorflow as tf
from utils.error_handling import show_error
from utils.preprocess_custom import preprocess_input_image_custom


# Default preprocessing function (based on Facenet model) :
def preprocess_input_image(face, custom_preprocess):
    if custom_preprocess:
        return preprocess_input_image_custom(face)

    try:
        # Use tf.cast to change data type to tensor float32
        face = tf.cast(face, tf.float32)

        # Use tf.math functions for mean and standard deviation
        mean, std = tf.math.reduce_mean(face), tf.math.reduce_std(face)
        face = (face - mean) / std
        if len(face.shape) == 3:
            face = tf.expand_dims(face, axis=0)
        return face
    except:
        show_error("ERROR_PREPROCESSING")


# Preprocessing function for ArcFace model :
# def preprocess_input_image(face):
#     try:
#         # Use tf.cast to change data type to tensor float32
#         face = tf.cast(face, tf.float32)
#         face -= 127.5
#         face /= 128
#         if len(face.shape) == 3:
#             face = tf.expand_dims(face, axis=0)
#         return face
#     except:
#         show_error("ERROR_PREPROCESSING")
