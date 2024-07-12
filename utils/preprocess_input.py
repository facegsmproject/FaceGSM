import tensorflow as tf
from utils.error_handling import show_error


# Default preprocessing function (based on Facenet model) :
def preprocess_input_image(face):
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


# Placeholder for custom preprocessing function :
# 1. Comment the code above
# 2. Uncomment the code below
# 3. Implement your custom preprocessing function in the space provided

# def preprocess_input_image(face):
    # face = tf.cast(face, tf.float32)  # Do not change this line
    # ---------------------------------------------
    # Code your custom preprocessing function here
    # ---------------------------------------------
    # return face
