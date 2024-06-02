import os
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from utils.error_handling import *


def extract_face(face, exit=True, required_size=(160, 160)):
    try:
        show_info("Extracting faces...")

        if os.path.isfile(face):
            image = Image.open(face)
        elif type(face) == np.ndarray:
            image = Image.fromarray(face)
        else:
            show_error("Face is neither a file nor an image array!")

        image = image.convert("RGB")
        pixels = asarray(image)
        detector = MTCNN()
        results = detector.detect_faces(pixels)

        x1, y1, width, height = results[0]["box"]
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]

        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array, results
    except:
        show_error("Face not found or error extracting face!", exit=exit)
        return None, None
