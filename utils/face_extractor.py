import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from utils.error_handling import *


def process_image_facemesh(face, exit=True):

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    try:
        show_info("Finding Facial Landmarks using FaceMesh...")

        if os.path.isfile(face):
            face = cv2.imread(face)
        elif type(face) == np.ndarray:
            face = face
        else:
            show_error("IMAGE_NEITHER_FILE_NOR_IMAGE_ARRAY")

        image_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            show_error("FACE_NOT_FOUND", exit=exit)
            return None, None
        return results, image_rgb
    except Exception as e:
        print(e)
        show_error("FACE_NOT_FOUND", exit=exit)
        return None, None


def extract_face(image, required_size, exit=True):
    results, face = process_image_facemesh(image, exit=exit)
    width, height = required_size

    if results is None:
        return None, None
    elif results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mask = np.zeros(face.shape[:2], dtype=np.uint8)
            points = []

            for landmark in face_landmarks.landmark:
                x = int(landmark.x * face.shape[1])
                y = int(landmark.y * face.shape[0])
                points.append([x, y])

            points = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

            white_background = np.ones_like(face, dtype=np.uint8) * 255

            face_image = cv2.bitwise_and(face, face, mask=mask)
            background = cv2.bitwise_and(
                white_background, white_background, mask=cv2.bitwise_not(mask)
            )
            result = cv2.add(face_image, background)

            x, y, w, h = cv2.boundingRect(hull)

            cropped_face = result[y : y + h, x : x + w]

            aspect_ratio = w / h
            if aspect_ratio > 1:
                new_w = width
                new_h = int(height / aspect_ratio)
            else:
                new_h = height
                new_w = int(width * aspect_ratio)

            resized_face = cv2.resize(cropped_face, (new_w, new_h))

            padded_face = np.ones((height, width, 3), dtype=np.uint8) * 255

            x_offset = (width - new_w) // 2
            y_offset = (height - new_h) // 2

            padded_face[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
                resized_face
            )

            box = [x, y, w, h]

            return padded_face, box
