import os
import json
import numpy as np
from utils.preprocess_input import preprocess_input_facenet
from utils.mtcnn_extractor import *
from utils.error_handling import *

predictions_dict = {}


def create_json(folder_path, model):
    show_info("Creating JSON database...")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        show_info(f"Processing {file_path}")
        if os.path.isfile(file_path):
            face, _ = extract_face(file_path)
            face = preprocess_input_facenet(face)

            predictions = model.predict(face)
            predictions = predictions.tolist()

            person_name = os.path.splitext(filename)[0]
            predictions_dict[person_name] = predictions
        else:
            show_error("FILE_INVALID")

    show_info("Writing JSON database...")
    output = {"predictions": predictions_dict}
    output_json = json.dumps(output)
    with open("./databases/database.json", "w") as f:
        f.write(output_json)
