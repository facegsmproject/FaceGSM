import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocess_input import preprocess_input_image
from utils.face_extractor import *
from utils.error_handling import *
from dotenv import load_dotenv

load_dotenv()

THRESHOLD = 60


def classify_face(face, model, required_size, isAdv=False, exit=True):
    database_path = os.getenv("DATABASE_PATH")

    if isAdv:
        original_face = face
        box = None
    else:
        original_face, box = extract_face(face, required_size, exit=exit)
        if original_face is None or box is None:
            return "Unknown", "0.0", box

    original_face = preprocess_input_image(original_face)
    original_embeddings = model.predict(original_face)

    with open(database_path, "r") as f:
        show_info("Classifying face...")

        highest_key, highest_result, i = "", 0, 0
        f = json.load(f)
        person = f["predictions"]
        for key, vector_embeddings in person.items():
            cos_sim = cosine_similarity(original_embeddings, vector_embeddings)[0][0]
            cos_sim = round(cos_sim * 100, 5)
            print(f"{key}: {cos_sim}%")
            i += 1
            if cos_sim >= THRESHOLD:
                if cos_sim >= highest_result:
                    highest_result = cos_sim
                    highest_key = key
            elif i == len(person) and highest_result == 0:
                highest_key = "Unknown"
                highest_result = "0.0"
                show_error("NO_MATCH_FOUND", exit=exit)
        return highest_key, highest_result, box
