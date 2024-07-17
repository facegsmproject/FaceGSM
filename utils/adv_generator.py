import cv2
import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import signal
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from utils.db_classifier import classify_face
from utils.preprocess_input import preprocess_input_image
from utils.face_extractor import *
from utils.process_image import *
from utils.checkpoint_delta import *

matplotlib.use("TkAgg")

EPS = 2 / 255.0
LR = 5e-3
TARGET_LOSS = -0.95

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LR)
loss_function = tf.keras.losses.CosineSimilarity()


def clip_eps(tensor, eps):
    # clip the values of the tensor to a given range
    return tf.clip_by_value(tensor, clip_value_min=-eps, clip_value_max=eps)


def adv(model, base_image, delta, target_embeddings, step=0):
    show_info("Starting Adversarial Attack...")
    handler = partial(signal_handler, delta=delta)
    signal.signal(signal.SIGINT, handler)
    while True:
        with tf.GradientTape() as tape:
            tape.watch(delta)
            adversary = preprocess_input_image(base_image + delta)
            predictions = model(adversary)
            # model give tf.tensor, model.predict give 2d array
            target_loss = loss_function(target_embeddings, predictions)
            # cosine similarity loss from tf = -sum(l2_norm(y_true) * l2_norm(y_pred)) OR -np.dot(A,B)/(norm(A)*norm(B))
        step += 1
        if step % 20 == 0 or step == 1:
            print("step: {}, loss: {}...".format(step, target_loss.numpy()))
        if (target_loss.numpy()) <= TARGET_LOSS:
            break

        gradients = tape.gradient(target_loss, delta)
        gradients = tf.sign(gradients)
        optimizer.apply_gradients([(gradients, delta)])
        delta.assign_add(clip_eps(delta, eps=EPS))
    return delta


def process_initial_input_image(path, role, required_size):
    face, _ = extract_face(path, required_size)
    show_image(face, f"{role} Image")
    constant = tf.constant(face, dtype=tf.float32)
    face = preprocess_input_image(face)
    show_image(face, f"Preprocessed {role} Image")
    return face, constant


def process_initial_input_image_live(initial_input, required_size):
    face, _ = extract_face(initial_input, required_size, exit=False)
    if face is None:
        return None, None
    constant = tf.constant(face, dtype=tf.float32)
    face = preprocess_input_image(face)
    return face, constant


def attack_adv(original_path, target_path, model, required_size, isCheckpoint=False):
    original_face, original_constant = process_initial_input_image(
        original_path, "Original", required_size
    )
    target_face, _ = process_initial_input_image(target_path, "Target", required_size)
    target_embeddings = model.predict(target_face)

    delta = tf.Variable(
        tf.zeros_like(original_constant), trainable=True, dtype=tf.float32
    )
    if isCheckpoint:
        delta = load_checkpoint(delta)

    start_time = time.perf_counter()
    perturbation_layer = adv(model, original_constant, delta, target_embeddings)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time}: seconds")

    save_checkpoint(perturbation_layer)
    show_save_perturbation_layer(perturbation_layer, "perturbation_layer")

    adv_image = (original_constant + perturbation_layer).numpy().squeeze()
    adv_image = np.clip(adv_image, 0, 255).astype("uint8")
    show_image(adv_image, "Adversarial Image")
    save_image(cv2.cvtColor(adv_image, cv2.COLOR_BGR2RGB), "adversarial_img")

    prediction_name, prediction_level, box = classify_face(
        adv_image, model, required_size, isAdv=True
    )

    _, box = extract_face(adv_image, required_size)  # take faces for rect_gen
    adv_image = cv2.cvtColor(adv_image, cv2.COLOR_RGB2BGR)
    rect_gen(
        prediction_name,
        prediction_level,
        adv_image,
        box,
        "adversarial_rect",
    )

    adv_image_preprocessed = preprocess_input_image(
        original_constant + perturbation_layer
    )
    show_image(adv_image_preprocessed, "Preprocessed Adversarial Image")

    original_embeddings = model.predict(original_face)
    adversarial_embeddings = model.predict(adv_image_preprocessed)

    show_info("Attack Finished...")
    cos_sim = cosine_similarity(original_embeddings, target_embeddings)[0][0]
    cos_sim = round(cos_sim * 100, 5)
    print(f"[+] Cosine Similarity between original and target embeddings:{cos_sim}%")
    # cos_sim = cosine_similarity(adversarial_embeddings, target_embeddings)[0][0]
    print(
        f"[+] Cosine Similarity between target and adversarial embeddings:{prediction_level}%"
    )

    return adv_image


def attack_adv_live(
    original_input, target_path, model, required_size, isFirstAttack, isCheckpoint=True
):
    original_face, original_constant = process_initial_input_image_live(
        original_input, required_size
    )

    if original_face is None:
        return "Unknown", "0.0"
    target_face, _ = process_initial_input_image_live(target_path, required_size)
    target_embeddings = model.predict(target_face)

    delta = tf.Variable(
        tf.zeros_like(original_constant), trainable=True, dtype=tf.float32
    )

    if os.path.isdir("./checkpoints") and len(os.listdir("./checkpoints")) != 0:
        delta = load_checkpoint(delta)
        perturbation_layer = delta

    print("Is First Attack:", isFirstAttack)
    if isFirstAttack:
        start_time = time.perf_counter()
        perturbation_layer = adv(model, original_constant, delta, target_embeddings)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time}: seconds")
        save_checkpoint(perturbation_layer)
        save_perturbation_layer(perturbation_layer, "perturbation_layer_live")

    adv_image = (original_constant + perturbation_layer).numpy().squeeze()
    adv_image = np.clip(adv_image, 0, 255).astype("uint8")
    save_image(cv2.cvtColor(adv_image, cv2.COLOR_BGR2RGB), "adversarial_img_live")

    prediction_name, prediction_level, box = classify_face(
        adv_image, model, required_size, isAdv=True, exit=False
    )

    adv_image_preprocessed = preprocess_input_image(
        original_constant + perturbation_layer
    )

    original_embeddings = model.predict(original_face)
    adversarial_embeddings = model.predict(adv_image_preprocessed)

    show_info("Attack Finished...")
    cos_sim = cosine_similarity(original_embeddings, target_embeddings)[0][0]
    cos_sim = round(cos_sim * 100, 5)
    print(f"[+] Cosine Similarity between original and target embeddings:{cos_sim}%")
    # cos_sim = cosine_similarity(adversarial_embeddings, target_embeddings)[0][0]
    print(
        f"[+] Cosine Similarity between target and adversarial embeddings:{prediction_level}%"
    )

    return prediction_name, prediction_level
