import os
import warnings
import logging

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import asyncio
import numpy as np
import random
import dotenv
import cv2
from dotenv import load_dotenv
from keras.models import load_model
from utils.db_classifier import classify_face
from utils.adv_generator import attack_adv_live
from utils.error_handling import *


load_dotenv()


async def send_to_client(writer, person_name, prediction_level):
    writer.write(f"{person_name}\n".encode())
    await writer.drain()
    writer.write(f"{prediction_level}\n".encode())
    await writer.drain()


async def attack(original_frame, target_path, model, required_size, isFirstAttack, writer):
    show_info("Attacking Original to Target...")

    person_name, prediction_level = attack_adv_live(original_frame, target_path, model, required_size, isFirstAttack)
    print("Person name:", person_name)
    print("Prediction level:", prediction_level)
    await send_to_client(writer, person_name, prediction_level)


async def classify(frame, model, required_size, writer):
    person_name, prediction_level, _ = classify_face(frame, model, required_size, exit=False)
    print("Person name:", person_name)
    print("Prediction level:", prediction_level)
    await send_to_client(writer, person_name, prediction_level)


async def handle_client(reader, writer):
    try:
        while True:
            data = await reader.readuntil(separator=b"\n")
            model_path = data.decode().strip()
            model = load_model(model_path)

            data = await reader.readuntil(separator=b"\n")
            target_path = data.decode().strip()

            data = await reader.readuntil(separator=b"\n")
            isAttack = data.decode().strip()

            data = await reader.readuntil(separator=b"\n")
            isFirstAttack = data.decode().strip()
            isFirstAttack = True if isFirstAttack == "True" else False

            data = await reader.readuntil(separator=b"\n")
            frame_size = int(data.decode().strip())

            frame_bytes = await reader.readexactly(frame_size)
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((480, 640, 3))

            data = await reader.readuntil(separator=b"\n")
            required_size = data.decode().strip()
            required_size = required_size[1:-1].split(", ")
            required_size = (int(required_size[0]), int(required_size[1]))

            cv2.imwrite("./outputs/server_frame.jpg", frame)

            if isAttack == "True":
                await attack(frame, target_path, model, required_size, isFirstAttack, writer)
            else:
                await classify(frame, model, required_size, writer)

    except asyncio.IncompleteReadError:
        show_error("CLIENT_DISCONNECTED")

    except Exception as e:
        print("Error:", e)

    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    server = await asyncio.start_server(handle_client, "127.0.0.1", 8888)
    show_info("Server started...")
    async with server:
        await server.serve_forever()


asyncio.run(main())
