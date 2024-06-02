import asyncio
import numpy as np

import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from dotenv import load_dotenv
from utils.db_classifier import classify_face
from utils.adv_generator import attack_adv
from utils.process_image import *
from utils.error_handling import *

load_dotenv()


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class LiveCameraClient:
    def __init__(self, IP_DroidCam, model_path, isCheckpoint):

        self.init_tk()
        self.init_droidcam(IP_DroidCam)

        self.model_path = model_path
        self.isCheckpoint = isCheckpoint

        self.init_canvas()

        # self.update()
        # self.window.mainloop()

    async def initialize(self):
        self.reader, self.writer = await asyncio.open_connection("127.0.0.1", 8888)
        prediction_result = asyncio.Queue()
        initial_flag = True

        self.ret, self.frame = self.vid.read()

        camera_feed_task = asyncio.create_task(self.update())

        self.window.mainloop()

        server_comm_task = asyncio.create_task(
            self.handle_server_communication(self.frame)
        )

        # if initial_flag:
        #     initial_flag = False
        #     frame_bytes = self.frame.tobytes()
        #     frame_size = len(frame_bytes)
        #     print("Sending initial frame:", self.frame)
        #     # Harusnya ngambil db classifier ke utils.db_classifier, harusnya async juga
        #     person_name, confidence_level = await self.send_to_server(
        #         self.model_path, frame_size, frame_bytes
        #     )
        #     await prediction_result.put((person_name, confidence_level))

        await asyncio.gather(camera_feed_task, server_comm_task)

        writer.close()
        await writer.wait_closed()

    async def handle_server_communication(self, frame):
        frame_bytes = self.frame.tobytes()
        frame_size = len(frame_bytes)
        person_name, confidence_level = await self.send_to_server(
            self.model_path, frame_size, frame_bytes
        )
        print("client rect", person_name, confidence_level)
        await prediction_result.put((person_name, confidence_level))

    async def send_to_server(self, model_path, frame_size, frame_bytes):
        self.writer.write(f"{model_path}\n".encode())
        await self.writer.drain()
        self.writer.write(f"{frame_size}\n".encode())
        await self.writer.drain()
        self.writer.write(frame_bytes)
        await self.writer.drain()

        # server return 2 values, person_name and confidence_level

        data = await self.reader.readuntil(separator=b"\n")
        person_name = data.decode().strip()

        data = await self.reader.readuntil(separator=b"\n")
        confidence_level = data.decode().strip()
        # server need to calculate data size and the amount of data to be send
        print("Received from server:", person_name, confidence_level)
        return person_name, confidence_level

    def init_tk(self):
        window = tk.Tk()
        self.window = window
        window_title = "FaceGSM"
        self.window.title(window_title)

    def init_droidcam(self, IP_DroidCam):
        IP_DroidCam = "https://" + IP_DroidCam + ":4343/video"
        self.video_source = IP_DroidCam
        self.vid = cv2.VideoCapture(self.video_source)

    def init_canvas(self):
        print("canvasssssssssssssssssssss")
        self.canvas = tk.Canvas(
            self.window,
            width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        self.canvas.pack()

        # self.btn_attack = tk.Button(
        #     window, text="Attack", width=50, command=self.attack
        # )
        # self.btn_attack.pack(anchor=tk.CENTER, expand=True)

        # def toggle():
        # if button.config('text')[-1] == 'OFF':
        # 	button.config(text='ON', bg='green')
        # else:
        # 	button.config(text='OFF', bg='red')
        # button = tk.Button(root, text='OFF', width=12, command=toggle, bg='red'

        # Bind keyboard shortcut to snapshot function
        # self.window.bind("a", self.attack)
        self.window.bind("q", self.window.quit())

    # Feed Image to Tk Interface
    async def update(self):
        ret, frame = self.vid.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces with Haar Cascade
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            frame = rect_gen_live(frame, faces)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update)

    def process_frame(self, role):
        ret, frame = self.vid.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            person_name, confidence_level, faces = classify_face(frame_rgb, self.model)
            save_image(frame, role)
            role_rect = role + "_rect"
            rect_gen(person_name, confidence_level, frame, faces, role_rect)

    # def attack(self, keybind=None):
    #     show_info("Attacking Original to Target...")
    #     original_path = os.getenv("ORIGINAL_IMAGE_PATH")
    #     target_path = os.getenv("TARGET_IMAGE_PATH")

    #     self.window.destroy()
    #     self.vid.release()

    #     attack_adv(original_path, target_path, self.model, self.isCheckpoint)


# async def print_every_second():
#     for i in range(1, 8):
#         print(f"Waiting for server response... {i} sec")
#         await asyncio.sleep(1)


# def generate_data():
#     message = np.random.rand(100, 100)
#     message_bytes = message.tobytes()
#     message_size = len(message_bytes)
#     return message, message_bytes, message_size


# async def show_camera_feed(confidence_level_queue):
#     while True:
#         # Retrieve the latest confidence level if available
#         if not confidence_level_queue.empty():
#             confidence_level = await confidence_level_queue.get()
#             print(f"Updated confidence level: {confidence_level}")
#         print(f"client rect {confidence_level}")
#         await asyncio.sleep(1)


# async def send_data(reader, writer, message_bytes, message_size):
#     writer.write(f"{message_size}\n".encode())
#     await writer.drain()
#     writer.write(message_bytes)
#     await writer.drain()

#     data = await reader.read(1024)
#     print("Received from server:", data.decode())
#     return data.decode()


# async def handle_server_communication(reader, writer, confidence_level_queue):
#     print("handle_server_communication")
#     while True:
#         message, message_bytes, message_size = generate_data()
#         print("Sending:", message)

#         confidence_level = await send_data(reader, writer, message_bytes, message_size)
#         print("confidence_level:", confidence_level)

#         # Put the updated confidence level in the queue
#         await confidence_level_queue.put(confidence_level)


# async def main():
#     reader, writer = await asyncio.open_connection("127.0.0.1", 8888)

#     confidence_level_queue = asyncio.Queue()
#     init_flag = True

#     if init_flag:
#         init_flag = False
#         message, message_bytes, message_size = generate_data()
#         print("Sending initial data:", message)
#         initial_confidence_level = await send_data(
#             reader, writer, message_bytes, message_size
#         )
#         await confidence_level_queue.put(initial_confidence_level)

#     camera_feed_task = asyncio.create_task(show_camera_feed(confidence_level_queue))

#     # Start the server communication task
#     server_comm_task = asyncio.create_task(
#         handle_server_communication(reader, writer, confidence_level_queue)
#     )

#     # Wait for both tasks to complete
#     await asyncio.gather(camera_feed_task, server_comm_task)

#     writer.close()
#     await writer.wait_closed()


# asyncio.run(main())
