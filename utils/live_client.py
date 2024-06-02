import asyncio
import numpy as np

import os
import cv2
from PIL import Image
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
    def __init__(self, target_path, URL_DROIDCAM, model_path):
        # URL_DROIDCAM = 'http://192.168.1.4:4747/video'
        self.init_droidcam(URL_DROIDCAM)
        self.ret, self.frame = self.vid.read()

        self.model_path = model_path
        self.target_path = target_path
        self.isAttack = False

    async def initialize(self):

        self.reader, self.writer = await asyncio.open_connection("127.0.0.1", 8888)
        self.prediction_result = asyncio.Queue()

        camera_feed_task = asyncio.create_task(
            self.update_frame(self.prediction_result)
        )
        server_task = asyncio.create_task(self.process_frame_to_be_sent(self.frame))

        await asyncio.gather(camera_feed_task, server_task)

        self.writer.close()
        await self.writer.wait_closed()

    async def process_frame_to_be_sent(self, frame):
        while True:
            # if cv2.waitKey(1) & 0xFF == ord("a"):
            #     break
            frame_bytes = self.frame.tobytes()
            frame_size = len(frame_bytes)

            person_name, confidence_level = await self.send_to_server(
                self.model_path, self.target_path, frame_size, frame_bytes
            )
            await self.prediction_result.put((person_name, confidence_level))

    async def send_to_server(self, model_path, target_path, frame_size, frame_bytes):
        self.writer.write(f"{model_path}\n".encode())
        await self.writer.drain()
        self.writer.write(f"{target_path}\n".encode())
        await self.writer.drain()
        self.writer.write(f"{self.isAttack}\n".encode())
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

        print("Received from server:", person_name, confidence_level)
        return person_name, confidence_level

    def init_droidcam(self, URL_DROIDCAM):
        self.video_source = URL_DROIDCAM
        self.vid = cv2.VideoCapture(self.video_source)

    async def update_frame(self, prediction_result):
        person_name, confidence_level = "Unknown", "0.0"
        while True:
            if not prediction_result.empty():
                person_name, confidence_level = await prediction_result.get()
                print(
                    f"person_name: {person_name}, confidence_level: {confidence_level}"
                )
            ret, self.frame = self.vid.read()
            if ret:
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                # Detect faces with Haar Cascade
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                frame = rect_gen_live(self.frame, faces, person_name, confidence_level)
                cv2.imshow("FaceGSM", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    asyncio.get_event_loop().stop()
                    break
                elif cv2.waitKey(1) & 0xFF == ord("a"):
                    self.isAttack = not self.isAttack
                    print("Attacking:", self.isAttack)
                    await asyncio.sleep(0.1)
            await asyncio.sleep(0.01)

        self.vid.release()
        cv2.destroyAllWindows()
