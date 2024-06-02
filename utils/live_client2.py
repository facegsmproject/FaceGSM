import tkinter as tk
from PIL import Image, ImageTk
import cv2
import asyncio
import aiofiles
import aiohttp

from utils.db_classifier import classify_face
from utils.adv_generator import attack_adv
from utils.process_image import *
from utils.error_handling import *

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # Open video source
        self.vid = cv2.VideoCapture(video_source)

        # Create a canvas that can fit the video source size
        self.canvas = tk.Canvas(
            window,
            width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        self.canvas.pack()

        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Initialize asyncio event loop
        self.loop = asyncio.get_event_loop()
        self.running = True

        # Start the async tasks
        self.loop.create_task(self.update())
        self.loop.create_task(self.server_communication())

        # Handle window closing
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.periodic_call()

        # Placeholder for frame
        self.frame = None

        # Placeholder for asyncio queue
        self.prediction_result = asyncio.Queue()

        # Placeholder for model path (set appropriately)
        self.model_path = "path_to_your_model"

    def periodic_call(self):
        self.loop.call_soon(self.loop.stop)
        self.loop.run_forever()
        if self.running:
            self.window.after(10, self.periodic_call)

    async def update(self):
        while self.running:
            ret, frame = self.vid.read()
            if ret:
                self.frame = frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5
                )
                frame = self.rect_gen_live(frame, faces)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            await asyncio.sleep(0.01)

    async def server_communication(self):
        while self.running:
            if self.frame is not None:
                await self.handle_server_communication(self.frame)
            await asyncio.sleep(0.01)

    async def handle_server_communication(self, frame):
        frame_bytes = frame.tobytes()
        frame_size = len(frame_bytes)
        person_name, confidence_level = await self.send_to_server(
            self.model_path, frame_size, frame_bytes
        )
        print("client rect", person_name, confidence_level)
        await self.prediction_result.put((person_name, confidence_level))

    async def send_to_server(self, model_path, frame_size, frame_bytes):
        self.writer.write(f"{model_path}\n".encode())
        await self.writer.drain()
        self.writer.write(f"{frame_size}\n".encode())
        await self.writer.drain()
        self.writer.write(frame_bytes)
        await self.writer.drain()

    def rect_gen_live(self, frame, faces):
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    def on_closing(self):
        self.running = False
        self.vid.release()
        self.loop.stop()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, "Tkinter and OpenCV")
    root.mainloop()
