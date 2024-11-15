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


class VideoCaptureApp:
    def __init__(self, target_path, URL_DROIDCAM, model, isCheckpoint, required_size):
        self.window = tk.Tk()
        self.window.title("FaceGSM")
        self.video_source = URL_DROIDCAM
        self.vid = cv2.VideoCapture(self.video_source)
        self.required_size = required_size

        self.model = model
        self.isCheckpoint = isCheckpoint

        self.canvas = tk.Canvas(
            self.window,
            width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        self.canvas.pack()

        self.btn_process_original = tk.Button(
            self.window,
            text="Capture Original",
            width=50,
            command=self.process_frame_original,
        )
        self.btn_process_original.pack(anchor=tk.CENTER, expand=True)
        self.btn_attack = tk.Button(
            self.window, text="Attack", width=50, command=self.attack
        )
        self.btn_attack.pack(anchor=tk.CENTER, expand=True)

        self.window.bind("o", self.process_frame_original)
        self.process_frame_target(target_path)
        self.window.bind("a", self.attack)
        self.window.bind("q", lambda e: self.on_closing())

        self.update_id = None
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.update_id = self.window.after(10, self.update)

    def process_frame_original(self, keybind=None):
        role = "original"
        ret, frame = self.vid.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            person_name, confidence_level, box = classify_face(
                frame_rgb,
                self.model,
                self.required_size,
                exit=False,
            )
            save_image(frame, role)
            role_rect = role + "_rect"
            rect_gen(person_name, confidence_level, frame, box, role_rect)

    def process_frame_target(self, target_path):
        role = "target"
        frame = cv2.imread(target_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        person_name, confidence_level, box = classify_face(
            frame_rgb,
            self.model,
            self.required_size
        )
        save_image(frame, role)
        role_rect = role + "_rect"
        rect_gen(person_name, confidence_level, frame, box, role_rect)

    def attack(self, keybind=None):
        show_info("Attacking Original to Target...")
        original_path = os.getenv("ORIGINAL_IMAGE_PATH")
        target_path = os.getenv("TARGET_IMAGE_PATH")

        self.on_closing()

        attack_adv(
            original_path,
            target_path,
            self.model,
            self.required_size,
            self.isCheckpoint,
        )

    def on_closing(self):
        if self.update_id is not None:
            self.window.after_cancel(self.update_id)
        self.vid.release()
        self.window.destroy()
