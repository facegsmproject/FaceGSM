import cv2
import asyncio
from dotenv import load_dotenv
from utils.process_image import rect_gen_live
import mediapipe as mp
from pynput import keyboard

load_dotenv()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
)


class LiveCameraClient:
    def __init__(self, target_path, URL_DROIDCAM, model_path, required_size):
        self.init_droidcam(URL_DROIDCAM)
        self.ret, self.frame = self.vid.read()

        self.model_path = model_path
        self.required_size = required_size
        self.target_path = target_path
        self.isAttack = False
        self.isFirstAttack = True

    async def initialize(self):

        self.reader, self.writer = await asyncio.open_connection("127.0.0.1", 8888)
        self.prediction_result = asyncio.Queue()

        live_feed_task = asyncio.create_task(self.update_frame(self.prediction_result))
        server_task = asyncio.create_task(self.process_frame_to_be_sent(self.frame))
        key_task = asyncio.create_task(self.check_a_key())

        await asyncio.gather(live_feed_task, server_task, key_task)

        self.writer.close()
        await self.writer.wait_closed()

    async def process_frame_to_be_sent(self, frame):
        while True:
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
        self.writer.write(f"{self.isFirstAttack}\n".encode())
        await self.writer.drain()
        self.writer.write(f"{frame_size}\n".encode())
        await self.writer.drain()
        self.writer.write(frame_bytes)
        await self.writer.drain()
        self.writer.write(f"{self.required_size}\n".encode())
        await self.writer.drain()

        if self.isAttack and self.isFirstAttack:
            self.isFirstAttack = False

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
                image_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                height, width, _ = self.frame.shape
                result = face_mesh.process(image_rgb)
                if result.multi_face_landmarks:
                    for face_landmarks in result.multi_face_landmarks:
                        x_coords = [
                            landmark.x * width for landmark in face_landmarks.landmark
                        ]
                        y_coords = [
                            landmark.y * height for landmark in face_landmarks.landmark
                        ]

                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)

                        w = max_x - min_x
                        h = max_y - min_y

                        x = int(min_x)
                        y = int(min_y)
                        w = int(w)
                        h = int(h)

                        faces = [x, y, w, h]

                    frame = rect_gen_live(
                        self.frame,
                        faces,
                        person_name,
                        confidence_level,
                        self.required_size,
                    )
                    cv2.imshow("FaceGSM", frame)
                else:
                    cv2.imshow("FaceGSM", self.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    asyncio.get_event_loop().stop()
                    break
                    # elif cv2.waitKey(1) & 0xFF == ord("a"):
                    #     self.isAttack = not self.isAttack
                    #     print("Attacking:", self.isAttack)
                    await asyncio.sleep(0.1)
            await asyncio.sleep(0.01)

        self.vid.release()
        cv2.destroyAllWindows()

    async def check_a_key(self):
        def on_press(key):
            try:
                if key.char == "a":
                    self.isAttack = not self.isAttack
                    print("Attacking:", self.isAttack)
            except AttributeError:
                pass

        def on_release(key):
            pass

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        while True:
            await asyncio.sleep(0.1)
