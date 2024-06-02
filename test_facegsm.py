import os
import sys
from ipaddress import IPv4Address
from utils.interface import VideoCaptureApp
from utils.live_client import LiveCameraClient
import signal
from dotenv import load_dotenv
from keras.models import load_model
from utils.db_generator import create_json
from utils.adv_generator import attack_adv
from utils.ascii_art import ascii_art
from utils.error_handling import *

import asyncio

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow warnings

load_dotenv()
    
def check_argv_folder(arg, i, argv):
    i += 1
    if i < len(argv):
        if os.path.isdir(argv[i]):
            if len(os.listdir(argv[i])) == 0:
                show_error_arg(os.getenv("EMPTY_FOLDER"), arg)
                sys.exit()
            else:
                return i, argv[i]
        else:
            show_error_arg(os.getenv("FOLDER_INVALID"), arg)
            sys.exit()
    else:
        show_error_arg(os.getenv("NO_VALUE_PROVIDED"), arg)
        sys.exit()


def check_argv_file(arg, i, argv):
    i += 1
    if i < len(argv):
        if os.path.isfile(argv[i]):
            return i, argv[i]
        else:
            show_error_arg(os.getenv("FILE_INVALID"), arg)
            sys.exit()
    else:
        show_error_arg(os.getenv("NO_VALUE_PROVIDED"), arg)
        sys.exit()


def check_argv_ip(arg, i, argv):
    i += 1
    if i < len(argv):
        try:
            IPv4Address(argv[i])
        except ValueError:
            show_error_arg(os.getenv("IPv4_INVALID"), arg)
            sys.exit()
        return i, argv[i]
    else:
        show_error_arg(os.getenv("NO_VALUE_PROVIDED"), arg)
        sys.exit()


def main():
    model_path = os.getenv("DEFAULT_MODEL_PATH")
    database_path = os.getenv("DATABASE_PATH")
    # print(database_path)
    dataset_path, custom_model, isCheckpoint = False, False, False
    ascii_art()
    
    if len(sys.argv) < 2:
        show_error(os.getenv("MODE_NOT_PROVIDED"))
        sys.exit()

    mode = sys.argv[1].lower()

    if mode not in ["live", "camera", "manual"]:
        show_error(os.getenv("MODE_INVALID"))
        sys.exit()

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--model":
            i, model_path = check_argv_file(arg, i, sys.argv)
            custom_model = True
        elif arg == "--help":
            if mode == "live":
                print("Usage: python3 facegsm.py live --host [ip] --target [target_image_path]")
                print("Options:")
                print("  --host: Specify the IP address for Droidcam to stream live camera feed.")
                print("  --target: Folder path of the victim's face for executing FGSM attack.")
                print("  --dataset: Directory containing the dataset to be used for database comparison.")
                sys.exit()
            elif mode == "camera":
                print("Usage: python3 facegsm.py camera --host [ip]")
                print("Options:")
                print("  --host: Specify the IP address for Droidcam to stream live camera feed.")
                print("  --checkpoint: Path to the model checkpoint file to load the trained neural network model.")
                print("  --dataset: Directory containing the dataset to be used for database comparison.")
                sys.exit()
            elif mode == "manual":
                print("Usage: python3 facegsm.py manual --original [original_image_path] -target [target_image_path]")
                print("Options:")
                print("  --original: Folder path of the original's face for executing FGSM attack")
                print("  --target: Folder path of the victim's face for executing FGSM attack")
                print("  --checkpoint: Path to the model checkpoint file to load the trained neural network model.")
                print("  --dataset: Directory containing the dataset to be used for database comparison.")
                sys.exit()
        elif arg == "--original" and mode == "manual":
            i, original_pic_path = check_argv_file(arg, i, sys.argv)
        elif arg == "--target" and (mode == "manual" or mode == "live"):
            i, target_pic_path = check_argv_file(arg, i, sys.argv)
        elif arg == "--dataset":
            # used only if, users want to create new database / first time running the app
            i, dataset_path = check_argv_folder(arg, i, sys.argv)
        elif arg == "--host" and (mode == "camera" or mode == "live"):
            i, ip_droid_cam = check_argv_ip(arg, i, sys.argv)
        elif arg == "--checkpoint":
            isCheckpoint = True
        i += 1

    if custom_model:
        show_info("Using Custom Model...")
    else:
        show_info("Using Default Model...")

    model = load_model(model_path)

    # only accept facenet model
    if model.input_shape[1:3] != (160, 160):
        show_error(os.getenv("MODEL_INVALID"))

    if dataset_path:
        create_json(dataset_path, model)
    elif not dataset_path and not os.path.isfile(database_path):
        print(os.path.isfile(database_path))
        show_error(os.getenv("DATABASE_JSON_NOT_FOUND"))

    if mode == "live":
        handler = LiveCameraClient(target_pic_path, ip_droid_cam, model_path)
        asyncio.run(handler.initialize())
        # LiveCameraClient(ip_droid_cam, model_path, isCheckpoint=True) # always use checkpoint for live mode
    elif mode == "camera":
        VideoCaptureApp(ip_droid_cam, model, isCheckpoint)
    elif mode == "manual":
        attack_adv(original_pic_path, target_pic_path, model, isCheckpoint)


if __name__ == "__main__":
    main()
