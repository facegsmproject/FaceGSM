import os
import warnings
import logging

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import re
import sys
import asyncio
from dotenv import load_dotenv
from utils.interface import VideoCaptureApp
from utils.live_client import LiveCameraClient
from keras.models import load_model
from utils.db_generator import create_json
from utils.adv_generator import attack_adv
from utils.ascii_art import ascii_art
from utils.error_handling import *


load_dotenv()


def check_argv_folder(arg):
    if os.path.isdir(arg):
        if len(os.listdir(arg)) == 0:
            show_error("EMPTY_FOLDER")
        else:
            return arg
    else:
        show_error("FOLDER_INVALID")


def check_argv_file(arg):
    if os.path.isfile(arg):
        return arg
    else:
        show_error("FILE_INVALID")


def check_argv_url_droidcam(arg):
    pattern = "(http|https):\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,65535}\/video"
    if re.match(pattern, arg):
        return arg
    else:
        show_error("URL_DROIDCAM_INVALID")


def check_database(database_path):
    if not os.path.isfile(database_path):
        with open(database_path, "w") as f:
            show_info("Database not found. Creating a new blank database...")
            f.write('{"predictions":{}}')


def check_outputs_folder():
    if not os.path.isdir("./outputs"):
        os.mkdir("./outputs")


def show_help():
    print("Usage: python3 facegsm.py [ static | live | capture | database ] --help")
    print("Options:")
    print("  static: Static input for FGSM attack in FaceGSM.")
    print("  capture: Capture original and target photos in FaceGSM.")
    print(
        "  live: Live camera feature in FaceGSM includes real-time face recognition and attack capabilities."
    )
    print("  database: Create a database based on datasets for FaceGSM.")
    print("  --help: Show help for available options.")
    sys.exit()


def show_help_mode(mode):
    if mode == "live":
        print("Usage: python3 facegsm.py live --host [ip] --target [target_image_path]")
        print("Options:")
        print(
            "  --host: Specify the IP address for Droidcam to stream live camera feed."
        )
        print("  --target: File path of the victim's face for executing FGSM attack.")
        print("  --checkpoint: Use the saved checkpoint in checkpoints folder")
        sys.exit()
    elif mode == "capture":
        print("Usage: python3 facegsm.py capture --host [ip]")
        print("Options:")
        print("  --host: Specify the IP address for Droidcam to capture the image.")
        print("  --target: File path of the victim's face for executing FGSM attack.")
        print("  --checkpoint: Use the saved checkpoint in checkpoints folder")
        sys.exit()
    elif mode == "static":
        print(
            "Usage: python3 facegsm.py static --original [original_image_path] --target [target_image_path]"
        )
        print("Options:")
        print(
            "  --original: File path of the original's face for executing FGSM attack"
        )
        print("  --target: File path of the victim's face for executing FGSM attack")
        print("  --checkpoint: Use the saved checkpoint in checkpoints folder")
        sys.exit()
    elif mode == "database":
        print("Usage: python3 facegsm.py database --dataset [dataset_path]")
        print("Options:")
        print(
            "  --dataset: Folder path of the custom dataset to create the database for FaceGSM."
        )
        sys.exit()


def main():
    model_path = os.getenv("MODEL_PATH")
    database_path = os.getenv("DATABASE_PATH")
    dataset_path, isCheckpoint = (
        False,
        False,
    )
    original_pic_path, target_pic_path, url_droid_cam = "", "", ""
    modes = ["static", "live", "capture", "database", "--help"]
    ascii_art()
    check_database(database_path)
    check_outputs_folder()

    try:
        mode = sys.argv[1].lower()
    except:
        show_error("MODE_NOT_PROVIDED")

    if mode == "--help":
        show_help()
    elif mode == "static":
        required_arg = ["--original", "--target"]
        has_all_required = all(arg in sys.argv for arg in required_arg)
        if has_all_required:
            for arg in sys.argv:
                if arg == "--original":
                    i = sys.argv.index("--original")
                    try:
                        original_pic_path = check_argv_file(sys.argv[i + 1])
                    except:
                        show_error_arg("NO_VALUE_PROVIDED", arg)
                elif arg == "--target":
                    i = sys.argv.index("--target")
                    try:
                        target_pic_path = check_argv_file(sys.argv[i + 1])
                    except:
                        show_error_arg("NO_VALUE_PROVIDED", arg)
                elif arg == "--checkpoint":
                    isCheckpoint = True
        elif "--help" in sys.argv:
            show_help_mode("static")
        else:
            show_error("STATIC_MODE_NEED_ARG")
    elif mode == "capture":
        required_arg = ["--host", "--target"]
        has_all_required = all(arg in sys.argv for arg in required_arg)
        if has_all_required:
            for arg in sys.argv:
                if arg == "--host":
                    i = sys.argv.index("--host")
                    try:
                        url_droid_cam = check_argv_url_droidcam(sys.argv[i + 1])
                    except:
                        show_error_arg("NO_VALUE_PROVIDED", arg)
                elif arg == "--target":
                    i = sys.argv.index("--target")
                    try:
                        target_pic_path = check_argv_file(sys.argv[i + 1])
                    except:
                        show_error_arg("NO_VALUE_PROVIDED", arg)
                elif arg == "--checkpoint":
                    isCheckpoint = True
        elif "--help" in sys.argv:
            show_help_mode("capture")
        else:
            show_error("CAPTURE_MODE_NEED_ARG")
    elif mode == "live":
        required_arg = ["--host", "--target"]
        has_all_required = all(arg in sys.argv for arg in required_arg)
        if has_all_required:
            for arg in sys.argv:
                if arg == "--host":
                    i = sys.argv.index("--host")
                    try:
                        url_droid_cam = check_argv_url_droidcam(sys.argv[i + 1])
                    except:
                        show_error_arg("NO_VALUE_PROVIDED", arg)
                elif arg == "--target":
                    i = sys.argv.index("--target")
                    try:
                        target_pic_path = check_argv_file(sys.argv[i + 1])
                    except:
                        show_error_arg("NO_VALUE_PROVIDED", arg)
                elif arg == "--checkpoint":
                    isCheckpoint = True
        elif sys.argv[2] == "--help":
            show_help_mode("live")
        else:
            show_error("LIVE_MODE_NEED_ARG")
    elif mode == "database":
        required_arg = ["--dataset"]
        has_all_required = all(arg in sys.argv for arg in required_arg)
        if has_all_required:
            for arg in sys.argv:
                if arg == "--dataset":
                    i = sys.argv.index("--dataset")
                    try:
                        dataset_path = check_argv_folder(sys.argv[i + 1])
                    except:
                        show_error_arg("NO_VALUE_PROVIDED", arg)
        elif "--help" in sys.argv:
            show_help_mode("database")
        else:
            show_error("DATABASE_MODE_NEED_ARG")
    else:
        show_error("MODE_INVALID")
        sys.exit()

    model = load_model(model_path)
    input_layer = model.layers[0].input
    output_layer = model.layers[-1].output
    required_size = model.input_shape[1:3]  # input shape size

    model_name = model_path.split("/")[-1]
    show_info(f"Using {model_name} model")

    if mode == "live":
        if os.path.isdir("./checkpoints") and len(os.listdir("./checkpoints")) != 0:
            show_info(
                "Checkpoint found. Do you want to use it or delete it? [y (use) / n (delete)]"
            )
            answer = input("Answer: ").lower()
            if answer == "n":
                for file in os.listdir("./checkpoints"):
                    os.remove(f"./checkpoints/{file}")
                show_info("Checkpoint deleted.")
            elif answer == "y":
                pass
            else:
                show_error("INVALID_CHECKPOINT_ANSWER")
        handler = LiveCameraClient(
            target_pic_path, url_droid_cam, model_path, required_size
        )
        asyncio.run(handler.initialize())
    elif mode == "capture":
        VideoCaptureApp(
            target_pic_path, url_droid_cam, model, isCheckpoint, required_size
        )
    elif mode == "static":
        attack_adv(
            original_pic_path, target_pic_path, model, required_size, isCheckpoint
        )
    elif mode == "database":
        create_json(dataset_path, model, required_size)


if __name__ == "__main__":
    main()
