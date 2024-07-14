import os
import warnings
import logging

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import re
import sys
import asyncio
from utils.interface import VideoCaptureApp
from utils.live_client import LiveCameraClient
from dotenv import load_dotenv
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
    print("Usage: python3 facegsm.py static/live/capture/database/--help")
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
        print("  --target: Folder path of the victim's face for executing FGSM attack.")
        print("  --checkpoint: Use the saved checkpoint in checkpoints folder")
        print(
            "  --model: Path to the custom model file to load the trained neural network model."
        )
        sys.exit()
    elif mode == "capture":
        print("Usage: python3 facegsm.py capture --host [ip]")
        print("Options:")
        print("  --host: Specify the IP address for Droidcam to capture the image.")
        print("  --checkpoint: Use the saved checkpoint in checkpoints folder")
        print(
            "  --model: Path to the custom model file to load the trained neural network model."
        )
        sys.exit()
    elif mode == "static":
        print(
            "Usage: python3 facegsm.py static --original [original_image_path] -target [target_image_path]"
        )
        print("Options:")
        print(
            "  --original: Folder path of the original's face for executing FGSM attack"
        )
        print("  --target: Folder path of the victim's face for executing FGSM attack")
        print("  --checkpoint: Use the saved checkpoint in checkpoints folder")
        print(
            "  --model: Path to the custom model file to load the trained neural network model."
        )
        sys.exit()
    elif mode == "database":
        print("Usage: python3 facegsm.py database --dataset [dataset_path]")
        print("Options:")
        print(
            "  --dataset: Folder path of the custom dataset to create the database for FaceGSM."
        )
        sys.exit()


def main():
    model_path = os.getenv("DEFAULT_MODEL_PATH")
    database_path = os.getenv("DATABASE_PATH")
    dataset_path, custom_model, isCheckpoint = False, False, False
    original_pic_path, target_pic_path, url_droid_cam = "", "", ""
    modes = ["static", "live", "capture", "database", "--help"]
    ascii_art()
    check_database(database_path)
    check_outputs_folder()

    model = load_model(model_path)
    input_layer = model.layers[0].input
    output_layer = model.layers[-1].output
    required_size = model.input_shape[1:3]  # input shape size

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
                elif arg == "--model":
                    i = sys.argv.index("--model")
                    try:
                        model_path = check_argv_file(sys.argv[i + 1])
                    except:
                        show_error_arg("NO_VALUE_PROVIDED", arg)
                    custom_model = True
                    try:
                        model = load_model(model_path)
                        required_size = model.input_shape[1:3]  # input shape size
                        model_output_dimension = model.output_shape[
                            1
                        ]  # embeddings size
                    except:
                        show_error("MODEL_INVALID")
        elif "--help" in sys.argv:
            show_help_mode("static")
        else:
            show_error("STATIC_MODE_NEED_ARG")
    elif mode == "capture":
        required_arg = ["--host"]
        has_all_required = all(arg in sys.argv for arg in required_arg)
        if has_all_required:
            for arg in sys.argv:
                if arg == "--host":
                    i = sys.argv.index("--host")
                    try:
                        url_droid_cam = check_argv_url_droidcam(sys.argv[i + 1])
                    except:
                        show_error_arg("NO_VALUE_PROVIDED", arg)
                elif arg == "--checkpoint":
                    isCheckpoint = True
                elif arg == "--model":
                    i = sys.argv.index("--model")
                    try:
                        model_path = check_argv_file(sys.argv[i + 1])
                    except:
                        show_error_arg("NO_VALUE_PROVIDED", arg)
                    custom_model = True
                    try:
                        model = load_model(model_path)
                        required_size = model.input_shape[1:3]  # input shape size
                        model_output_dimension = model.output_shape[
                            1
                        ]  # embeddings size
                    except:
                        show_error("MODEL_INVALID")
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
                elif arg == "--model":
                    i = sys.argv.index("--model")
                    try:
                        model_path = check_argv_file(sys.argv[i + 1])
                    except:
                        show_error_arg("NO_VALUE_PROVIDED", arg)
                    custom_model = True
                    try:
                        model = load_model(model_path)
                        required_size = model.input_shape[1:3]  # input shape size
                        model_output_dimension = model.output_shape[
                            1
                        ]  # embeddings size
                    except:
                        show_error("MODEL_INVALID")
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
                elif arg == "--model":
                    i = sys.argv.index("--model")
                    try:
                        model_path = check_argv_file(sys.argv[i + 1])
                    except:
                        show_error_arg("NO_VALUE_PROVIDED", arg)
                    custom_model = True
                    try:
                        model = load_model(model_path)
                        required_size = model.input_shape[1:3]  # input shape size
                        model_output_dimension = model.output_shape[
                            1
                        ]  # embeddings size
                    except:
                        show_error("MODEL_INVALID")
        elif "--help" in sys.argv:
            show_help_mode("database")
        else:
            show_error("DATABASE_MODE_NEED_ARG")
    else:
        show_error("MODE_INVALID")
        sys.exit()

    if custom_model:
        show_info("Using Custom Model...")
    else:
        show_info("Using Default Model...")

    if mode == "live":
        handler = LiveCameraClient(
            target_pic_path, url_droid_cam, model_path, required_size
        )
        asyncio.run(handler.initialize())
    elif mode == "capture":
        VideoCaptureApp(url_droid_cam, model, isCheckpoint, required_size)
    elif mode == "static":
        attack_adv(
            original_pic_path, target_pic_path, model, required_size, isCheckpoint
        )
    elif mode == "database":
        create_json(dataset_path, model, required_size)


if __name__ == "__main__":
    main()
