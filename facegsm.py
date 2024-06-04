import os
import sys
import re
import asyncio
from utils.interface import VideoCaptureApp
from utils.live_client import LiveCameraClient
from dotenv import load_dotenv
from keras.models import load_model
from utils.db_generator import create_json
from utils.adv_generator import attack_adv
from utils.ascii_art import ascii_art
from utils.error_handling import *


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
        show_error("DATABASE_JSON_NOT_FOUND")


def check_outputs_folder():
    if not os.path.isdir("./outputs"):
        os.mkdir("./outputs")


def show_help():
    print("Usage: python3 facegsm.py live/camera/manual/--help")
    print("Options:")
    print(
        "  live: Live camera feature in FaceGSM includes real-time face recognition and attack capabilities."
    )
    print("  camera: Camera original and target photos in FaceGSM.")
    print("  manual: Manual input for FGSM attack in FaceGSM.")
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
    elif mode == "camera":
        print("Usage: python3 facegsm.py camera --host [ip]")
        print("Options:")
        print(
            "  --host: Specify the IP address for Droidcam to stream live camera feed."
        )
        print("  --checkpoint: Use the saved checkpoint in checkpoints folder")
        print(
            "  --model: Path to the custom model file to load the trained neural network model."
        )
        sys.exit()
    elif mode == "manual":
        print(
            "Usage: python3 facegsm.py manual --original [original_image_path] -target [target_image_path]"
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


def main():
    model_path = os.getenv("DEFAULT_MODEL_PATH")
    database_path = os.getenv("DATABASE_PATH")
    dataset_path, custom_model, isCheckpoint = False, False, False
    original_pic_path, target_pic_path, url_droid_cam = "", "", ""
    modes = ["live", "camera", "manual", "database", "--help"]
    ascii_art()
    check_database(database_path)
    check_outputs_folder()

    model = load_model(model_path)

    try:
        mode = sys.argv[1].lower()
    except:
        show_error("MODE_NOT_PROVIDED")

    if mode == "--help":
        show_help()
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
                    create_json(dataset_path, model)
        else:
            show_error("DATABASE_MODE_NEED_ARG")
    elif mode == "manual":
        try:
            if sys.argv[2] == "--help":
                show_help_mode("manual")
        except:
            show_error("MANUAL_MODE_NEED_ARG")

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
                    model = load_model(model_path)
        else:
            show_error("MANUAL_MODE_NEED_ARG")
    elif mode == "camera":
        try:
            if sys.argv[2] == "--help":
                show_help_mode("camera")
        except:
            show_error("CAMERA_MODE_NEED_ARG")

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
                    model = load_model(model_path)
        else:
            show_error("CAMERA_MODE_NEED_ARG")
    elif mode == "live":
        try:
            if sys.argv[2] == "--help":
                show_help_mode("live")
        except:
            show_error("LIVE_MODE_NEED_ARG")

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
                    model = load_model(model_path)
        else:
            show_error("LIVE_MODE_NEED_ARG")
    else:
        show_error("MODE_INVALID")
        sys.exit()

    if custom_model:
        show_info("Using Custom Model...")
    else:
        show_info("Using Default Model...")

    # only accept facenet model with input shape 160x160
    if model.input_shape[1:3] != (160, 160):
        show_error("MODEL_INVALID")

    if mode == "live":
        handler = LiveCameraClient(target_pic_path, url_droid_cam, model_path)
        asyncio.run(handler.initialize())
    elif mode == "camera":
        VideoCaptureApp(url_droid_cam, model, isCheckpoint)
    elif mode == "manual":
        attack_adv(original_pic_path, target_pic_path, model, isCheckpoint)


if __name__ == "__main__":
    main()
