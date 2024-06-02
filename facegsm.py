import os
import sys
import re
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


# def check_argv_folder(arg):
#     if os.path.isdir(arg):
#         if len(os.listdir(arg)) == 0:
#             show_error_arg(os.getenv("EMPTY_FOLDER"), arg)
#             sys.exit()
#         else:
#             return True
#     else:
#         show_error_arg(os.getenv("FOLDER_INVALID"), arg)
#         sys.exit()

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

# def check_argv_file(arg):
#     if os.path.isfile(arg):
#             return arg
#     else:
#         show_error_arg(os.getenv("FILE_INVALID"), arg)
#         sys.exit()


def check_argv_url_droidcam(arg, i, argv):
    i += 1
    if i < len(argv):
        pattern = "(http|https):\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,65535}\/video"
        if re.match(pattern, argv[i]):
            return i, argv[i]
        else:
            show_error_arg(os.getenv("IPv4_INVALID"), arg)
            sys.exit()
    else:
        show_error_arg(os.getenv("NO_VALUE_PROVIDED"), arg)
        sys.exit()


# def check_database(database_path):
#     if not os.path.isfile(database_path):
#         show_error(os.getenv("DATABASE_JSON_NOT_FOUND"))
        
def main():
    model_path = os.getenv("DEFAULT_MODEL_PATH")
    database_path = os.getenv("DATABASE_PATH")
    dataset_path, custom_model, isCheckpoint = False, False, False
    original_pic_path, target_pic_path, url_droid_cam = "", "", ""
    ascii_art()

    # facegsm.py manual -> original reference before assingment -> solve cek -> harus ada argumen setelah "manual"
    # facegsm.py manual --ngasal ./path -> gw gatau output
    # facegsm.py manual --original ./path --target -> target reference before assignment -> 

    # check_database(database_path)

    modes = ["live", "camera", "manual", "database"]
    
    if len(sys.argv) < 2:
        show_error(os.getenv("MODE_NOT_PROVIDED"))
        sys.exit()

    mode = sys.argv[1].lower()

    if mode not in modes:
        show_error(os.getenv("MODE_INVALID"))
        sys.exit()

    # if sys.argv[1]:
    #     mode = sys.argv[1].lower()
    #     if mode not in modes:
    #         if mode == "--help":
    #             print("Usage: python3 facegsm.py live/camera/manual/--help")
    #             print("Options:")
    #             print("  live: Live camera feature in FaceGSM includes real-time face recognition and attack capabilities.")
    #             print("  camera: Camer original and target photos in FaceGSM.")
    #             print("  manual: Manual input for FGSM attack in FaceGSM.")
    #             print("  --help: Show help for available options.")
    #         else:
    #             show_error(os.getenv("MODE_INVALID"))
    #             sys.exit()
    #     elif mode == "database":
    #         # required :
    #         # --path ./path_to_folder
    #         if sys.argv[2] and sys.argv[3]:
    #             dataset_path = sys.argv[3]
    #             if check_argv_folder(database_path):
    #                 create_json(dataset_path, model)
    #         else:
    #             show_error_arg(os.getenv("NO_VALUE_PROVIDED"), arg)
    #             sys.exit()
    #     elif mode == "manual":
    #         # required :
    #         # --original ./path
    #         # --target ./path
    #         # optional :
    #         # --checkpoint
    #         # --model
    #         # e.g -> facegsm.py manual --original ./path --target -> while
    #         # input kotor -> facegsm.py manual --original --target ./path
    #         # input kotor -> facegsm.py manual --original
    #         required_arg = ["--original", "--target"]
    #         has_all_required = all(arg in sys.argv for arg in required_arg)
    #         if has_all_required:
    #             for arg in sys.argv:
    #                 if arg == "--original":
    #                     i = sys.argv.index("--original")
    #                     original_pic_path = check_argv_file(sys.argv[i+1])
    #                 elif arg == "--target":
    #                     i = sys.argv.index("--target")
    #                     target_pic_path = check_argv_file(sys.argv[i+1])
    #                 elif arg == "--checkpoint":
    #                     isCheckpoint = True
    #                 elif arg == "--model":
    #                     i = sys.argv.index("--model")
    #                     model_path = check_argv_file(sys.argv[i+1])
    #         else:
                
    #     elif mode == "camera":
    #         # required :
    #         # --host ./path
    #         # --checkpoint ./path
    #         # optional :
    #         # e.g -> facegsm.py camera --original ./path --target -> while
                
    #         return 1
    #     elif mode == "live":
    #         # required :
    #         # --host ./path
    #         # --target ./path
    #         # optional :
    #         # e.g -> facegsm.py camera --host ip --target -> while
            
    #         return 1
    # else:
    #     show_error(os.getenv("MODE_NOT_PROVIDED"))
    #     sys.exit()

    i = 2
    if len(sys.argv) > 2:
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg != None:
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
                    i, url_droid_cam = check_argv_url_droidcam(arg, i, sys.argv)
                elif arg == "--checkpoint":
                    isCheckpoint = True
                else:
                    print("Invalid argument: " + arg)
                    print("Use --help for more information.")
                    sys.exit()
            i += 1
    else:
        print(f"{mode} Mode need arguments, use --help to see the supported arguments.")
        sys.exit()

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
        handler = LiveCameraClient(target_pic_path, url_droid_cam, model_path)
        asyncio.run(handler.initialize())
    elif mode == "camera":
        VideoCaptureApp(url_droid_cam, model, isCheckpoint)
    elif mode == "manual":
        attack_adv(original_pic_path, target_pic_path, model, isCheckpoint)


if __name__ == "__main__":
    main()
