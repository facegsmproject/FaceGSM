import sys

modes = ["static", "live", "capture", "database", "--help"]

error_list = {
    "MODE_NOT_PROVIDED": f"Mode is not provided. Mode should be one of {modes}",
    "MODE_INVALID": f"Mode is invalid. Mode should be one of {modes}",
    "DATABASE_JSON_NOT_FOUND": "Database not found. Please provide a dataset path to create database",
    "MODEL_INVALID": "The model provided is not a valid model. Please use a valid h5 embeddings model",
    "NO_VALUE_PROVIDED": "No value provided. Please provide a value",
    "FOLDER_INVALID": "Folder path is invalid. Please provide a valid folder path",
    "EMPTY_FOLDER": "Folder is empty. Please provide a folder with images",
    "FILE_INVALID": "File path is invalid. Please provide a valid file path",
    "URL_DROIDCAM_INVALID": "Droid Cam URL address is invalid. Please provide a valid Droid Cam URL address",
    "DATABASE_MODE_NEED_ARG": "Database Mode need arguments. Usage: python3 facegsm.py database --dataset [dataset_path]",
    "FACE_NEITHER_FILE_NOR_IMAGE_ARRAY": "Face is neither a file nor an image array",
    "FACE_NOT_FOUND": "Face not found or error extracting face!",
    "NO_MATCH_FOUND": "No match found!",
    "ERROR_PREPROCESSING": "Error in preprocessing input",
    "ERROR_RECTANGLE": "Error in generating rectangle",
    "ERROR_SAVING_IMAGE": "Error in saving image",
    "ERROR_SHOWING_IMAGE": "Error in showing image",
    "ERROR_CREATING_PADDING": "Error in creating padding",
    "STATIC_MODE_NEED_ARG": "Static Mode need arguments. Usage: python3 facegsm.py static --original [original_pic_path] --target [target_pic_path]",
    "CAPTURE_MODE_NEED_ARG": "Capture Mode need arguments. Usage: python3 facegsm.py capture --target [target_pic_path] --host [droid_cam_url]",
    "LIVE_MODE_NEED_ARG": "Live Mode need arguments. Usage: python3 facegsm.py live --host [droid_cam_url] --target [target_pic_path]",
    "CLIENT_DISCONNECTED": "Client disconnected",
}


def show_error(error_code, exit=True):
    e = error_list.get(error_code)
    print("[!] Error: ", e)
    if exit:
        sys.exit()


def show_error_arg(error_code, arg):
    e = error_list.get(error_code)
    print(f"[!] Error: {e} for {arg} argument")
    sys.exit()


def show_info(message):
    print(f"[+] {message}")
