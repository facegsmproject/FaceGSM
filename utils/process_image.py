import cv2
import matplotlib.pyplot as plt
from utils.error_handling import show_error

output_path = "./outputs/"


def rect_gen_live(frame, faces, person_name, confidence_level, image_size=(160, 160)):
    for x, y, w, h in faces:

        text = f"{person_name}: {confidence_level}%"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return frame


def rect_gen(person_name, cos_sim, frame, faces, filename):
    try:
        for face in faces:
            x, y, w, h = face["box"]
            text = f"{person_name}: {cos_sim:.2f}%"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 204), 2)
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 255, 204),
                1,
            )  # frame, text, location, font, font size, color, thickness
            cv2.imwrite(output_path + filename + ".png", frame)
    except:
        show_error("Error generating rectangle!")


def save_image(image, filename):
    try:
        cv2.imwrite(output_path + filename + ".png", image)
    except:
        show_error("Error saving image!")


def show_image(image, title):
    try:
        image = image[0] if len(image.shape) == 4 else image
        plt.imshow(image)
        plt.title(title)
        plt.show()
    except:
        show_error("Error showing image!")


def show_save_perturbation_layer(image, title):
    try:
        image = image[0] if len(image.shape) == 4 else image
        plt.imshow(image)
        plt.title(title)
        plt.savefig(output_path + title + ".png")
        plt.show()
    except:
        show_error("Error showing image!")


def create_padding(adv_image, faces):
    try:
        x, y, width, height = faces[0]["box"]
        face_img = adv_image[y : y + height, x : x + width, :]
        face_img = cv2.copyMakeBorder(
            face_img,
            200,
            200,
            200,
            200,
            cv2.BORDER_CONSTANT,  # border type
            value=[255, 255, 255],
        )
        return face_img
    except:
        show_error("Error creating padding!")
