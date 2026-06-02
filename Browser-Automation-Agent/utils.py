import cv2
import base64
import numpy as np


def process_image(image_bytes):
    """Convert raw screenshot bytes (PNG) to a base64-encoded JPEG string."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the image bytes.")
    success, buffer = cv2.imencode(".jpg", img)
    if success:
        return base64.b64encode(buffer).decode("utf-8")
    else:
        raise ValueError("Could not process the image.")


if __name__ == "__main__":
    print(process_image("screenshots/after_login.png"))