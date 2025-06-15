import dlib
import cv2
import numpy as np
import os
import base64
from PIL import Image
from io import BytesIO

# Load face detector, shape predictor, and face recognizer
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

def read_image_from_b64(b64_string):
    try:
        image_data = base64.b64decode(b64_string.split(',')[1])
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("Error reading image from base64:", e)
        return None

def get_face_descriptor(image):
    faces = detector(image, 1)
    if len(faces) == 0:
        return None
    shape = predictor(image, faces[0])
    return face_rec_model.compute_face_descriptor(image, shape)

def is_face_match(enrollment_no, face_image_b64):
    try:
        # ✅ Load submitted image from base64
        submitted_img = read_image_from_b64(face_image_b64)
        if submitted_img is None:
            return False

        submitted_desc = get_face_descriptor(submitted_img)
        if submitted_desc is None:
            return False

        # ✅ Load reference image
        ref_path = f"face_data/{enrollment_no}.jpg"
        if not os.path.exists(ref_path):
            print(f"No reference face found for {enrollment_no}")
            return False

        ref_img = cv2.imread(ref_path)
        ref_desc = get_face_descriptor(ref_img)
        if ref_desc is None:
            return False

        # ✅ Compare face descriptors (Euclidean distance)
        distance = np.linalg.norm(np.array(submitted_desc) - np.array(ref_desc))
        print(f"Face distance: {distance}")
        return distance < 0.6  # Threshold
    except Exception as e:
        print("Error in is_face_match:", e)
        return False
