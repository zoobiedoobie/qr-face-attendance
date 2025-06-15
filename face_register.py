import cv2
import dlib
import numpy as np
import os

# Load dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

# Ask for enrollment number
enroll_no = input("Enter enrollment number: ")

# Create folder if not exists
if not os.path.exists("face_data"):
    os.makedirs("face_data")

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 's' to save face, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Register Face", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        if len(faces) == 1:
            shape = sp(frame, faces[0])
            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            np.save(f"face_data/{enroll_no}.npy", np.array(face_descriptor))
            print("Face encoding saved.")
            break
        else:
            print("Make sure only one face is visible!")
    elif key == ord('q'):
        print("Aborted.")
        break

cap.release()
cv2.destroyAllWindows()
