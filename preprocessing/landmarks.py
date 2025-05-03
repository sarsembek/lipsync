import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def extract_mouth_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    mouth_sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if faces:
            shape = predictor(gray, faces[0])
            mouth = [(shape.part(i).x, shape.part(i).y) for i in range(48, 68)]
            mouth_sequence.append(mouth)

    cap.release()
    return mouth_sequence  # List of 20 (x, y) pairs per frame
