import cv2
import time
import pygame
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# Initialize pygame for alarm
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# EAR and MAR thresholds
EAR_THRESHOLD = 0.25
EYE_CLOSE_DURATION = 3  # seconds
MAR_THRESHOLD = 0.9  # only wide-open mouth considered
MOUTH_OPEN_FRAMES = 60  # ~2 seconds at 30 FPS

# MediaPipe landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
MOUTH_POINTS = [UPPER_LIP, LOWER_LIP, LEFT_MOUTH, RIGHT_MOUTH]

# State variables
eye_closed_start = None
mouth_counter = 0
alarm_on = False

# EAR calculation
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# MAR calculation
def calculate_MAR(landmarks, w, h):
    upper_lip = landmarks[UPPER_LIP]
    lower_lip = landmarks[LOWER_LIP]
    left_mouth = landmarks[LEFT_MOUTH]
    right_mouth = landmarks[RIGHT_MOUTH]

    vertical = distance.euclidean(
        (int(upper_lip.x * w), int(upper_lip.y * h)),
        (int(lower_lip.x * w), int(lower_lip.y * h))
    )
    horizontal = distance.euclidean(
        (int(left_mouth.x * w), int(left_mouth.y * h)),
        (int(right_mouth.x * w), int(right_mouth.y * h))
    )

    return vertical / horizontal if horizontal != 0 else 0

print("📷 Starting webcam. Press 'q' to quit.")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame grab failed.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    eyes_closed = False
    mouth_open = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w = frame.shape[:2]
            landmarks = face_landmarks.landmark

            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

            left_ear = calculate_EAR(left_eye)
            right_ear = calculate_EAR(right_eye)
            ear = (left_ear + right_ear) / 2.0

            mar = calculate_MAR(landmarks, w, h)

            print(f"EAR: {ear:.2f}, MAR: {mar:.2f}")  # Debug log

            # Drowsiness: Eyes closed logic
            if ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif time.time() - eye_closed_start >= EYE_CLOSE_DURATION:
                    eyes_closed = True
            else:
                eye_closed_start = None

            # Yawning: Mouth open logic
            if mar > MAR_THRESHOLD:
                mouth_counter += 1
                if mouth_counter >= MOUTH_OPEN_FRAMES:
                    mouth_open = True
            else:
                mouth_counter = 0

            # Draw eyes (green) and mouth (yellow)
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # green dots

            for i in MOUTH_POINTS:
                mx = int(landmarks[i].x * w)
                my = int(landmarks[i].y * h)
                cv2.circle(frame, (mx, my), 2, (0, 255, 255), -1)  # yellow dots

    # ALERTS
    if eyes_closed:
        if not alarm_on:
            pygame.mixer.music.play(-1)
            alarm_on = True
        cv2.putText(frame, "😴 DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    elif mouth_open:
        if not alarm_on:
            pygame.mixer.music.play(-1)
            alarm_on = True
        cv2.putText(frame, "🥱 YAWN ALERT!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    else:
        if alarm_on:
            pygame.mixer.music.stop()
            alarm_on = False

    cv2.imshow("MediaPipe Drowsiness & Yawn Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
