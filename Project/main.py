import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import winsound
import threading

# Load model
model = load_model("eye_model.h5")

# MediaPipe setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR calculation
def calculate_EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# 🔊 Default system alarm
def play_alarm():
    for i in range(5):
        winsound.Beep(2500, 500)  # frequency, duration

# Variables
eye_buffer = deque(maxlen=5)
blink_count = 0
closed_frames = 0
alarm_on = False

EAR_THRESHOLD = 0.25

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []

            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((x, y))

            left_eye = np.array([landmarks[i] for i in LEFT_EYE])
            right_eye = np.array([landmarks[i] for i in RIGHT_EYE])

            # Draw eye points
            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # EAR
            left_EAR = calculate_EAR(left_eye)
            right_EAR = calculate_EAR(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # Crop eye
            x_min = min(left_eye[:,0])
            x_max = max(left_eye[:,0])
            y_min = min(left_eye[:,1])
            y_max = max(left_eye[:,1])

            eye_img = frame[y_min:y_max, x_min:x_max]

            if eye_img.size != 0:
                eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
                eye_img = cv2.resize(eye_img, (64, 64))
                eye_img = eye_img / 255.0
                eye_img = eye_img.reshape(1, 64, 64, 1)

                pred = model.predict(eye_img, verbose=0)
                eye_state = np.argmax(pred)

                # Smoothing
                eye_buffer.append(eye_state)
                final_state = round(sum(eye_buffer) / len(eye_buffer))

                # Fatigue logic
                if avg_EAR < EAR_THRESHOLD:
                    closed_frames += 1

                    if closed_frames > 20:
                        # 🔥 TOP RIGHT CORNER
                        cv2.putText(frame, "DROWSY!", (w - 250, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                        if not alarm_on:
                            alarm_on = True
                            threading.Thread(target=play_alarm).start()

                else:
                    if closed_frames > 2:
                        blink_count += 1

                    closed_frames = 0
                    alarm_on = False

                status = "OPEN" if final_state == 1 else "CLOSED"

                # Display
                cv2.putText(frame, f"Eye: {status}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                cv2.putText(frame, f"Blinks: {blink_count}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Eye Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()