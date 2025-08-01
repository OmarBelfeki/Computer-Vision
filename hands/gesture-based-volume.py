import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

prev_distance = 0
threshold = 10
last_mute_time = 0
mute_cooldown = 1 

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)   # Thumb
            x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)   # Index
            x3, y3 = int(hand_landmarks.landmark[12].x * w), int(hand_landmarks.landmark[12].y * h) # Middle
            x4, y4 = int(hand_landmarks.landmark[16].x * w), int(hand_landmarks.landmark[16].y * h) # Ring
            x5, y5 = int(hand_landmarks.landmark[20].x * w), int(hand_landmarks.landmark[20].y * h) # Pinky

            cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (255, 0, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            distance = np.hypot(x2 - x1, y2 - y1)

            fingers = [y2 < y1, y3 < y1, y4 < y1, y5 < y1]  # True = finger up
            finger_count = fingers.count(True)

            if finger_count == 0 and time.time() - last_mute_time > mute_cooldown:
                pyautogui.press("volumemute")
                last_mute_time = time.time()

            if prev_distance != 0:
                if distance - prev_distance > threshold:
                    pyautogui.press("volumeup")
                elif prev_distance - distance > threshold:
                    pyautogui.press("volumedown")

            prev_distance = distance

            # Draw volume bar (mapped to distance range)
            vol_bar = int(np.interp(distance, [20, 200], [400, 150]))
            cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 2)
            cv2.rectangle(frame, (50, vol_bar), (85, 400), (0, 255, 0), -1)
            cv2.putText(frame, "Volume", (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Gesture Volume Control PRO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
