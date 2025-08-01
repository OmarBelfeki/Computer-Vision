import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_index = int(hand_landmarks.landmark[8].x * w)
            y_index = int(hand_landmarks.landmark[8].y * h)

            mouse_x = int((x_index / w) * screen_w)
            mouse_y = int((y_index / h) * screen_h)
            pyautogui.moveTo(mouse_x, mouse_y)

            x_thumb = int(hand_landmarks.landmark[4].x * w)
            y_thumb = int(hand_landmarks.landmark[4].y * h)

            cv2.circle(frame, (x_index, y_index), 10, (0, 255, 0), -1)
            cv2.circle(frame, (x_thumb, y_thumb), 10, (0, 0, 255), -1)

            distance = ((x_thumb - x_index) ** 2 + (y_thumb - y_index) ** 2) ** 0.5

            if distance < 30:
                pyautogui.click()
                pyautogui.sleep(0.2)

    cv2.imshow("Hand Gesture Mouse", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
