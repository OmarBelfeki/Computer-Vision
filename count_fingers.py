import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

finger_tips = [4, 8, 12, 16, 20]

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            lm_list = []
            h, w, _ = frame.shape
            for idd, lm in enumerate(hand_landmark.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            fingers_up = []

            if lm_list[finger_tips[0]][0] > lm_list[finger_tips[0]-1][0]:
                fingers_up.append(0)
            else:
                fingers_up.append(1)

            for tip in finger_tips[1:]:
                if lm_list[tip][1] < lm_list[tip-2][1]:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            total_fingers = fingers_up.count(1)

            cv2.putText(
                frame,
                f'Fingers: {total_fingers}',
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("finger count", frame)
    if cv2.waitKey(1) & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()

