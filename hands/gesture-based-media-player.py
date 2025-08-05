import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
video = cv2.VideoCapture("/home/omarbelfeki/Desktop/learn/Build an E-commerce Website with Next.js_ Tailwind CSS_ Zustand _ Stripe – Project Overview(1080P_60FPS).mp4")


def detect_gesture(landmark):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    fingers = []

    if landmark[thumb_tip].x < landmark[thumb_tip - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for tip in finger_tips:
        if landmark[tip].y < landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)


prev_x = None
swipe = ""
paused = True
frame_pos = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_lms.landmark
            finger_count = detect_gesture(landmarks)

            curr_x = landmarks[0].x

            # Detect swipe1
            curr_x = landmarks[0].x
            if prev_x is not None:
                if curr_x - prev_x > 0.05:
                    swipe = "right"
                elif prev_x - curr_x > 0.05:
                    swipe = "left"
            prev_x = curr_x

            if finger_count == 5:
                paused = False
            elif finger_count == 0:
                paused = True

    if not paused:
        ret2, video_frame = video.read()
        if not ret2:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
    else:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret2, video_frame = video.read()

    if swipe == "right":
        frame_pos += 30
        swipe = ""
    elif swipe == "left":
        frame_pos = -30
        if frame_pos < 0: frame_pos = 0
        swipe = ""

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    frame_pos = int(video.get(cv2.CAP_PROP_POS_FRAMES))

    if ret2:
        video_frame = cv2.resize(video_frame, (640, 360))
        cv2.imshow("Media Player", video_frame)

    cv2.imshow("webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
