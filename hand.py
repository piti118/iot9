import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hlm = mp_hands.HandLandmark
print(mp_hands.HandLandmark.THUMB_CMC)


def count_fingers(hand_landmarks, handedness) -> int:
    # from https://github.com/pdhruv93/computer-vision/blob/main/fingers-count/fingers-count.py
    # we can do better with dot product but it's late night....
    # print(hand_landmarks)
    lm = [lm for lm in hand_landmarks.landmark]
    count = 0

    if lm[hlm.INDEX_FINGER_TIP].y < lm[hlm.INDEX_FINGER_PIP].y:  # Index finger
        count = count + 1
    if lm[hlm.MIDDLE_FINGER_TIP].y < lm[hlm.MIDDLE_FINGER_PIP].y:  # Middle finger
        count = count + 1
    if lm[hlm.RING_FINGER_TIP].y < lm[hlm.RING_FINGER_PIP].y:  # Ring finger
        count = count + 1
    if lm[hlm.PINKY_TIP].y < lm[hlm.PINKY_PIP].y:  # Little finger
        count = count + 1
    hand = handedness.classification[0].label
    if hand == 'Left':
        if lm[hlm.THUMB_TIP].x > lm[hlm.THUMB_MCP].x:
            count = count + 1
    if hand == 'Right':
        if lm[hlm.THUMB_TIP].x < lm[hlm.THUMB_MCP].x:
            count = count + 1

    return count


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # print([x.classification[0].label for x in results.multi_handedness])
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # print(hand_landmarks)
                count = count_fingers(hand_landmarks, handedness)
                lm = [lm for lm in hand_landmarks.landmark]
                h, w, _ = image.shape
                wrist_location = (int(lm[hlm.WRIST].x * w), int(lm[hlm.WRIST].y * h))
                cv2.putText(image, str(count), wrist_location, cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 25)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
