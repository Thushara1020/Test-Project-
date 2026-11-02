

import cv2
import mediapipe as mp
import math

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 
circles = [
    {"x": 150, "y": 200, "r": 40, "color": (0, 0, 255)},   
    {"x": 300, "y": 200, "r": 40, "color": (0, 255, 0)}, 
    {"x": 450, "y": 200, "r": 40, "color": (255, 0, 0)}   
]

selected_circle = None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    h, w, c = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    for circle in circles:
        cv2.circle(image, (circle["x"], circle["y"]), circle["r"], circle["color"], -1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Hand tracking lines ඇඳීම
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]

            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)

            # 🎯 ඇඟිලි දෙකේ මැද ලක්ෂ්‍යය (Center point) සෙවීම
            cx, cy = (tx + ix) // 2, (ty + iy) // 2
            cv2.circle(image, (cx, cy), 6, (255, 255, 255), -1) # මැද ලක්ෂ්‍යය සුදු පාටින්

            # 📏 ඇඟිලි දෙක අතර දුර (Distance) ගණනය කිරීම
            distance = math.hypot(ix - tx, iy - ty)


            if distance < 40:

                if selected_circle is None:
                    for circle in circles:
  
                        dist_to_circle = math.hypot(cx - circle["x"], cy - circle["y"])
                        if dist_to_circle < circle["r"]: 
                            selected_circle = circle
                            break
                

                if selected_circle:
                    selected_circle["x"] = cx
                    selected_circle["y"] = cy
            else:

                selected_circle = None

    cv2.imshow('Drag and Drop Shapes', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()