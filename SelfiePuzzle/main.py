import cv2
import mediapipe as mp
import math
import numpy as np

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
color_names = ["BLUE", "GREEN", "RED"]
current_color_index = 0

canvas = None
xp, yp = 0, 0

print("🎨 Air Canvas Started!")
print("- Draw with your Index Finger ☝️")
print("- Bring Index and Middle fingers together up to switch color 🖕") 
print("- Pinch Index and Thumb together to erase 🤌")
print("- Press 'q' to exit.")

color_switch_ready = True

while True:
    success, frame = cam.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    draw_color = colors[current_color_index]
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            
            x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
            x_middle, y_middle = int(middle_tip.x * w), int(middle_tip.y * h)
            
            pinch_distance = math.hypot(x_index - x_thumb, y_index - y_thumb)
            color_distance = math.hypot(x_middle - x_index, y_middle - y_index)
            
            if pinch_distance < 35:
                cv2.circle(frame, (x_index, y_index), 30, (0, 0, 0), cv2.FILLED)
                cv2.circle(canvas, (x_index, y_index), 30, (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, "ERASER", (x_index - 40, y_index - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                xp, yp = 0, 0
                
            elif color_distance < 35 and y_middle < hand_landmarks.landmark[9].y * h:
                if color_switch_ready:
                    current_color_index = (current_color_index + 1) % len(colors)
                    color_switch_ready = False
                xp, yp = 0, 0
                
            else:
                color_switch_ready = True
                
                cv2.circle(frame, (x_index, y_index), 10, draw_color, cv2.FILLED)
                
                if xp == 0 and yp == 0:
                    xp, yp = x_index, y_index
                    
                cv2.line(canvas, (xp, yp), (x_index, y_index), draw_color, 8)
                xp, yp = x_index, y_index
                
    else:
        xp, yp = 0, 0

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)
    
    cv2.rectangle(frame, (10, 10), (250, 60), (30, 30, 30), -1)
    cv2.rectangle(frame, (20, 20), (40, 40), draw_color, -1)
    cv2.putText(frame, f"COLOR: {color_names[current_color_index]}", (55, 42), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Magic Air Canvas", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()