import cv2
import mediapipe as mp
from typing import NamedTuple
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

TIME_TO_COUNT = 30

answer = ""
prev_key_count = 0
prev_key = ""

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():

    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    h = len(image)
    w = len(image[1])
    #w: 540-740, h:260-460
    #print("Box Coords: w: ", w//2 - 100, "h: ", h//2 - 100)
   
    #mid_w = w // 7
    mid_w = w // 2

    i = 0
    while i < w:        
        cv2.rectangle(image,(i,h//2-100),(i+mid_w,h//2+100),(0,0,255),2)
        crop = image[h//2 - 100 : h//2 + 100, i : i  + mid_w]   
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) 
        img = crop.reshape((crop.shape[0] * crop.shape[1],3))
        i += mid_w
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #Get the pointer finger coordinates
    p_x = 0.0
    p_y = 0.0
    p_z = 0.0
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
          i = 0
          for landmark in enumerate(hand_landmarks.landmark):  
              if i == 8:
                  p_x = landmark[1].x
                  p_y = landmark[1].y
                  p_z = landmark[1].z
                  break
              i += 1

    #print("x: ", p_x*w, "y: ", p_y*h)
    '''
    key = 'C'
    in_h = False
    if p_y*h < (h//2 + 100) and p_y*h > (h//2 - 100):
        in_h = True
        p_x *= w
        if p_x < mid_w*2 and p_x > mid_w:
            key = 'A'
        if p_x < mid_w*3 and p_x > mid_w*2:
            key = 'G'
        if p_x < mid_w*4 and p_x > mid_w*3:
            key = 'F'
        if p_x < mid_w*5 and p_x > mid_w*4:
            key = 'E'
        if p_x < mid_w*6 and p_x > mid_w*5:
            key = 'D'
        print(key)
    '''
    key = 'd'
    in_h = False
    if p_y*h < (h//2 + 100) and p_y*h > (h//2 - 100):
        in_h = True
        p_x *= w
        if p_x < mid_w*2 and p_x > mid_w:
            key = 'f'
        
        if prev_key == "":
            prev_key = key
            prev_key_count = 1
        elif key == prev_key:
            prev_key_count = prev_key_count + 1
        elif key != prev_key:
            prev_key_count = 1
            prev_key = key

        if prev_key_count == TIME_TO_COUNT:
            answer  = answer + key
            prev_key_count = 0
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

print(answer)

cap.release()
