#Using code from
#https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
#https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_blue = np.array([101,100,100])
    upper_blue = np.array([104,255,255])
    
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    res = cv.bitwise_and(frame,frame, mask= mask)
    gray_mask = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

    #create the contour
    ret,thresh = cv.threshold(gray_mask,127,255,cv.THRESH_BINARY)
    contours = cv.findContours(thresh, 1, 2)[-2]

    #create the rectangle
    for c in contours:
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(frame,[box],0,(0,0,255),2)

    # Bitwise-AND mask and original image
    cv.imshow('frame',frame)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
