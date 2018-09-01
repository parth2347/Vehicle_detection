import cv2
import numpy as np
cap = cv2.VideoCapture('video.avi')

car_cascade = cv2.CascadeClassifier('D:/python/cars1.xml')

while True:
    ret,frames = cap.read()
    gray = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(frames,1.1,1)

    for(x,y,z,w) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+z),(0,0,255),2)

    cv2.imshow('image',frames)

    if cv2.waitKey(33) == 27:
        break
cap.release()    
cv2.destroyAllWindows()

    
    
